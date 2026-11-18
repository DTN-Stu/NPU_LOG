// =============================================================================
// Module: snax_simple_dma.sv
// Description: Multi-Port Simple valid/ready Gearbox DMA between 512-bit External
//              DRAM and TCDM_PORTS x 72-bit TCDM Masters (default 4 ports = 288b/cyc).
//              - No AXI overhead: simple req/rsp valid/ready handshakes.
//              - Pipelined 1024-bit (2x512b) buffer to hide DRAM read latency.
//              - 4-port holding registers for 100% in-order, conflict-safe writes.
//              - dir[0] = 0: DRAM -> TCDM, dir[0] = 1: TCDM -> DRAM (port 0)
//              - dir[1] = 0: 16-bank weight layout (banks 0..15), dir[1] = 1: linear
// =============================================================================
`timescale 1ns/1ps

module snax_simple_dma #(
    parameter int unsigned ADDR_WIDTH  = 32,
    parameter int unsigned TCDM_DATA_W = 72,
    parameter int unsigned TCDM_STRB_W = 9,
    parameter int unsigned TCDM_PORTS  = 4,
    parameter int unsigned DRAM_DATA_W = 512,
    parameter int unsigned DRAM_STRB_W = DRAM_DATA_W / 8
)(
    input  logic                                    clk,
    input  logic                                    rst_n,

    // Descriptor Slave Interface
    input  logic                                    desc_valid_i,
    output logic                                    desc_ready_o,
    input  logic [1:0]                              desc_dir_i,
    input  logic [ADDR_WIDTH-1:0]                   desc_dram_addr_i,
    input  logic [ADDR_WIDTH-1:0]                   desc_tcdm_addr_i,
    input  logic [31:0]                             desc_size_bytes_i,
    output logic                                    desc_done_o,

    // TCDM Master Interfaces (Port 0..TCDM_PORTS-1)
    output logic [TCDM_PORTS-1:0]                   tcdm_req_valid_o,
    input  logic [TCDM_PORTS-1:0]                   tcdm_req_ready_i,
    output logic [TCDM_PORTS-1:0][ADDR_WIDTH-1:0]   tcdm_req_addr_o,
    output logic [TCDM_PORTS-1:0]                   tcdm_req_write_o,
    output logic [TCDM_PORTS-1:0][TCDM_DATA_W-1:0]  tcdm_req_wdata_o,
    output logic [TCDM_PORTS-1:0][TCDM_STRB_W-1:0]  tcdm_req_strb_o,
    input  logic [TCDM_PORTS-1:0]                   tcdm_rsp_valid_i,
    input  logic [TCDM_PORTS-1:0][TCDM_DATA_W-1:0]  tcdm_rsp_rdata_i,

    // External DRAM Master Interface (512-bit simple valid/ready)
    output logic                                    dram_req_valid_o,
    input  logic                                    dram_req_ready_i,
    output logic [ADDR_WIDTH-1:0]                   dram_req_addr_o,
    output logic                                    dram_req_write_o,
    output logic [DRAM_DATA_W-1:0]                  dram_req_wdata_o,
    output logic [DRAM_STRB_W-1:0]                  dram_req_strb_o,
    input  logic                                    dram_rsp_valid_i,
    input  logic [DRAM_DATA_W-1:0]                  dram_rsp_rdata_i,

    // Telemetry counters
    output logic [31:0]                             stat_dram_rd_beats_o,
    output logic [31:0]                             stat_dram_wr_beats_o,
    output logic [31:0]                             stat_tcdm_words_o,
    output logic [31:0]                             stat_busy_cycles_o
);

    localparam int unsigned BUF_W = DRAM_DATA_W * 2 + TCDM_DATA_W; // 1096 bits (2 DRAM beats + margin)
    localparam int unsigned DRAM_BEAT_BYTES = DRAM_DATA_W / 8;     // 64 bytes

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_DRAM_TO_TCDM,
        ST_TCDM_TO_DRAM,
        ST_DONE
    } dma_state_t;

    dma_state_t state_q;

    logic [1:0]            dir_q;
    logic [ADDR_WIDTH-1:0] dram_addr_q;
    logic [ADDR_WIDTH-1:0] tcdm_addr_q;
    logic [31:0]           rem_bytes_tcdm_q; // Remaining bytes to pop into TCDM holding regs / read from TCDM
    logic [31:0]           rem_bytes_dram_q; // Remaining bytes to request from DRAM
    logic [3:0]            dram_rd_inflight_q;
    logic                  tcdm_rd_inflight_q;

    // Bit-level gearbox shift buffer
    logic [BUF_W-1:0]      buf_q;
    logic [11:0]           buf_bits_q;

    // Per-port holding registers for conflict-safe parallel TCDM writes
    logic [TCDM_PORTS-1:0]                   hold_valid_q;
    logic [TCDM_PORTS-1:0][ADDR_WIDTH-1:0]   hold_addr_q;
    logic [TCDM_PORTS-1:0][TCDM_DATA_W-1:0]  hold_data_q;

    function automatic logic [ADDR_WIDTH-1:0] next_tcdm_addr(
        input logic [ADDR_WIDTH-1:0] cur,
        input logic                  linear_mode
    );
        if (linear_mode) begin
            next_tcdm_addr = cur + 32'h0000_0200;
        end else if (cur[7:4] == 4'hF) begin
            next_tcdm_addr = {cur[ADDR_WIDTH-1:9] + 23'd1, cur[8], 4'h0, cur[3:0]};
        end else begin
            next_tcdm_addr = cur + 32'h0000_0010;
        end
    endfunction

    assign desc_ready_o = (state_q == ST_IDLE);

    // Combinational request outputs
    logic dram_rd_issue, dram_wr_issue, tcdm_rd_issue;
    logic [TCDM_PORTS-1:0] tcdm_wr_fire;

    always_comb begin
        dram_req_valid_o = 1'b0;
        dram_req_addr_o  = dram_addr_q;
        dram_req_write_o = 1'b0;
        dram_req_wdata_o = buf_q[DRAM_DATA_W-1:0];
        dram_req_strb_o  = {DRAM_STRB_W{1'b1}};

        for (int p = 0; p < TCDM_PORTS; p++) begin
            tcdm_req_valid_o[p] = 1'b0;
            tcdm_req_addr_o[p]  = hold_addr_q[p];
            tcdm_req_write_o[p] = 1'b0;
            tcdm_req_wdata_o[p] = hold_data_q[p];
            tcdm_req_strb_o[p]  = {TCDM_STRB_W{1'b1}};
        end

        if (state_q == ST_DRAM_TO_TCDM) begin
            // Pipeline DRAM reads up to buffer capacity
            if ((rem_bytes_dram_q != 32'd0) &&
                ((buf_bits_q + (12'(dram_rd_inflight_q) * 12'(DRAM_DATA_W))) <= 12'(BUF_W - DRAM_DATA_W))) begin
                dram_req_valid_o = 1'b1;
                dram_req_write_o = 1'b0;
            end
            for (int p = 0; p < TCDM_PORTS; p++) begin
                tcdm_req_valid_o[p] = hold_valid_q[p];
                tcdm_req_write_o[p] = hold_valid_q[p];
            end
        end else if (state_q == ST_TCDM_TO_DRAM) begin
            // Port 0 reads 72b word from TCDM
            if ((rem_bytes_tcdm_q != 32'd0) && !tcdm_rd_inflight_q &&
                (buf_bits_q <= 12'(BUF_W - TCDM_DATA_W))) begin
                tcdm_req_valid_o[0] = 1'b1;
                tcdm_req_write_o[0] = 1'b0;
                tcdm_req_addr_o[0]  = tcdm_addr_q;
            end
            // Flush 512b beat to DRAM
            if ((buf_bits_q >= 12'(DRAM_DATA_W)) ||
                ((rem_bytes_tcdm_q == 32'd0) && !tcdm_rd_inflight_q && (buf_bits_q > 12'd0))) begin
                dram_req_valid_o = 1'b1;
                dram_req_write_o = 1'b1;
            end
        end
    end

    assign dram_rd_issue = dram_req_valid_o && !dram_req_write_o && dram_req_ready_i;
    assign dram_wr_issue = dram_req_valid_o &&  dram_req_write_o && dram_req_ready_i;
    assign tcdm_rd_issue = tcdm_req_valid_o[0] && !tcdm_req_write_o[0] && tcdm_req_ready_i[0];
    always_comb begin
        for (int p = 0; p < TCDM_PORTS; p++) begin
            tcdm_wr_fire[p] = tcdm_req_valid_o[p] && tcdm_req_write_o[p] && tcdm_req_ready_i[p];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q              <= ST_IDLE;
            desc_done_o          <= 1'b0;
            dir_q                <= 2'd0;
            dram_addr_q          <= '0;
            tcdm_addr_q          <= '0;
            rem_bytes_tcdm_q     <= 32'd0;
            rem_bytes_dram_q     <= 32'd0;
            dram_rd_inflight_q   <= 4'd0;
            tcdm_rd_inflight_q   <= 1'b0;
            buf_q                <= '0;
            buf_bits_q           <= 12'd0;
            hold_valid_q         <= '0;
            hold_addr_q          <= '0;
            hold_data_q          <= '0;
            stat_dram_rd_beats_o <= 32'd0;
            stat_dram_wr_beats_o <= 32'd0;
            stat_tcdm_words_o    <= 32'd0;
            stat_busy_cycles_o   <= 32'd0;
        end else begin
            desc_done_o <= 1'b0;

            if (state_q != ST_IDLE) begin
                stat_busy_cycles_o <= stat_busy_cycles_o + 32'd1;
            end

            case (state_q)
                ST_IDLE: begin
                    if (desc_valid_i) begin
                        dir_q              <= desc_dir_i;
                        dram_addr_q        <= desc_dram_addr_i;
                        tcdm_addr_q        <= desc_tcdm_addr_i;
                        rem_bytes_tcdm_q   <= desc_size_bytes_i;
                        rem_bytes_dram_q   <= desc_size_bytes_i;
                        dram_rd_inflight_q <= 4'd0;
                        tcdm_rd_inflight_q <= 1'b0;
                        buf_q              <= '0;
                        buf_bits_q         <= 12'd0;
                        hold_valid_q       <= '0;
                        if (desc_size_bytes_i == 32'd0) begin
                            state_q <= ST_DONE;
                        end else if (desc_dir_i[0] == 1'b0) begin
                            state_q <= ST_DRAM_TO_TCDM;
                        end else begin
                            state_q <= ST_TCDM_TO_DRAM;
                        end
                    end
                end

                ST_DRAM_TO_TCDM: begin
                    logic [BUF_W-1:0]      next_buf;
                    logic [11:0]           next_bits;
                    logic [TCDM_PORTS-1:0] next_hold_valid;
                    logic [ADDR_WIDTH-1:0] cur_taddr;
                    logic [31:0]           cur_rem;
                    int unsigned           wr_done_cnt;

                    next_buf        = buf_q;
                    next_bits       = buf_bits_q;
                    next_hold_valid = hold_valid_q;
                    cur_taddr       = tcdm_addr_q;
                    cur_rem         = rem_bytes_tcdm_q;
                    wr_done_cnt     = 0;

                    // 1. Retire accepted TCDM writes
                    for (int p = 0; p < TCDM_PORTS; p++) begin
                        if (tcdm_wr_fire[p]) begin
                            next_hold_valid[p] = 1'b0;
                            wr_done_cnt++;
                        end
                    end
                    stat_tcdm_words_o <= stat_tcdm_words_o + 32'(wr_done_cnt);

                    // 2. Push arriving 512b DRAM beat
                    if (dram_rsp_valid_i) begin
                        next_buf  = next_buf | ({{(BUF_W-DRAM_DATA_W){1'b0}}, dram_rsp_rdata_i} << next_bits);
                        next_bits = next_bits + 12'(DRAM_DATA_W);
                        stat_dram_rd_beats_o <= stat_dram_rd_beats_o + 32'd1;
                    end

                    // Update inflight DRAM read counter
                    dram_rd_inflight_q <= dram_rd_inflight_q +
                                          (dram_rd_issue ? 4'd1 : 4'd0) -
                                          (dram_rsp_valid_i ? 4'd1 : 4'd0);

                    if (dram_rd_issue) begin
                        dram_addr_q      <= dram_addr_q + DRAM_BEAT_BYTES;
                        rem_bytes_dram_q <= (rem_bytes_dram_q >= DRAM_BEAT_BYTES) ?
                                            (rem_bytes_dram_q - DRAM_BEAT_BYTES) : 32'd0;
                    end

                    // 3. When all holding slots are empty, refill up to TCDM_PORTS words in parallel
                    if (next_hold_valid == '0) begin
                        for (int p = 0; p < TCDM_PORTS; p++) begin
                            if ((cur_rem >= TCDM_STRB_W) && (next_bits >= 12'(TCDM_DATA_W))) begin
                                next_hold_valid[p] = 1'b1;
                                hold_addr_q[p]     <= cur_taddr;
                                hold_data_q[p]     <= next_buf[TCDM_DATA_W-1:0];
                                next_buf           = next_buf >> TCDM_DATA_W;
                                next_bits          = next_bits - 12'(TCDM_DATA_W);
                                cur_taddr          = next_tcdm_addr(cur_taddr, dir_q[1]);
                                cur_rem            = cur_rem - TCDM_STRB_W;
                            end
                        end
                    end

                    hold_valid_q     <= next_hold_valid;
                    tcdm_addr_q      <= cur_taddr;
                    rem_bytes_tcdm_q <= cur_rem;
                    buf_q            <= next_buf;
                    buf_bits_q       <= next_bits;

                    if ((cur_rem == 32'd0) && (next_hold_valid == '0) && (dram_rd_inflight_q == 4'd0) && !dram_rsp_valid_i) begin
                        state_q <= ST_DONE;
                    end
                end

                ST_TCDM_TO_DRAM: begin
                    logic [BUF_W-1:0] next_buf;
                    logic [11:0]      next_bits;

                    next_buf  = buf_q;
                    next_bits = buf_bits_q;

                    if (dram_wr_issue) begin
                        next_buf  = next_buf >> DRAM_DATA_W;
                        next_bits = (next_bits >= 12'(DRAM_DATA_W)) ?
                                    (next_bits - 12'(DRAM_DATA_W)) : 12'd0;
                        dram_addr_q <= dram_addr_q + DRAM_BEAT_BYTES;
                        stat_dram_wr_beats_o <= stat_dram_wr_beats_o + 32'd1;
                    end

                    if (tcdm_rsp_valid_i[0]) begin
                        next_buf  = next_buf | ({{(BUF_W-TCDM_DATA_W){1'b0}}, tcdm_rsp_rdata_i[0]} << next_bits);
                        next_bits = next_bits + 12'(TCDM_DATA_W);
                        tcdm_rd_inflight_q <= 1'b0;
                        stat_tcdm_words_o <= stat_tcdm_words_o + 32'd1;
                    end

                    if (tcdm_rd_issue) begin
                        tcdm_rd_inflight_q <= 1'b1;
                        tcdm_addr_q        <= next_tcdm_addr(tcdm_addr_q, dir_q[1]);
                        rem_bytes_tcdm_q   <= (rem_bytes_tcdm_q >= TCDM_STRB_W) ?
                                              (rem_bytes_tcdm_q - TCDM_STRB_W) : 32'd0;
                    end

                    buf_q      <= next_buf;
                    buf_bits_q <= next_bits;

                    if ((rem_bytes_tcdm_q == 32'd0) && !tcdm_rd_inflight_q && !tcdm_rsp_valid_i[0] &&
                        (next_bits == 12'd0)) begin
                        state_q <= ST_DONE;
                    end
                end

                ST_DONE: begin
                    desc_done_o <= 1'b1;
                    state_q     <= ST_IDLE;
                end

                default: state_q <= ST_IDLE;
            endcase
        end
    end

endmodule
