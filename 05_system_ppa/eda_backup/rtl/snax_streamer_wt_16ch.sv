// =============================================================================
// Module: snax_streamer_wt_16ch.sv
// Description: Clean, High-PPA 16-Channel Parallel Weight/Key/Value Streamer
//              - Connects to TCDM Interconnect via Masters 2..17 (1152 bits/cycle)
//              - 3D Nested Loop AGU (Loops: Replay, K-Block, Tile)
//              - Bank-Interleaved Zero-Conflict Fanout:
//                    Addr[w] = cfg_base_addr_i + w*16
//                cfg_base_addr_i is the real TCDM byte address of lane 0
//                (16-byte aligned); lane banks start at base[8:4].
//              - 16x Elastic Skid Buffers with FIFO-head replay
//              - 16-Lane Barrier Synchronizer for lock-step Core delivery
//
// Replay semantics:
//   One TCDM read is fetched per (tile, K-block) vector. The FIFO head is held
//   for cfg_repeat_tokens core handshakes; it is popped only on the last
//   handshake. No TCDM request is issued while a replay is active.
// =============================================================================
`timescale 1ns/1ps

module snax_streamer_wt_16ch #(
    parameter int unsigned DATA_WIDTH = 72,   // 72-bit Packet (64b FP4 + 8b Exp)
    parameter int unsigned ADDR_WIDTH = 32,   // 32-bit Address
    parameter int unsigned NUM_LANES  = 16,   // 16 Parallel Lanes
    parameter int unsigned STRB_WIDTH = 9     // 9-bit Byte Strobe
)(
    input  logic                                  clk,
    input  logic                                  rst_n,

    // -------------------------------------------------------------------------
    // 1. CSR Control & 3D AGU Configuration
    // -------------------------------------------------------------------------
    input  logic                                  csr_start_i,
    output logic                                  busy_o,
    output logic                                  done_irq_o,

    input  logic [ADDR_WIDTH-1:0]                 cfg_base_addr_i,    // Real TCDM byte address of lane 0 (16B aligned)
    input  logic [31:0]                           cfg_repeat_tokens_i,// Replay count per vector
    // K packets per K_TILE: cfg_k_blocks_i = K_TILE / 16.  This is NOT a
    // count of K tiles.  One start covers one 16 x K_TILE weight tile
    // (per cfg_num_tiles N tiles); the streamer walks the K packets
    // internally, one 16-K packet per lane per beat.
    input  logic [31:0]                           cfg_k_blocks_i,     // Số blocks theo chiều K (Loop 1)
    input  logic [31:0]                           cfg_k_stride_i,     // Bước nhảy sang block K kế
    input  logic [31:0]                           cfg_num_tiles_i,    // Số tiles trọng số (Loop 2)
    input  logic [31:0]                           cfg_tile_step_i,    // Bước nhảy sang tile kế

    // -------------------------------------------------------------------------
    // 2. TCDM Interconnect Master Ports (Masters 2..17)
    // -------------------------------------------------------------------------
    output logic [NUM_LANES-1:0]                  m_tcdm_req_valid_o,
    output logic [NUM_LANES-1:0]                  m_tcdm_req_write_o,
    output logic [NUM_LANES-1:0][ADDR_WIDTH-1:0]  m_tcdm_req_addr_o,
    output logic [NUM_LANES-1:0][DATA_WIDTH-1:0]  m_tcdm_req_wdata_o,
    output logic [NUM_LANES-1:0][STRB_WIDTH-1:0]  m_tcdm_req_strb_o,
    input  logic [NUM_LANES-1:0]                  m_tcdm_req_ready_i,

    input  logic [NUM_LANES-1:0][DATA_WIDTH-1:0]  m_tcdm_rsp_rdata_i,
    input  logic [NUM_LANES-1:0]                  m_tcdm_rsp_valid_i,

    // -------------------------------------------------------------------------
    // 3. Accelerator Shell Interface (wt_stream_*)
    // -------------------------------------------------------------------------
    output logic [NUM_LANES-1:0][DATA_WIDTH-1:0]  wt_stream_data_o ,
    output logic [NUM_LANES-1:0]                  wt_stream_valid_o,
    input  logic [NUM_LANES-1:0]                  wt_stream_ready_i
);

    // =========================================================================
    // 1. FSM & 3D AGU COUNTERS
    // =========================================================================
    typedef enum logic [1:0] { IDLE, RUN, DRAIN_WAIT } state_t;
    state_t state_q, state_d;

    logic [31:0] rep_cnt_q,   rep_cnt_d;
    logic [31:0] kb_cnt_q,    kb_cnt_d;
    logic [31:0] tile_cnt_q,  tile_cnt_d;
    logic [ADDR_WIDTH-1:0] base_addr_q, base_addr_d;
    logic                  replay_active_q, replay_active_d;
    logic                  next_fetched_q, next_fetched_d;

    // Job descriptor, sampled at start.  A producer is free to change cfg_*
    // after the launch cycle, so every field must be captured here.
    logic [31:0] repeat_q, k_blocks_q, num_tiles_q, k_stride_q, tile_step_q;

    // Rollovers
    wire rep_last  = (rep_cnt_q  == repeat_q    - 1'b1);
    wire kb_last   = (kb_cnt_q   == k_blocks_q  - 1'b1);
    wire tile_last = (tile_cnt_q == num_tiles_q - 1'b1);

    // =========================================================================
    // 2. SKID BUFFER HANDSHAKE
    // =========================================================================
    logic [NUM_LANES-1:0] skid_ready, skid_valid;
    logic [DATA_WIDTH-1:0] skid_data_out [0:NUM_LANES-1];
    logic [2:0]            skid_level   [0:NUM_LANES-1];

    localparam int unsigned SKID_DEPTH = 4;

    logic [31:0]           f_kb_q, f_tile_q;
    logic [ADDR_WIDTH-1:0] f_addr_q;
    logic                  f_done_q;
    logic [2:0]            inflight_q [0:NUM_LANES-1];
    logic [NUM_LANES-1:0]  lane_credit_ok;

    wire stream_mode = (repeat_q == 32'd1);
    wire f_kb_last   = (f_kb_q   == k_blocks_q  - 1'b1);
    wire f_tile_last = (f_tile_q == num_tiles_q - 1'b1);

    wire all_skids_ready = &skid_ready;
    wire all_lanes_ready = &skid_valid;
    wire all_core_ready  = &wt_stream_ready_i;
    wire tcdm_all_gnt    = &m_tcdm_req_ready_i;

    // A vector is consumed only when all 16 lanes handshake together.
    wire core_pop_group = all_lanes_ready && all_core_ready;
    wire core_pop_last  = core_pop_group && rep_last;

    // Address of the next (tile, K-block) vector.  It matches the AGU state
    // that is committed when the current replay finishes.
    wire has_next = !(kb_last && tile_last);
    wire [ADDR_WIDTH-1:0] next_base_addr =
        kb_last ? (base_addr_q + tile_step_q) : (base_addr_q + k_stride_q);

    // First fetch of a vector, and speculative fetch of the next vector while
    // the current one is being replayed.  The second uses the free slot of the
    // depth-2 skid buffers, so the next replay starts with no TCDM latency
    // bubble.  It is only issued once the current head is actually present.
    wire fetch_ok    = !stream_mode && (state_q == RUN) && !replay_active_q && all_skids_ready;
    wire prefetch_ok = !stream_mode && (state_q == RUN) && replay_active_q && all_lanes_ready &&
                       !next_fetched_q && all_skids_ready && has_next &&
                       !core_pop_last;
    wire fetch_fire        = fetch_ok    && tcdm_all_gnt;
    wire prefetch_fire     = prefetch_ok && tcdm_all_gnt;
    wire stream_fetch_ok   = stream_mode && (state_q == RUN) && !f_done_q && (&lane_credit_ok);
    wire stream_fetch_fire = stream_fetch_ok && tcdm_all_gnt;

    // =========================================================================
    // 3. FSM & AGU
    // =========================================================================
    always_comb begin
        state_d         = state_q;
        rep_cnt_d       = rep_cnt_q;
        kb_cnt_d        = kb_cnt_q;
        tile_cnt_d      = tile_cnt_q;
        base_addr_d     = base_addr_q;
        replay_active_d = replay_active_q;
        next_fetched_d  = next_fetched_q;
        done_irq_o      = 1'b0;

        case (state_q)
            IDLE: begin
                if (csr_start_i) begin
                    state_d         = RUN;
                    rep_cnt_d       = '0;
                    kb_cnt_d        = '0;
                    tile_cnt_d      = '0;
                    base_addr_d     = cfg_base_addr_i;
                    replay_active_d = 1'b0;
                    next_fetched_d  = 1'b0;
                end
            end

            RUN: begin
                if (core_pop_group) begin
                    if (rep_last) begin
                        // Last replay: pop the FIFO head and advance the AGU.
                        rep_cnt_d = '0;
                        if (!kb_last) begin
                            kb_cnt_d    = kb_cnt_q + 1'b1;
                            base_addr_d = base_addr_q + k_stride_q;
                        end else begin
                            kb_cnt_d = '0;
                            if (!tile_last) begin
                                tile_cnt_d  = tile_cnt_q + 1'b1;
                                base_addr_d = base_addr_q + tile_step_q;
                            end else begin
                                state_d = DRAIN_WAIT;
                            end
                        end
                        // If the next vector was prefetched it is already the
                        // new FIFO head: start replaying it immediately.
                        if (has_next && next_fetched_q) begin
                            replay_active_d = 1'b1;
                            next_fetched_d  = 1'b0;
                        end else begin
                            replay_active_d = 1'b0;
                            next_fetched_d  = 1'b0;
                        end
                    end else begin
                        rep_cnt_d = rep_cnt_q + 1'b1;
                    end
                end
                // Prefetch may overlap a normal replay handshake (not the last
                // one).  The normal fetch only happens when no replay is active.
                if (prefetch_fire)
                    next_fetched_d = 1'b1;
                else if (fetch_fire)
                    replay_active_d = 1'b1;
            end

            DRAIN_WAIT: begin
                if (!skid_valid) begin
                    done_irq_o = 1'b1;
                    state_d    = IDLE;
                end
            end

            default: state_d = IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q         <= IDLE;
            rep_cnt_q       <= '0;
            kb_cnt_q        <= '0;
            tile_cnt_q      <= '0;
            base_addr_q     <= '0;
            replay_active_q <= 1'b0;
            next_fetched_q  <= 1'b0;
            repeat_q        <= '0;
            k_blocks_q      <= 32'd1;
            num_tiles_q     <= 32'd1;
            k_stride_q      <= '0;
            tile_step_q     <= '0;
            f_kb_q          <= '0;
            f_tile_q        <= '0;
            f_addr_q        <= '0;
            f_done_q        <= 1'b0;
            for (int l = 0; l < NUM_LANES; l++) inflight_q[l] <= '0;
        end else begin
            state_q         <= state_d;
            rep_cnt_q       <= rep_cnt_d;
            kb_cnt_q        <= kb_cnt_d;
            tile_cnt_q      <= tile_cnt_d;
            base_addr_q     <= base_addr_d;
            replay_active_q <= replay_active_d;
            next_fetched_q  <= next_fetched_d;

            for (int l = 0; l < NUM_LANES; l++) begin
                inflight_q[l] <= inflight_q[l]
                               + ((stream_fetch_fire || fetch_fire || prefetch_fire) ? 3'd1 : 3'd0)
                               - (m_tcdm_rsp_valid_i[l] ? 3'd1 : 3'd0);
            end

            if (state_q == IDLE && csr_start_i) begin
                repeat_q    <= cfg_repeat_tokens_i;
                k_blocks_q  <= cfg_k_blocks_i;
                num_tiles_q <= cfg_num_tiles_i;
                k_stride_q  <= cfg_k_stride_i;
                tile_step_q <= cfg_tile_step_i;
                f_kb_q      <= '0;
                f_tile_q    <= '0;
                f_addr_q    <= cfg_base_addr_i;
                f_done_q    <= 1'b0;
            end else if (stream_fetch_fire) begin
                if (!f_kb_last) begin
                    f_kb_q   <= f_kb_q + 32'd1;
                    f_addr_q <= f_addr_q + k_stride_q;
                end else begin
                    f_kb_q <= '0;
                    if (!f_tile_last) begin
                        f_tile_q <= f_tile_q + 32'd1;
                        f_addr_q <= f_addr_q + tile_step_q;
                    end else begin
                        f_done_q <= 1'b1;
                    end
                end
            end
        end
    end

    // =========================================================================
    // 4. 16-WAY ZERO-CONFLICT INTERLEAVED ADDRESS FANOUT
    // =========================================================================
    genvar w;
    generate
        for (w = 0; w < NUM_LANES; w++) begin : GEN_TCDM_REQS
            assign lane_credit_ok[w]     = ({1'b0, skid_level[w]} + {1'b0, inflight_q[w]}) < 4'd4;
            assign m_tcdm_req_valid_o[w] = stream_mode ? stream_fetch_ok : (fetch_ok || prefetch_ok);
            assign m_tcdm_req_write_o[w] = 1'b0;
            // cfg_base_addr_i is the real TCDM address of lane 0.
            // Lane w is 16 bytes higher, i.e. the next bank at the same word.
            assign m_tcdm_req_addr_o[w]  = stream_mode ? (f_addr_q + (w << 4)) :
                                           (prefetch_ok ? (next_base_addr + (w << 4))
                                                        : (base_addr_q    + (w << 4)));
            assign m_tcdm_req_wdata_o[w] = '0;
            assign m_tcdm_req_strb_o[w]  = '0;
        end
    endgenerate

    // =========================================================================
    // 5. 16x SKID BUFFERS (FIFO HEAD HELD DURING REPLAY)
    // =========================================================================
    generate
        for (w = 0; w < NUM_LANES; w++) begin : GEN_SKID_BUFFERS
            skid_buffer #(
                .DATA_WIDTH (DATA_WIDTH),
                .DEPTH      (SKID_DEPTH)
            ) u_skid_buffer (
                .clk        (clk),
                .rst_n      (rst_n),
                .push       (m_tcdm_rsp_valid_i[w]),
                .data_in    (m_tcdm_rsp_rdata_i[w]),
                .pop        (core_pop_last),
                .data_out   (skid_data_out[w]),
                .ready      (skid_ready[w]),
                .valid      (skid_valid[w]),
                .level      (skid_level[w])
            );
        end
    endgenerate

    // =========================================================================
    // 6. 16-LANE BARRIER OUTPUT
    // =========================================================================
    generate
        for (w = 0; w < NUM_LANES; w++) begin : GEN_OUTPUTS
            assign wt_stream_data_o[w]  = skid_data_out[w];
            assign wt_stream_valid_o[w] = all_lanes_ready;
        end
    endgenerate

    assign busy_o = (state_q != IDLE) || (|skid_valid);

endmodule
