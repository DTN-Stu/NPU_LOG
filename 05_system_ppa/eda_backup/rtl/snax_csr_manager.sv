`timescale 1ns/1ps

module snax_csr_manager #(
    parameter int unsigned NUM_RW     = 32,
    parameter int unsigned NUM_RO     = 8,
    parameter int unsigned ADDR_WIDTH = 12,
    parameter int unsigned START_IDX  = NUM_RW - 1
)(
    input  logic                         clk,
    input  logic                         rst_n,

    // CPU / Snitch CSR request channel
    input  logic                         csr_req_valid_i,
    output logic                         csr_req_ready_o,
    input  logic                         csr_req_write_i,
    input  logic [ADDR_WIDTH-1:0]        csr_req_addr_i,
    input  logic [31:0]                  csr_req_data_i,

    // CPU / Snitch CSR response channel
    output logic                         csr_rsp_valid_o,
    output logic [31:0]                  csr_rsp_data_o,
    input  logic                         csr_rsp_ready_i,

    // Accelerator side: active RW configuration and bulk-commit
    output logic [NUM_RW-1:0][31:0]      cfg_rw_o,
    output logic [31:0]                  start_mask_o,
    output logic                         cfg_valid_o,
    input  logic                         cfg_ready_i,
    output logic                         cfg_commit_o,

    // Accelerator side: read-only registers
    input  logic [NUM_RO-1:0][31:0]      ro_i
);

    localparam logic [ADDR_WIDTH-1:0] ADDR_START  = ADDR_WIDTH'(START_IDX);
    localparam logic [ADDR_WIDTH-1:0] ADDR_RO0    = ADDR_WIDTH'(START_IDX + 1);
    localparam logic [ADDR_WIDTH-1:0] ADDR_RO_END = ADDR_WIDTH'(START_IDX + 1 + NUM_RO);

    logic [NUM_RW-1:0][31:0] shadow_q;
    logic [NUM_RW-1:0][31:0] active_q;
    logic                    cfg_ready_q;
    logic                    req_vld_q;
    logic                    req_wr_q;
    logic [ADDR_WIDTH-1:0]   req_addr_q;
    logic [31:0]             req_data_q;
    logic                    pending_q;
    logic                    commit_d1_q;
    logic                    commit_q;
    logic                    rsp_valid_q;
    logic [31:0]             rsp_data_q;
    int                      csr_i;

    wire req_fire      = csr_req_valid_i && csr_req_ready_o;
    wire addr_is_rw    = (req_addr_q < ADDR_START) ||
                         ((req_addr_q >= ADDR_RO_END) && (req_addr_q < ADDR_WIDTH'(NUM_RW)));
    wire addr_is_start = (req_addr_q == ADDR_START);
    wire addr_is_ro    = (req_addr_q >= ADDR_RO0) && (req_addr_q < ADDR_RO_END);

    // Registered ready: no combinational loop from cfg_ready_i/csr_req_write_i/csr_rsp_ready_i
    assign csr_req_ready_o = !req_vld_q && !rsp_valid_q && !pending_q && !commit_d1_q && !commit_q;
    assign csr_rsp_valid_o = rsp_valid_q;
    assign csr_rsp_data_o  = rsp_data_q;

    assign cfg_rw_o     = active_q;
    assign start_mask_o = active_q[ADDR_START];
    assign cfg_valid_o  = pending_q;
    assign cfg_commit_o = commit_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (csr_i = 0; csr_i < NUM_RW; csr_i++) begin
                shadow_q[csr_i] <= '0;
                active_q[csr_i] <= '0;
            end
            cfg_ready_q <= 1'b0;
            req_vld_q   <= 1'b0;
            req_wr_q    <= 1'b0;
            req_addr_q  <= '0;
            req_data_q  <= '0;
            pending_q   <= 1'b0;
            commit_d1_q <= 1'b0;
            commit_q    <= 1'b0;
            rsp_valid_q <= 1'b0;
            rsp_data_q  <= '0;
        end else begin
            cfg_ready_q <= cfg_ready_i;
            commit_q    <= commit_d1_q;
            commit_d1_q <= 1'b0;

            if (pending_q && cfg_ready_q) begin
                active_q    <= shadow_q;
                pending_q   <= 1'b0;
                // Delay commit_q by +1 cycle after active_q loads (multicycle-safe setup)
                commit_d1_q <= 1'b1;
            end

            if (rsp_valid_q && csr_rsp_ready_i) begin
                rsp_valid_q <= 1'b0;
            end

            if (req_fire) begin
                req_vld_q  <= 1'b1;
                req_wr_q   <= csr_req_write_i;
                req_addr_q <= csr_req_addr_i;
                req_data_q <= csr_req_data_i;
            end else if (req_vld_q) begin
                req_vld_q   <= 1'b0;
                rsp_valid_q <= 1'b1;

                if (req_wr_q) begin
                    rsp_data_q <= 32'd0;

                    if (addr_is_rw) begin
                        shadow_q[req_addr_q] <= req_data_q;
                    end else if (addr_is_start) begin
                        shadow_q[ADDR_START] <= req_data_q;
                        pending_q <= req_data_q[0];
                    end
                end else begin
                    if (addr_is_rw) begin
                        rsp_data_q <= shadow_q[req_addr_q];
                    end else if (addr_is_start) begin
                        rsp_data_q <= shadow_q[ADDR_START];
                    end else if (addr_is_ro) begin
                        rsp_data_q <= ro_i[req_addr_q - ADDR_RO0];
                    end else begin
                        rsp_data_q <= 32'd0;
                    end
                end
            end
        end
    end

endmodule
