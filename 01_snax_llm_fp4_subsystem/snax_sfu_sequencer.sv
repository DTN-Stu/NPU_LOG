// =============================================================================
// Module: snax_sfu_sequencer.sv
// Description: Autonomous 2D / 2-Pass Hardware Sequencer for snax_llm_sfu_opt.
//              Replaces per-vector CPU CSR polling loops for:
//                - RMSNorm (op=2): 2-pass per token across M_TOKENS
//                - Residual / SwiGLU (op=1,4): 2D nested loop (t in M_TOKENS,
//                  g in N_TILES) with independent 16-byte-aligned T/G strides
//                  for Operand A, Operand B, and Output.
// =============================================================================
`timescale 1ns/1ps

module snax_sfu_sequencer #(
    parameter int unsigned ADDR_WIDTH = 32
)(
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    start_i,
    output logic                    busy_o,
    output logic                    done_o,

    // Descriptor (from CSR)
    input  logic [2:0]              cfg_sfu_op_i,
    input  logic [15:0]             cfg_m_tokens_i,
    input  logic [15:0]             cfg_n_tiles_i,
    input  logic [15:0]             cfg_k_vecs_i,

    input  logic [ADDR_WIDTH-1:0]   cfg_base_a_i,
    input  logic [ADDR_WIDTH-1:0]   cfg_stride_a_t_i,
    input  logic [ADDR_WIDTH-1:0]   cfg_stride_a_g_i,

    input  logic [ADDR_WIDTH-1:0]   cfg_base_b_i,
    input  logic [ADDR_WIDTH-1:0]   cfg_stride_b_t_i,
    input  logic [ADDR_WIDTH-1:0]   cfg_stride_b_g_i,

    input  logic [ADDR_WIDTH-1:0]   cfg_base_o_i,
    input  logic [ADDR_WIDTH-1:0]   cfg_stride_o_t_i,
    input  logic [ADDR_WIDTH-1:0]   cfg_stride_o_g_i,

    // Streamer & SFU busy status
    input  logic                    rd_busy_i,
    input  logic                    rd_b_busy_i,
    input  logic                    sfu_busy_i,
    input  logic                    wo_busy_i,

    // Sequencer-driven control outputs
    output logic                    rd_start_o,
    output logic [ADDR_WIDTH-1:0]   rd_base_o,
    output logic [31:0]             rd_nv_o,
    output logic [31:0]             rd_ls_o,

    output logic                    rd_b_start_o,
    output logic [ADDR_WIDTH-1:0]   rd_b_base_o,
    output logic [31:0]             rd_b_nv_o,
    output logic [31:0]             rd_b_ls_o,

    output logic                    sfu_start_o,

    output logic                    wo_start_o,
    output logic [ADDR_WIDTH-1:0]   wo_base_o,
    output logic [31:0]             wo_nv_o,
    output logic [31:0]             wo_ls_o
);

    localparam logic [2:0] OP_RMSNORM = 3'd2;

    typedef enum logic [2:0] {
        S_IDLE,
        S_RMS_P1_LAUNCH,
        S_RMS_P1_WAIT,
        S_RMS_P2_LAUNCH,
        S_RMS_P2_WAIT,
        S_2D_LAUNCH,
        S_2D_WAIT,
        S_DONE
    } state_t;

    state_t state_q;

    logic [15:0]           t_idx_q, g_idx_q;
    logic [1:0]            wait_guard_q;
    logic [ADDR_WIDTH-1:0] addr_a_t_q, addr_a_g_q;
    logic [ADDR_WIDTH-1:0] addr_b_t_q, addr_b_g_q;
    logic [ADDR_WIDTH-1:0] addr_o_t_q, addr_o_g_q;

    assign busy_o = (state_q != S_IDLE);

    wire [15:0] eff_m = (cfg_m_tokens_i == 16'd0) ? 16'd1 : cfg_m_tokens_i;
    wire [15:0] eff_n = (cfg_n_tiles_i  == 16'd0) ? 16'd1 : cfg_n_tiles_i;

    wire any_busy_p1 = rd_busy_i;
    wire any_busy_p2 = rd_busy_i || rd_b_busy_i || sfu_busy_i || wo_busy_i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q      <= S_IDLE;
            done_o       <= 1'b0;
            t_idx_q      <= 16'd0;
            g_idx_q      <= 16'd0;
            wait_guard_q <= 2'd0;
            addr_a_t_q   <= '0;
            addr_a_g_q   <= '0;
            addr_b_t_q   <= '0;
            addr_b_g_q   <= '0;
            addr_o_t_q   <= '0;
            addr_o_g_q   <= '0;

            rd_start_o   <= 1'b0;
            rd_base_o    <= '0;
            rd_nv_o      <= 32'd1;
            rd_ls_o      <= 32'h0000_0200;
            rd_b_start_o <= 1'b0;
            rd_b_base_o  <= '0;
            rd_b_nv_o    <= 32'd1;
            rd_b_ls_o    <= 32'h0000_0200;
            sfu_start_o  <= 1'b0;
            wo_start_o   <= 1'b0;
            wo_base_o    <= '0;
            wo_nv_o      <= 32'd1;
            wo_ls_o      <= 32'h0000_0200;
        end else begin
            done_o       <= 1'b0;
            rd_start_o   <= 1'b0;
            rd_b_start_o <= 1'b0;
            sfu_start_o  <= 1'b0;
            wo_start_o   <= 1'b0;

            case (state_q)
                S_IDLE: begin
                    if (start_i) begin
                        t_idx_q    <= 16'd0;
                        g_idx_q    <= 16'd0;
                        addr_a_t_q <= cfg_base_a_i;
                        addr_a_g_q <= '0;
                        addr_b_t_q <= cfg_base_b_i;
                        addr_b_g_q <= '0;
                        addr_o_t_q <= cfg_base_o_i;
                        addr_o_g_q <= '0;
                        state_q    <= (cfg_sfu_op_i == OP_RMSNORM) ? S_RMS_P1_LAUNCH : S_2D_LAUNCH;
                    end
                end

                // -------------------------------------------------------------
                // RMSNorm Pass 1: Read X -> SFU accumulate sum(x^2) + rsqrt
                // -------------------------------------------------------------
                S_RMS_P1_LAUNCH: begin
                    rd_base_o    <= addr_a_t_q;
                    rd_nv_o      <= {16'd0, cfg_k_vecs_i};
                    rd_ls_o      <= 32'h0000_0200;
                    rd_start_o   <= 1'b1;
                    sfu_start_o  <= 1'b1;
                    wait_guard_q <= 2'd2;
                    state_q      <= S_RMS_P1_WAIT;
                end

                S_RMS_P1_WAIT: begin
                    if (wait_guard_q != 2'd0) begin
                        wait_guard_q <= wait_guard_q - 2'd1;
                    end else if (!any_busy_p1) begin
                        state_q <= S_RMS_P2_LAUNCH;
                    end
                end

                // -------------------------------------------------------------
                // RMSNorm Pass 2: Read X + Gamma -> SFU scale -> Write Out
                // -------------------------------------------------------------
                S_RMS_P2_LAUNCH: begin
                    rd_base_o    <= addr_a_t_q;
                    rd_nv_o      <= {16'd0, cfg_k_vecs_i};
                    rd_ls_o      <= 32'h0000_0200;
                    rd_b_base_o  <= cfg_base_b_i;
                    rd_b_nv_o    <= {16'd0, cfg_k_vecs_i};
                    rd_b_ls_o    <= 32'h0000_0200;
                    wo_base_o    <= addr_o_t_q;
                    wo_nv_o      <= {16'd0, cfg_k_vecs_i};
                    wo_ls_o      <= 32'h0000_0200;
                    rd_start_o   <= 1'b1;
                    rd_b_start_o <= 1'b1;
                    wo_start_o   <= 1'b1;
                    wait_guard_q <= 2'd2;
                    state_q      <= S_RMS_P2_WAIT;
                end

                S_RMS_P2_WAIT: begin
                    if (wait_guard_q != 2'd0) begin
                        wait_guard_q <= wait_guard_q - 2'd1;
                    end else if (!any_busy_p2) begin
                        if (t_idx_q + 16'd1 < eff_m) begin
                            t_idx_q    <= t_idx_q + 16'd1;
                            addr_a_t_q <= addr_a_t_q + cfg_stride_a_t_i;
                            addr_o_t_q <= addr_o_t_q + cfg_stride_o_t_i;
                            state_q    <= S_RMS_P1_LAUNCH;
                        end else begin
                            state_q <= S_DONE;
                        end
                    end
                end

                // -------------------------------------------------------------
                // 2D Batch (Residual / SwiGLU): Loop (t, g)
                // -------------------------------------------------------------
                S_2D_LAUNCH: begin
                    rd_base_o    <= addr_a_t_q + addr_a_g_q;
                    rd_nv_o      <= 32'd1;
                    rd_ls_o      <= 32'h0000_0200;
                    rd_b_base_o  <= addr_b_t_q + addr_b_g_q;
                    rd_b_nv_o    <= 32'd1;
                    rd_b_ls_o    <= 32'h0000_0200;
                    wo_base_o    <= addr_o_t_q + addr_o_g_q;
                    wo_nv_o      <= 32'd1;
                    wo_ls_o      <= 32'h0000_0200;
                    rd_start_o   <= 1'b1;
                    rd_b_start_o <= 1'b1;
                    sfu_start_o  <= 1'b1;
                    wo_start_o   <= 1'b1;
                    wait_guard_q <= 2'd2;
                    state_q      <= S_2D_WAIT;
                end

                S_2D_WAIT: begin
                    if (wait_guard_q != 2'd0) begin
                        wait_guard_q <= wait_guard_q - 2'd1;
                    end else if (!any_busy_p2) begin
                        if (g_idx_q + 16'd1 < eff_n) begin
                            g_idx_q    <= g_idx_q + 16'd1;
                            addr_a_g_q <= addr_a_g_q + cfg_stride_a_g_i;
                            addr_b_g_q <= addr_b_g_q + cfg_stride_b_g_i;
                            addr_o_g_q <= addr_o_g_q + cfg_stride_o_g_i;
                            state_q    <= S_2D_LAUNCH;
                        end else if (t_idx_q + 16'd1 < eff_m) begin
                            g_idx_q    <= 16'd0;
                            addr_a_g_q <= '0;
                            addr_b_g_q <= '0;
                            addr_o_g_q <= '0;
                            t_idx_q    <= t_idx_q + 16'd1;
                            addr_a_t_q <= addr_a_t_q + cfg_stride_a_t_i;
                            addr_b_t_q <= addr_b_t_q + cfg_stride_b_t_i;
                            addr_o_t_q <= addr_o_t_q + cfg_stride_o_t_i;
                            state_q    <= S_2D_LAUNCH;
                        end else begin
                            state_q <= S_DONE;
                        end
                    end
                end

                S_DONE: begin
                    done_o  <= 1'b1;
                    state_q <= S_IDLE;
                end

                default: state_q <= S_IDLE;
            endcase
        end
    end

endmodule
