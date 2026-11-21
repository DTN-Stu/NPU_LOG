// =============================================================================
// Module: snax_gemm_os_sequencer.sv
// Description: Dedicated 4-Loop Output-Stationary (OS) GEMM Sequencer per
//              GEMM_SEQUENCER_REDESIGN_SPEC.md with 3 decoupled Sub-FSMs:
//
//   +-----------------------------------------------------------------------+
//   | SUB-FSM 3: DMA Descriptor & Credit Controller (dma_state_q)           |
//   |   Arbiter driving snax_simple_dma (O WB > Urgent A > Prefetch A / W)  |
//   +---------+------------------------------+----------------------+-------+
//             | [DMA -> CMP]                 | [CMP -> DMA]         ^
//             | a_buf_ready, w_buf_ready,    | w_release_credit,    |
//             | o_buf_wb_busy_q, dma_busy_q  | o_flush_commit_pulse |
//             v                              v                      |
//   +---------------------------------------------------------------+-------+
//   | SUB-FSM 1: Compute FSM (cmp_state_q)                                  |
//   |   4-Loop: M_macro -> N_16 -> M_sub -> K_TILES (CAP_O threshold for O) |
//   |   Drives act_streamer, wt_streamer, wo/v_ctrl & MAC Port A            |
//   +---------+-----------------------------------------------------+-------+
//             | [CMP -> DR] drain_start_pulse      ^ [DR -> CMP]
//             v                                    | dr_busy, inflight_out_cnt_q,
//   +----------------------------------------------+ gemm_beat_ok (WAR guard)
//   | SUB-FSM 2: Drain & Output Guard FSM (dr_state_q)                      |
//   |   Port B Rolling Drain + inflight_out_cnt_q Credit + WAR Guard        |
//   +-----------------------------------------------------------------------+
//
// Code Organization (4-block FSM style per Sub-FSM):
//   - Section 0 : Inter-FSM Interface Wires & Shared Geometry Calculations
//   - Section 1 : SUB-FSM 1 (Compute)         [4 blocks: state_q, state_d, comb out, regs]
//   - Section 2 : SUB-FSM 2 (Drain & Guard)   [4 blocks: state_q, state_d, comb out, regs]
//   - Section 3 : SUB-FSM 3 (DMA & Credits)   [4 blocks: state_q, state_d, comb out, regs]
//   - Section 4 : Top-Level Status (busy_o / done_o)
// =============================================================================

`timescale 1ns/1ps

module snax_gemm_os_sequencer #(
    parameter int unsigned ADDR_WIDTH       = 32,
    parameter int unsigned NUM_LANES        = 16,
    parameter int unsigned NUM_RING_SLOTS   = 4,
    parameter int unsigned SLOT_SIZE_BYTES  = 128 * 1024,
    parameter int unsigned OUT_PIPE_CREDITS = 4,
    parameter int unsigned CAP_O_TILES      = 16  // Auto O-buf flush threshold (16 FP4 tiles = 36x64B DRAM beats)
)(
    input  logic                                clk,
    input  logic                                rst_n,

    // Job control
    input  logic                                start_i,
    output logic                                busy_o,
    output logic                                done_o,

    // Configuration descriptor
    input  logic                                cfg_enable_dma_i,
    input  logic                                cfg_gemm_decode_i,
    input  logic [1:0]                          cfg_out_mode_i,         // 0: FP16, 1: FP4 lin, 2: FP4 scatter, 3: Fused V
    input  logic [7:0]                          cfg_gemm_k_tiles_m1_i,  // K_TILES - 1
    input  logic                                cfg_act_bank_layout_i,  // 1: 16-bank A (banks 16..31)
    input  logic                                cfg_out_bank_layout_i,  // 1: 16-bank FP16 output scatter
    input  logic [15:0]                         cfg_gemm_n_tiles_i,
    input  logic [15:0]                         cfg_gemm_m_tiles_i,
    input  logic [15:0]                         cfg_gemm_k_blocks_i,

    input  logic [ADDR_WIDTH-1:0]               cfg_tcdm_ring_base_i,
    input  logic [ADDR_WIDTH-1:0]               cfg_tcdm_q_base_i,
    input  logic [ADDR_WIDTH-1:0]               cfg_tcdm_out_base_i,
    input  logic [ADDR_WIDTH-1:0]               cfg_v_out_base_i,
    input  logic [ADDR_WIDTH-1:0]               cfg_dram_kv_base_i,
    input  logic [31:0]                         cfg_stride_tile_i,

    // Macro-Tiling DMA controls (Ping-Pong A on TCDM + O_buf flush to DRAM)
    input  logic                                cfg_gemm_macro_dma_i,
    input  logic [7:0]                          cfg_gemm_m_sub_tiles_i, // M_sub tiles per M_macro batch
    input  logic [7:0]                          cfg_gemm_n_sub_tiles_i, // Optional O-buf column cap (0 = auto CAP_O_TILES)
    input  logic [ADDR_WIDTH-1:0]               cfg_dram_act_base_i,
    input  logic [ADDR_WIDTH-1:0]               cfg_dram_out_base_i,
    input  logic [31:0]                         cfg_macro_a_buf_stride_i,
    input  logic [31:0]                         cfg_macro_o_buf_stride_i,

    // DMA Master handshake
    output logic                                dma_req_valid_o,
    input  logic                                dma_req_ready_i,
    output logic [1:0]                          dma_req_dir_o,
    output logic [ADDR_WIDTH-1:0]               dma_req_dram_addr_o,
    output logic [ADDR_WIDTH-1:0]               dma_req_tcdm_addr_o,
    output logic [31:0]                         dma_req_size_bytes_o,
    input  logic                                dma_done_i,

    // Activation Streamer
    output logic                                act_start_o,
    output logic [ADDR_WIDTH-1:0]               act_base_o,
    output logic [31:0]                         act_k_blocks_o,
    output logic [31:0]                         act_k_stride_o,
    output logic [31:0]                         act_tokens_o,
    output logic [31:0]                         act_token_stride_o,
    input  logic                                act_busy_i,
    input  logic                                act_valid_i,
    output logic                                act_ready_o,

    // 16-Channel Weight Streamer
    output logic                                wt_start_o,
    output logic [ADDR_WIDTH-1:0]               wt_base_o,
    output logic [31:0]                         wt_repeat_o,
    output logic [31:0]                         wt_k_blocks_o,
    output logic [31:0]                         wt_k_stride_o,
    output logic [31:0]                         wt_num_tiles_o,
    output logic [31:0]                         wt_tile_step_o,
    input  logic                                wt_busy_i,
    input  logic [NUM_LANES-1:0]                wt_valid_i,
    output logic [NUM_LANES-1:0]                wt_ready_o,

    // Write-Out Streamer
    output logic                                wo_start_o,
    output logic [1:0]                          wo_mode_o,
    output logic [ADDR_WIDTH-1:0]               wo_base_o,
    output logic [31:0]                         wo_num_vectors_o,
    output logic [31:0]                         wo_line_stride_o,
    output logic [31:0]                         wo_num_batches_o,
    output logic [31:0]                         wo_batch_stride_o,
    output logic [31:0]                         wo_dim2_cnt_o,
    output logic [31:0]                         wo_dim2_stride_o,
    output logic [31:0]                         wo_dim3_cnt_o,
    output logic [31:0]                         wo_dim3_stride_o,
    input  logic                                wo_busy_i,
    input  logic                                wo_done_i,

    // Fused V Transpose controller
    output logic                                v_start_o,
    output logic [31:0]                         v_num_batches_o,
    input  logic                                v_busy_i,

    // Core Compute & Output Drain interface
    output logic [1:0]                          cfg_mode_o,
    output logic                                cfg_row_start_o,
    output logic [NUM_LANES-1:0]                cfg_key_mask_o,
    output logic                                cfg_enable_slq_o,
    output logic                                tile_drain_req_o,
    output logic [3:0]                          tile_drain_idx_o,
    output logic                                tile_clear_on_drain_o,
    output logic [NUM_LANES-1:0]                lane_valid_o,
    output logic [3:0]                          lane_acc_idx_o,
    input  logic                                core_fp16_valid_i,
    input  logic                                core_fp16_ready_i,
    input  logic                                core_out_fifo_ready_i
);

    localparam int unsigned PTR_W = (NUM_RING_SLOTS > 1) ? $clog2(NUM_RING_SLOTS) : 1;
    localparam int unsigned CNT_W = $clog2(NUM_RING_SLOTS + 1);

    // =========================================================================
    // SECTION 0: INTER-FSM INTERFACE & SHARED GEOMETRY DECLARATIONS
    // -------------------------------------------------------------------------
    // Signals crossing between Sub-FSMs:
    //   [CMP <-> DR ]
    //     - drain_start_pulse    (CMP -> DR)   : 1-cycle trigger at last K-tile beat.
    //     - dr_busy              (DR  -> CMP)  : High while Drain FSM is active.
    //     - inflight_out_cnt_q   (DR  -> CMP)  : Vectors still inside Core drain pipe.
    //     - gemm_beat_ok         (DR  -> CMP)  : Slot-level WAR hazard guard.
    //   [CMP <-> DMA]
    //     - dma_window_active    (CMP -> DMA)  : High whenever a job is running.
    //     - w_release_credit_pulse (CMP -> DMA): Pops W ring slot after M_sub reuse.
    //     - o_flush_commit_pulse (CMP -> DMA)  : Hands O_buf[o_cmp_sel_q] to DMA WB.
    //     - macro_a_rel_pulse    (CMP -> DMA)  : Releases A_buf[a_cmp_rd_sel_q] at M_macro end.
    //     - w_buf_ready          (DMA -> CMP)  : At least 1 W ring slot valid.
    //     - a_buf_ready          (DMA -> CMP)  : Active ping-pong A buffer valid.
    //     - a_buf_ready_q        (DMA <-> CMP) : Ping-pong A valid bits [1:0].
    //     - o_buf_wb_busy_q      (CMP <-> DMA) : O buffer [1:0] busy writing to DRAM.
    //     - dma_busy_q           (DMA -> CMP)  : High while snax_simple_dma is active.
    //     - a_cmp_rd_sel_q       (CMP -> DMA)  : Active A buffer read by Compute.
    //     - o_cmp_sel_q          (CMP -> DMA)  : Active O buffer written by Compute.
    //     - cmp_m_base_q, o_col_start_q, cur_m_sub, cur_o_cols (CMP -> DMA):
    //                            Latched on o_flush_commit_pulse for DRAM O WB.
    // =========================================================================

    typedef enum logic [2:0] {
        CMP_IDLE,
        CMP_O_INIT,
        CMP_KT_LAUNCH,
        CMP_KT_RUN,
        CMP_O_COMMIT,
        CMP_WAIT_DONE,
        CMP_DONE
    } cmp_state_t;

    typedef enum logic [1:0] {
        DR_IDLE,
        DR_WAIT_D1,
        DR_RUN
    } dr_state_t;

    typedef enum logic {
        DMA_IDLE,
        DMA_INFLIGHT
    } dma_state_t;

    typedef enum logic [1:0] {
        DMA_TAG_W = 2'd0,
        DMA_TAG_A = 2'd1,
        DMA_TAG_O = 2'd2
    } dma_tag_t;

    // Mode Helpers
    wire macro_dma_en        = cfg_gemm_macro_dma_i && cfg_enable_dma_i;
    wire w_ring_reuse_active = cfg_enable_dma_i && (cfg_gemm_k_tiles_m1_i == 8'd0);

    // Sub-FSM State Registers & Next-States
    cmp_state_t cmp_state_q, cmp_state_d;
    dr_state_t  dr_state_q,  dr_state_d;
    dma_state_t dma_state_q, dma_state_d;

    // Inter-FSM Handshake Wires
    logic        drain_start_pulse;       // [CMP -> DR ]
    logic        dr_busy;                 // [DR  -> CMP]
    logic [3:0]  inflight_out_cnt_q;      // [DR  -> CMP]
    logic        gemm_beat_ok;            // [DR  -> CMP]

    logic        dma_window_active;       // [CMP -> DMA]
    logic        w_release_credit_pulse;  // [CMP -> DMA]
    logic        o_flush_commit_pulse;    // [CMP -> DMA]
    logic        macro_a_rel_pulse;       // [CMP -> DMA]
    logic        w_buf_ready;             // [DMA -> CMP]
    logic        a_buf_ready;             // [DMA -> CMP]
    logic        dma_busy_q;              // [DMA -> CMP]

    // Cross-FSM State Registers & Geometry
    logic [15:0] cmp_m_base_q, o_col_start_q;
    logic        a_cmp_rd_sel_q, o_cmp_sel_q;
    logic [1:0]  a_buf_ready_q, o_buf_wb_busy_q;
    logic [CNT_W-1:0] w_ready_tiles_q;
    logic [PTR_W-1:0] ring_rd_ptr_q;

    // 4-Loop M_sub bound & O-buffer flush column threshold (no N_macro loop)
    wire [15:0] eff_m_sub = (!macro_dma_en || (cfg_gemm_m_sub_tiles_i == 8'd0) ||
                             ({8'd0, cfg_gemm_m_sub_tiles_i} >= cfg_gemm_m_tiles_i))
                            ? cfg_gemm_m_tiles_i : {8'd0, cfg_gemm_m_sub_tiles_i};

    wire [15:0] cmp_m_rem    = cfg_gemm_m_tiles_i - cmp_m_base_q;
    wire [15:0] cur_m_sub    = (cmp_m_rem > eff_m_sub) ? eff_m_sub : cmp_m_rem;
    wire        last_macro_m = ((cmp_m_base_q + cur_m_sub) >= cfg_gemm_m_tiles_i);

    // O_buf chunk capacity in N_16 columns: explicit CSR cap if != 0, else auto <= CAP_O_TILES
    wire [15:0] auto_o_cols  = (cur_m_sub <= 16'd1) ? 16'(CAP_O_TILES)     :
                               (cur_m_sub <= 16'd2) ? 16'(CAP_O_TILES / 2) :
                               (cur_m_sub <= 16'd4) ? 16'(CAP_O_TILES / 4) :
                               (cur_m_sub <= 16'd8) ? 16'(CAP_O_TILES / 8) : 16'd1;
    wire [15:0] o_max_cols   = !macro_dma_en                    ? cfg_gemm_n_tiles_i :
                               (cfg_gemm_n_sub_tiles_i != 8'd0) ? {8'd0, cfg_gemm_n_sub_tiles_i} :
                                                                  auto_o_cols;
    wire [15:0] rem_n_cols   = cfg_gemm_n_tiles_i - o_col_start_q;
    wire [15:0] cur_o_cols   = (rem_n_cols > o_max_cols) ? o_max_cols : rem_n_cols;
    wire        last_o_chunk = ((o_col_start_q + cur_o_cols) >= cfg_gemm_n_tiles_i);

    wire [31:0] a_stride_eff   = (cfg_macro_a_buf_stride_i != 32'd0) ? cfg_macro_a_buf_stride_i : 32'h0002_0000;
    wire [31:0] o_stride_eff   = (cfg_macro_o_buf_stride_i != 32'd0) ? cfg_macro_o_buf_stride_i : 32'h0002_0000;
    wire [31:0] total_k_blocks = (32'(cfg_gemm_k_tiles_m1_i) + 32'd1) * 32'(cfg_gemm_k_blocks_i);

    wire [31:0] o_cmp_base_0 = (cfg_out_mode_i == 2'd3) ? cfg_v_out_base_i : cfg_tcdm_out_base_i;
    wire [31:0] o_cmp_base   = o_cmp_base_0 + (macro_dma_en ? (o_cmp_sel_q ? o_stride_eff : 32'd0) : 32'd0);
    wire [31:0] a_cmp_base   = cfg_tcdm_q_base_i + (macro_dma_en ? (a_cmp_rd_sel_q ? a_stride_eff : 32'd0) : 32'd0);

    wire [31:0] gemm_tile_out_bytes = cfg_gemm_decode_i
                                      ? ((cfg_out_mode_i == 2'd0) ? 32'd36  : 32'd9)
                                      : ((cfg_out_mode_i == 2'd0) ? 32'd576 : 32'd144);
    wire        o_dma_is_16bank     = (cfg_out_mode_i == 2'd2) || (cfg_out_mode_i == 2'd3) ||
                                      ((cfg_out_mode_i == 2'd0) && cfg_out_bank_layout_i && !cfg_gemm_decode_i);

    assign v_num_batches_o = 32'(cur_m_sub) * 32'(cur_o_cols);

    // =========================================================================
    // SECTION 1: SUB-FSM 1 — COMPUTE FSM
    //   Exact 4-Loop Order: M_macro (cmp_m_base_q) -> N_16 (cnt_gn_q)
    //                    -> M_sub (cnt_gm_q) -> K_TILES (cnt_kt_q)
    // =========================================================================
    logic [15:0] cnt_gn_q, cnt_gm_q, cnt_gk_q;
    logic [7:0]  cnt_kt_q;
    logic [3:0]  cnt_gbeat_q;

    wire last_gbeat = cfg_gemm_decode_i ? (cnt_gbeat_q == 4'd0) : (cnt_gbeat_q == 4'd15);
    wire last_gk    = (cnt_gk_q == (cfg_gemm_k_blocks_i - 16'd1));
    wire last_kt    = (cnt_kt_q == cfg_gemm_k_tiles_m1_i);
    wire last_gm    = (cnt_gm_q == (cur_m_sub - 16'd1));
    wire last_gn    = (cnt_gn_q == (cfg_gemm_n_tiles_i - 16'd1));
    wire last_o_col = (cnt_gn_q == (o_col_start_q + cur_o_cols - 16'd1));

    wire gemm_compute_fire = act_valid_i && (&wt_valid_i) && gemm_beat_ok;

    // Inter-FSM qualifiers [CMP -> DR / DMA]
    assign drain_start_pulse = (cmp_state_q == CMP_KT_RUN) && gemm_compute_fire &&
                               last_gbeat && last_gk && last_kt;
    assign dma_window_active = (cmp_state_q != CMP_IDLE) && (cmp_state_q != CMP_DONE);

    // Compute Streamer TCDM Base Addresses
    wire [ADDR_WIDTH-1:0] core_w_tcdm_addr = cfg_enable_dma_i
                                             ? (cfg_tcdm_ring_base_i + ({16'd0, ring_rd_ptr_q} * SLOT_SIZE_BYTES))
                                             : (cfg_tcdm_ring_base_i + ((32'(cnt_gn_q) * total_k_blocks +
                                                                         32'(cnt_kt_q) * 32'(cfg_gemm_k_blocks_i) +
                                                                         32'(cnt_gk_q)) * 32'h200));

    wire [ADDR_WIDTH-1:0] act_tile_tcdm_addr = a_cmp_base +
        (((cfg_act_bank_layout_i
           ? (32'(cnt_gm_q) * total_k_blocks)
           : ((32'(cnt_gm_q) * 32'd16) * total_k_blocks)) +
          32'(cnt_kt_q) * 32'(cfg_gemm_k_blocks_i) +
          32'(cnt_gk_q)) * 32'h200);

    // -------------------------------------------------------------------------
    // 1.1 State Register (always_ff)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cmp_state_q <= CMP_IDLE;
        else
            cmp_state_q <= cmp_state_d;
    end

    // -------------------------------------------------------------------------
    // 1.2 Next-State Logic (always_comb)
    // -------------------------------------------------------------------------
    always_comb begin
        cmp_state_d = cmp_state_q;

        case (cmp_state_q)
            CMP_IDLE: begin
                if (start_i)
                    cmp_state_d = CMP_O_INIT;
            end

            CMP_O_INIT: begin
                if (!macro_dma_en || (!wo_busy_i && !v_busy_i && !dr_busy &&
                                      (inflight_out_cnt_q == 4'd0) && !o_buf_wb_busy_q[o_cmp_sel_q])) begin
                    cmp_state_d = CMP_KT_LAUNCH;
                end
            end

            CMP_KT_LAUNCH: begin
                if (w_buf_ready && a_buf_ready && !act_busy_i && !wt_busy_i)
                    cmp_state_d = CMP_KT_RUN;
            end

            CMP_KT_RUN: begin
                if (gemm_compute_fire && last_gbeat && last_gk) begin
                    if (last_kt && last_gm && last_o_col)
                        cmp_state_d = macro_dma_en ? CMP_O_COMMIT : CMP_WAIT_DONE;
                    else
                        cmp_state_d = CMP_KT_LAUNCH;
                end
            end

            CMP_O_COMMIT: begin
                if (!wo_busy_i && !dr_busy && (inflight_out_cnt_q == 4'd0) &&
                    !v_busy_i && !act_busy_i && !wt_busy_i) begin
                    cmp_state_d = (last_o_chunk && last_macro_m) ? CMP_WAIT_DONE : CMP_O_INIT;
                end
            end

            CMP_WAIT_DONE: begin
                if (!dr_busy && (inflight_out_cnt_q == 4'd0) && !wo_busy_i && !v_busy_i &&
                    (!macro_dma_en || (!o_buf_wb_busy_q[0] && !o_buf_wb_busy_q[1] && !dma_busy_q))) begin
                    cmp_state_d = CMP_DONE;
                end
            end

            CMP_DONE: begin
                if (start_i)
                    cmp_state_d = CMP_O_INIT;
            end

            default: cmp_state_d = CMP_IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // 1.3 Combinational Output & Inter-FSM Signal Logic (always_comb)
    // -------------------------------------------------------------------------
    always_comb begin
        w_release_credit_pulse = 1'b0;
        o_flush_commit_pulse   = 1'b0;
        macro_a_rel_pulse      = 1'b0;

        // Activation Streamer defaults
        act_start_o            = 1'b0;
        act_base_o             = act_tile_tcdm_addr;
        act_k_blocks_o         = 32'(cfg_gemm_k_blocks_i);
        act_k_stride_o         = 32'h200;
        act_tokens_o           = cfg_gemm_decode_i ? 32'd1 : 32'd16;
        act_token_stride_o     = cfg_act_bank_layout_i ? 32'h10 : (total_k_blocks * 32'h200);
        act_ready_o            = 1'b0;

        // Weight Streamer defaults
        wt_start_o             = 1'b0;
        wt_base_o              = core_w_tcdm_addr;
        wt_repeat_o            = cfg_gemm_decode_i ? 32'd1 : 32'd16;
        wt_k_blocks_o          = 32'(cfg_gemm_k_blocks_i);
        wt_k_stride_o          = 32'h200;
        wt_num_tiles_o         = 32'd1;
        wt_tile_step_o         = 32'd0;
        wt_ready_o             = 16'h0000;

        // Write-Out Streamer & Fused V defaults
        wo_start_o             = 1'b0;
        wo_mode_o              = 2'd1;
        wo_base_o              = o_cmp_base;
        wo_num_vectors_o       = 32'd16;
        wo_line_stride_o       = 32'h200;
        wo_num_batches_o       = 32'd1;
        wo_batch_stride_o      = 32'd0;
        wo_dim2_cnt_o          = 32'd1;
        wo_dim2_stride_o       = 32'd0;
        wo_dim3_cnt_o          = 32'd1;
        wo_dim3_stride_o       = 32'd0;
        v_start_o              = 1'b0;

        // Core Compute defaults (GEMM mode)
        cfg_mode_o             = 2'b00;
        cfg_row_start_o        = 1'b0;
        cfg_key_mask_o         = 16'hFFFF;
        cfg_enable_slq_o       = 1'b1;
        lane_valid_o           = 16'h0000;
        lane_acc_idx_o         = 4'd0;

        case (cmp_state_q)
            CMP_O_INIT: begin
                if (!macro_dma_en || (!wo_busy_i && !v_busy_i && !dr_busy &&
                                      (inflight_out_cnt_q == 4'd0) && !o_buf_wb_busy_q[o_cmp_sel_q])) begin
                    wo_base_o = o_cmp_base;
                    if (cfg_out_mode_i == 2'd3) begin
                        v_start_o         = 1'b1;
                        wo_mode_o         = 2'd2;
                        wo_num_vectors_o  = 32'd16;
                        wo_line_stride_o  = 32'd16;
                        wo_num_batches_o  = 32'(cur_m_sub) * 32'(cur_o_cols);
                        wo_batch_stride_o = 32'h200;
                    end else begin
                        wo_start_o = 1'b1;
                        case (cfg_out_mode_i)
                            2'd0: begin // FP16 linear or 16-bank scatter
                                wo_mode_o = 2'd1;
                                if (cfg_out_bank_layout_i && !cfg_gemm_decode_i) begin
                                    wo_num_vectors_o  = 32'd16;
                                    wo_line_stride_o  = 32'd16;
                                    wo_num_batches_o  = 32'(cur_m_sub) * 32'(cur_o_cols);
                                    wo_batch_stride_o = 32'h800;
                                end else begin
                                    wo_num_vectors_o  = cfg_gemm_decode_i ? 32'(cur_o_cols) :
                                                        (32'(cur_m_sub) * 32'(cur_o_cols) * 32'd16);
                                end
                            end
                            2'd1: begin // FP4 linear
                                wo_mode_o        = 2'd0;
                                wo_num_vectors_o = cfg_gemm_decode_i ? 32'(cur_o_cols) :
                                                   (32'(cur_m_sub) * 32'(cur_o_cols) * 32'd16);
                            end
                            2'd2: begin // FP4 bank-scatter across banks 16..31
                                wo_mode_o         = 2'd2;
                                wo_num_vectors_o  = cfg_gemm_decode_i ? 32'd1 : 32'd16;
                                wo_line_stride_o  = 32'd16;
                                wo_num_batches_o  = 32'(cur_m_sub) * 32'(cur_o_cols);
                                wo_batch_stride_o = 32'h200;
                            end
                            default: begin
                                wo_mode_o        = 2'd1;
                                wo_num_vectors_o = cfg_gemm_decode_i ? 32'(cur_o_cols) :
                                                   (32'(cur_m_sub) * 32'(cur_o_cols) * 32'd16);
                            end
                        endcase
                    end
                end
            end

            CMP_KT_LAUNCH: begin
                if (w_buf_ready && a_buf_ready && !act_busy_i && !wt_busy_i) begin
                    wt_start_o  = 1'b1;
                    act_start_o = 1'b1;
                end
            end

            CMP_KT_RUN: begin
                if (gemm_compute_fire) begin
                    lane_valid_o   = 16'hFFFF;
                    lane_acc_idx_o = cnt_gbeat_q;
                    act_ready_o    = 1'b1;
                    wt_ready_o     = 16'hFFFF;

                    if (last_gbeat && last_gk) begin
                        // Hold W ring slot across M_sub when K_TILES==1 for M_sub x reuse
                        w_release_credit_pulse = cfg_enable_dma_i &&
                                                 (w_ring_reuse_active ? last_gm : 1'b1);
                    end
                end
            end

            CMP_O_COMMIT: begin
                if (!wo_busy_i && !dr_busy && (inflight_out_cnt_q == 4'd0) &&
                    !v_busy_i && !act_busy_i && !wt_busy_i) begin
                    o_flush_commit_pulse = 1'b1;
                    macro_a_rel_pulse    = last_o_chunk;
                end
            end

            default: ;
        endcase
    end

    // -------------------------------------------------------------------------
    // 1.4 Registered 4-Loop Counters & Ping-Pong Pointers (always_ff)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmp_m_base_q   <= 16'd0;
            o_col_start_q  <= 16'd0;
            cnt_gn_q       <= 16'd0;
            cnt_gm_q       <= 16'd0;
            cnt_kt_q       <= 8'd0;
            cnt_gk_q       <= 16'd0;
            cnt_gbeat_q    <= 4'd0;
            a_cmp_rd_sel_q <= 1'b0;
            o_cmp_sel_q    <= 1'b0;
        end else if (start_i) begin
            cmp_m_base_q   <= 16'd0;
            o_col_start_q  <= 16'd0;
            cnt_gn_q       <= 16'd0;
            cnt_gm_q       <= 16'd0;
            cnt_kt_q       <= 8'd0;
            cnt_gk_q       <= 16'd0;
            cnt_gbeat_q    <= 4'd0;
            a_cmp_rd_sel_q <= 1'b0;
            o_cmp_sel_q    <= 1'b0;
        end else begin
            // 4-Loop Compute hierarchy: beat -> gk -> kt -> gm (M_sub) -> gn (N_16)
            if (cmp_state_q == CMP_KT_RUN && gemm_compute_fire) begin
                if (last_gbeat) begin
                    cnt_gbeat_q <= 4'd0;
                    if (last_gk) begin
                        cnt_gk_q <= 16'd0;
                        if (last_kt) begin
                            cnt_kt_q <= 8'd0;
                            if (last_gm) begin
                                cnt_gm_q <= 16'd0;
                                cnt_gn_q <= last_gn ? 16'd0 : (cnt_gn_q + 16'd1);
                            end else begin
                                cnt_gm_q <= cnt_gm_q + 16'd1;
                            end
                        end else begin
                            cnt_kt_q <= cnt_kt_q + 8'd1;
                        end
                    end else begin
                        cnt_gk_q <= cnt_gk_q + 16'd1;
                    end
                end else begin
                    cnt_gbeat_q <= cnt_gbeat_q + 4'd1;
                end
            end

            // O_buf flush commit & M_macro advance (triggered at CAP_O or last_gn)
            if (o_flush_commit_pulse) begin
                o_cmp_sel_q <= ~o_cmp_sel_q;
                if (!last_o_chunk) begin
                    o_col_start_q <= o_col_start_q + cur_o_cols;
                end else begin
                    o_col_start_q  <= 16'd0;
                    a_cmp_rd_sel_q <= ~a_cmp_rd_sel_q;
                    cmp_m_base_q   <= last_macro_m ? 16'd0 : (cmp_m_base_q + eff_m_sub);
                end
            end
        end
    end

    // =========================================================================
    // SECTION 2: SUB-FSM 2 — DRAIN & OUTPUT GUARD FSM
    //   - Drains Port B slots 0..15 (or slot 0 in Decode) without blocking Compute
    //   - Tracks inflight_out_cnt_q across Core 3-stage drain pipe
    //   - Enforces WAR hazard guard (gemm_beat_ok) back to Compute FSM
    // =========================================================================
    logic [4:0] dr_slot_q, dr_slot_d;

    wire out_retire_fire = core_fp16_valid_i && core_fp16_ready_i;
    wire dr_can_issue    = core_out_fifo_ready_i &&
                           (inflight_out_cnt_q < 4'(OUT_PIPE_CREDITS));

    // -------------------------------------------------------------------------
    // 2.1 State Register (always_ff)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            dr_state_q <= DR_IDLE;
        else if (start_i)
            dr_state_q <= DR_IDLE;
        else
            dr_state_q <= dr_state_d;
    end

    // -------------------------------------------------------------------------
    // 2.2 Next-State Logic (always_comb)
    // -------------------------------------------------------------------------
    always_comb begin
        dr_state_d = dr_state_q;

        case (dr_state_q)
            DR_IDLE: begin
                if (drain_start_pulse)
                    dr_state_d = cfg_gemm_decode_i ? DR_WAIT_D1 : DR_RUN;
            end

            DR_WAIT_D1: begin
                dr_state_d = DR_RUN;
            end

            DR_RUN: begin
                if (dr_can_issue && (dr_slot_q == (cfg_gemm_decode_i ? 5'd0 : 5'd15)))
                    dr_state_d = DR_IDLE;
            end

            default: dr_state_d = DR_IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // 2.3 Combinational Output & WAR Guard Logic (always_comb)
    // -------------------------------------------------------------------------
    always_comb begin
        dr_slot_d             = dr_slot_q;
        tile_drain_req_o      = 1'b0;
        tile_drain_idx_o      = 4'd0;
        tile_clear_on_drain_o = 1'b0;

        dr_busy      = (dr_state_q != DR_IDLE);
        gemm_beat_ok = (dr_state_q == DR_IDLE) || ((dr_state_q == DR_RUN) && ({1'b0, cnt_gbeat_q} < dr_slot_q));

        case (dr_state_q)
            DR_IDLE: begin
                if (drain_start_pulse)
                    dr_slot_d = 5'd0;
            end

            DR_WAIT_D1: begin
                dr_slot_d = 5'd0;
            end

            DR_RUN: begin
                if (dr_can_issue) begin
                    tile_drain_req_o      = 1'b1;
                    tile_drain_idx_o      = dr_slot_q[3:0];
                    tile_clear_on_drain_o = 1'b1;
                    dr_slot_d             = (dr_slot_q == (cfg_gemm_decode_i ? 5'd0 : 5'd15))
                                            ? 5'd16 : (dr_slot_q + 5'd1);
                end
            end

            default: dr_slot_d = 5'd0;
        endcase
    end

    // -------------------------------------------------------------------------
    // 2.4 Registered Drain Slot & Pipeline Credit Counter (always_ff)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dr_slot_q          <= 5'd0;
            inflight_out_cnt_q <= 4'd0;
        end else if (start_i) begin
            dr_slot_q          <= 5'd0;
            inflight_out_cnt_q <= 4'd0;
        end else begin
            dr_slot_q          <= dr_slot_d;
            inflight_out_cnt_q <= inflight_out_cnt_q
                                + (tile_drain_req_o ? 4'd1 : 4'd0)
                                - (out_retire_fire  ? 4'd1 : 4'd0);
        end
    end

    // =========================================================================
    // SECTION 3: SUB-FSM 3 — DMA DESCRIPTOR & CREDIT CONTROLLER
    //   Priority: DMA_TAG_O (TCDM->DRAM) > Urgent DMA_TAG_A > Prefetch A / W
    //   Mirrors exact 4-Loop order: M_macro -> N_16 -> M_sub -> K_TILES
    // =========================================================================
    logic [PTR_W-1:0] ring_wr_ptr_q;
    logic [CNT_W-1:0] w_free_credits_q;

    dma_tag_t dma_inflight_tag_q;
    logic     dma_inflight_buf_q;

    logic [15:0] dma_w_m_base_q, dma_gn_q, dma_gm_q;
    logic [7:0]  dma_kt_q;
    logic        dma_w_pending_q;

    logic [15:0] dma_a_m_base_q;
    logic        dma_a_pending_q;
    logic        a_dma_wr_sel_q;

    logic [ADDR_WIDTH-1:0] o_buf_wb_dram_q [0:1];
    logic [31:0]           o_buf_wb_bytes_q [0:1];
    logic                  o_dma_sel_q;

    // Inter-FSM [DMA -> CMP] ready & busy status
    assign dma_busy_q  = (dma_state_q == DMA_INFLIGHT);
    assign w_buf_ready = !cfg_enable_dma_i || (w_ready_tiles_q > '0);
    assign a_buf_ready = !macro_dma_en || a_buf_ready_q[a_cmp_rd_sel_q];

    // DMA-local bounds & arbitration qualifiers
    wire [15:0] dma_w_m_rem     = cfg_gemm_m_tiles_i - dma_w_m_base_q;
    wire [15:0] dma_w_cur_m_sub = (dma_w_m_rem > eff_m_sub) ? eff_m_sub : dma_w_m_rem;
    wire [15:0] dma_a_m_rem     = cfg_gemm_m_tiles_i - dma_a_m_base_q;
    wire [15:0] dma_a_cur_m_sub = (dma_a_m_rem > eff_m_sub) ? eff_m_sub : dma_a_m_rem;

    wire [ADDR_WIDTH-1:0] dma_w_tcdm_addr = cfg_tcdm_ring_base_i + ({16'd0, ring_wr_ptr_q} * SLOT_SIZE_BYTES);
    wire                  dma_arb_en      = (dma_state_q == DMA_IDLE) && dma_window_active;

    wire dma_o_wb_req  = dma_arb_en && macro_dma_en && o_buf_wb_busy_q[o_dma_sel_q];
    wire dma_a_can_req = dma_arb_en && macro_dma_en && dma_a_pending_q && !a_buf_ready_q[a_dma_wr_sel_q];
    wire dma_a_urg_req = dma_a_can_req && !a_buf_ready_q[a_cmp_rd_sel_q];
    wire dma_w_can_req = dma_arb_en && cfg_enable_dma_i && dma_w_pending_q && (w_free_credits_q > '0);
    wire dma_w_urg_req = dma_w_can_req && (w_ready_tiles_q == '0);

    logic     dma_issue_fire;
    logic     dma_ring_issue_fire;
    dma_tag_t dma_issue_tag;
    logic     dma_issue_buf;
    wire      dma_ring_done = dma_done_i && (!macro_dma_en || (dma_inflight_tag_q == DMA_TAG_W));

    // -------------------------------------------------------------------------
    // 3.1 State Register (always_ff)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            dma_state_q <= DMA_IDLE;
        else if (start_i)
            dma_state_q <= DMA_IDLE;
        else
            dma_state_q <= dma_state_d;
    end

    // -------------------------------------------------------------------------
    // 3.2 Next-State Logic (always_comb)
    // -------------------------------------------------------------------------
    always_comb begin
        dma_state_d = dma_state_q;

        case (dma_state_q)
            DMA_IDLE: begin
                if (dma_issue_fire)
                    dma_state_d = DMA_INFLIGHT;
            end

            DMA_INFLIGHT: begin
                if (dma_done_i)
                    dma_state_d = DMA_IDLE;
            end

            default: dma_state_d = DMA_IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // 3.3 Combinational Arbiter & Descriptor Generation (always_comb)
    // -------------------------------------------------------------------------
    always_comb begin
        dma_req_valid_o      = 1'b0;
        dma_req_dir_o        = 2'd0;
        dma_req_dram_addr_o  = cfg_dram_kv_base_i;
        dma_req_tcdm_addr_o  = dma_w_tcdm_addr;
        dma_req_size_bytes_o = SLOT_SIZE_BYTES;
        dma_issue_fire       = 1'b0;
        dma_ring_issue_fire  = 1'b0;
        dma_issue_tag        = DMA_TAG_W;
        dma_issue_buf        = 1'b0;

        if (dma_o_wb_req) begin
            dma_req_valid_o      = 1'b1;
            dma_req_dir_o        = o_dma_is_16bank ? 2'd1 : 2'd3;
            dma_req_dram_addr_o  = o_buf_wb_dram_q[o_dma_sel_q];
            dma_req_tcdm_addr_o  = o_cmp_base_0 + (o_dma_sel_q ? o_stride_eff : 32'd0);
            dma_req_size_bytes_o = o_buf_wb_bytes_q[o_dma_sel_q];
            dma_issue_tag        = DMA_TAG_O;
            dma_issue_buf        = o_dma_sel_q;
            dma_issue_fire       = dma_req_ready_i;
        end else if (dma_a_urg_req || (dma_a_can_req && !dma_w_urg_req)) begin
            dma_req_valid_o      = 1'b1;
            dma_req_dir_o        = cfg_act_bank_layout_i ? 2'd0 : 2'd2;
            dma_req_dram_addr_o  = cfg_dram_act_base_i +
                                   (32'(dma_a_m_base_q) * total_k_blocks *
                                    (cfg_gemm_decode_i ? 32'd9 : 32'd144));
            dma_req_tcdm_addr_o  = cfg_tcdm_q_base_i + (a_dma_wr_sel_q ? a_stride_eff : 32'd0);
            dma_req_size_bytes_o = 32'(dma_a_cur_m_sub) * total_k_blocks * (cfg_gemm_decode_i ? 32'd9 : 32'd144);
            dma_issue_tag        = DMA_TAG_A;
            dma_issue_buf        = a_dma_wr_sel_q;
            dma_issue_fire       = dma_req_ready_i;
        end else if (dma_w_can_req) begin
            dma_req_valid_o      = 1'b1;
            dma_req_dir_o        = 2'd0;
            dma_req_dram_addr_o  = cfg_dram_kv_base_i +
                                   (32'(dma_gn_q) * cfg_stride_tile_i) +
                                   (32'(dma_kt_q) * 32'(cfg_gemm_k_blocks_i) * 32'd144);
            dma_req_tcdm_addr_o  = dma_w_tcdm_addr;
            dma_req_size_bytes_o = 32'(cfg_gemm_k_blocks_i) * 32'd144;
            dma_issue_tag        = DMA_TAG_W;
            dma_issue_fire       = dma_req_ready_i;
            dma_ring_issue_fire  = dma_req_ready_i;
        end
    end

    // -------------------------------------------------------------------------
    // 3.4 Registered DMA Pointers, Ring Credits & Ping-Pong Books (always_ff)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ring_wr_ptr_q       <= '0;
            ring_rd_ptr_q       <= '0;
            w_free_credits_q    <= NUM_RING_SLOTS[CNT_W-1:0];
            w_ready_tiles_q     <= '0;

            dma_inflight_tag_q  <= DMA_TAG_W;
            dma_inflight_buf_q  <= 1'b0;

            dma_w_m_base_q      <= 16'd0;
            dma_gn_q            <= 16'd0;
            dma_gm_q            <= 16'd0;
            dma_kt_q            <= 8'd0;
            dma_w_pending_q     <= 1'b0;

            dma_a_m_base_q      <= 16'd0;
            dma_a_pending_q     <= 1'b0;
            a_buf_ready_q       <= 2'b00;
            a_dma_wr_sel_q      <= 1'b0;

            o_buf_wb_busy_q     <= 2'b00;
            o_buf_wb_dram_q[0]  <= '0;
            o_buf_wb_dram_q[1]  <= '0;
            o_buf_wb_bytes_q[0] <= 32'd0;
            o_buf_wb_bytes_q[1] <= 32'd0;
            o_dma_sel_q         <= 1'b0;
        end else if (start_i) begin
            ring_wr_ptr_q       <= '0;
            ring_rd_ptr_q       <= '0;
            w_free_credits_q    <= NUM_RING_SLOTS[CNT_W-1:0];
            w_ready_tiles_q     <= '0;

            dma_inflight_tag_q  <= DMA_TAG_W;
            dma_inflight_buf_q  <= 1'b0;

            dma_w_m_base_q      <= 16'd0;
            dma_gn_q            <= 16'd0;
            dma_gm_q            <= 16'd0;
            dma_kt_q            <= 8'd0;
            dma_w_pending_q     <= cfg_enable_dma_i;

            dma_a_m_base_q      <= 16'd0;
            dma_a_pending_q     <= macro_dma_en;
            a_buf_ready_q       <= 2'b00;
            a_dma_wr_sel_q      <= 1'b0;

            o_buf_wb_busy_q     <= 2'b00;
            o_dma_sel_q         <= 1'b0;
        end else begin
            // Weight Ring Credits & Pointers ([DMA] producer / [CMP] consumer)
            if (dma_ring_issue_fire && !w_release_credit_pulse)
                w_free_credits_q <= w_free_credits_q - 1'b1;
            else if (!dma_ring_issue_fire && w_release_credit_pulse)
                w_free_credits_q <= w_free_credits_q + 1'b1;

            if (dma_ring_issue_fire)
                ring_wr_ptr_q <= (ring_wr_ptr_q == PTR_W'(NUM_RING_SLOTS - 1)) ? '0 : ring_wr_ptr_q + 1'b1;

            if (dma_ring_done && !w_release_credit_pulse)
                w_ready_tiles_q <= w_ready_tiles_q + 1'b1;
            else if (!dma_ring_done && w_release_credit_pulse)
                w_ready_tiles_q <= w_ready_tiles_q - 1'b1;

            if (w_release_credit_pulse)
                ring_rd_ptr_q <= (ring_rd_ptr_q == PTR_W'(NUM_RING_SLOTS - 1)) ? '0 : ring_rd_ptr_q + 1'b1;

            // In-flight Tag Capture & Ping-Pong Completion Updates
            if (dma_issue_fire) begin
                dma_inflight_tag_q <= dma_issue_tag;
                dma_inflight_buf_q <= dma_issue_buf;
            end else if (dma_done_i) begin
                if (macro_dma_en && (dma_inflight_tag_q == DMA_TAG_A)) begin
                    a_buf_ready_q[dma_inflight_buf_q] <= 1'b1;
                end else if (macro_dma_en && (dma_inflight_tag_q == DMA_TAG_O)) begin
                    o_buf_wb_busy_q[dma_inflight_buf_q] <= 1'b0;
                    o_dma_sel_q                         <= ~o_dma_sel_q;
                end
            end

            // Inter-FSM Handoffs from [CMP]
            if (macro_a_rel_pulse)
                a_buf_ready_q[a_cmp_rd_sel_q] <= 1'b0;

            if (o_flush_commit_pulse) begin
                o_buf_wb_busy_q[o_cmp_sel_q]  <= 1'b1;
                o_buf_wb_dram_q[o_cmp_sel_q]  <= cfg_dram_out_base_i +
                    ((32'(cmp_m_base_q) * 32'(cfg_gemm_n_tiles_i) +
                      32'(o_col_start_q) * 32'(cur_m_sub)) * gemm_tile_out_bytes);
                o_buf_wb_bytes_q[o_cmp_sel_q] <= 32'(cur_m_sub) * 32'(cur_o_cols) * gemm_tile_out_bytes;
            end

            // DMA Activation Prefetch Progression (M_macro batches)
            if (dma_issue_fire && macro_dma_en && (dma_issue_tag == DMA_TAG_A) && dma_a_pending_q) begin
                a_dma_wr_sel_q <= ~a_dma_wr_sel_q;
                if ((dma_a_m_base_q + dma_a_cur_m_sub) >= cfg_gemm_m_tiles_i) begin
                    dma_a_m_base_q  <= 16'd0;
                    dma_a_pending_q <= 1'b0;
                end else begin
                    dma_a_m_base_q <= dma_a_m_base_q + eff_m_sub;
                end
            end

            // DMA Weight Prefetch Progression (4-Loop: M_macro -> N_16 -> M_sub -> K_TILES)
            if (dma_issue_fire && (dma_issue_tag == DMA_TAG_W) && dma_w_pending_q) begin
                if (dma_kt_q == cfg_gemm_k_tiles_m1_i) begin
                    dma_kt_q <= 8'd0;
                    if (w_ring_reuse_active || (dma_gm_q == (dma_w_cur_m_sub - 16'd1))) begin
                        dma_gm_q <= 16'd0;
                        if (dma_gn_q == (cfg_gemm_n_tiles_i - 16'd1)) begin
                            dma_gn_q <= 16'd0;
                            if (!macro_dma_en || ((dma_w_m_base_q + dma_w_cur_m_sub) >= cfg_gemm_m_tiles_i)) begin
                                dma_w_m_base_q  <= 16'd0;
                                dma_w_pending_q <= 1'b0;
                            end else begin
                                dma_w_m_base_q <= dma_w_m_base_q + eff_m_sub;
                            end
                        end else begin
                            dma_gn_q <= dma_gn_q + 16'd1;
                        end
                    end else begin
                        dma_gm_q <= dma_gm_q + 16'd1;
                    end
                end else begin
                    dma_kt_q <= dma_kt_q + 8'd1;
                end
            end
        end
    end

    // =========================================================================
    // SECTION 4: TOP-LEVEL STATUS (Combines Compute + Drain + DMA WB)
    // =========================================================================
    wire cmp_busy      = dma_window_active;
    wire macro_wb_busy = macro_dma_en && ((o_buf_wb_busy_q != 2'b00) || dma_busy_q);

    assign busy_o = cmp_busy || dr_busy || (inflight_out_cnt_q != 4'd0) || v_busy_i || macro_wb_busy;
    assign done_o = (cmp_state_q == CMP_DONE) && !dr_busy && (inflight_out_cnt_q == 4'd0) &&
                    !v_busy_i && !macro_wb_busy;

endmodule
