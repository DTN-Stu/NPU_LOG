// =============================================================================
// Module: snax_llm_sequencer.sv
// Description: Unified Hardware Orchestration Sequencer for SNAX FP4 LLM
//              Subsystem (Single Sequencer Architecture).
//
// Supported Workloads:
//   - cfg_dma_wb_i == 1          : Standalone Gearbox DMA (TCDM <-> DRAM)
//   - cfg_op_mode_i == 2'b00     : 4-Loop Output-Stationary GEMM
//                                  (M_macro -> N_16 -> M_sub -> K_TILES)
//                                  with Ping-Pong A/O Macro-DMA, W Ring Reuse,
//                                  Port-B Rolling Drain, and Fused V-Transpose
//   - cfg_op_mode_i == 2'b01     : FlashAttention + GQA + Online Softmax
//                                  (h_kv -> h_grp -> Q_macro -> qi -> dvi -> kj)
//                                  with KV-Pinning, Ring Prefetch, Macro-DMA
//                                  Q/O Ping-Pong, and Dynamic Causal Masking
//   - cfg_op_mode_i == 2'b10     : Hardware Decode / TCDM V-Transpose + LogQ
//                                  (u_read_fp16 2D -> u_transpose_buffer ->
//                                   u_log_quantizer -> u_write_out MODE_V)
//   - cfg_op_mode_i == 2'b11     : TCDM FP16 Readback + Requantize
//
// Decoupled 3-Sub-FSM Architecture:
//   - Sub-FSM 1 (cmp_state_q): Compute, Streamer, V/RD & Loop Coordinator
//   - Sub-FSM 2 (dr_state_q) : Port-B Drain & OUT_PIPE_CREDITS Guard + WAR Guard
//   - Sub-FSM 3 (dma_state_q): DMA Arbiter & Ring/Ping-Pong/KV-Pin Controller
// =============================================================================

`timescale 1ns/1ps

module snax_llm_sequencer #(
    parameter int unsigned ADDR_WIDTH       = 32,
    parameter int unsigned NUM_LANES        = 16,
    parameter int unsigned NUM_RING_SLOTS   = 4,
    parameter int unsigned SLOT_SIZE_BYTES  = 128 * 1024,
    parameter int unsigned OUT_PIPE_CREDITS = 4,
    parameter int unsigned CAP_O_TILES      = 16
)(
    input  logic                                clk,
    input  logic                                rst_n,

    // =========================================================================
    // 1. JOB CONTROL & CONFIGURATION DESCRIPTOR
    // =========================================================================
    input  logic                                start_i,
    output logic                                busy_o,
    output logic                                done_o,

    input  logic [1:0]                          cfg_op_mode_i,
    input  logic                                cfg_is_causal_i,
    input  logic                                cfg_enable_dma_i,
    input  logic                                cfg_dma_wb_i,
    input  logic                                cfg_gemm_decode_i,
    input  logic [1:0]                          cfg_out_mode_i,
    input  logic [7:0]                          cfg_gemm_k_tiles_m1_i,
    input  logic [7:0]                          cfg_gqa_group_m1_i,
    input  logic                                cfg_act_bank_layout_i,
    input  logic                                cfg_out_bank_layout_i,

    // Attention dimensions
    input  logic [7:0]                          cfg_attn_h_i,
    input  logic [15:0]                         cfg_attn_q_tiles_i,
    input  logic [3:0]                          cfg_attn_m_q_i,
    input  logic [15:0]                         cfg_attn_dv_tiles_i,
    input  logic [15:0]                         cfg_attn_kv_tiles_i,
    input  logic [15:0]                         cfg_attn_k_blocks_i,

    // GEMM dimensions
    input  logic [15:0]                         cfg_gemm_n_tiles_i,
    input  logic [15:0]                         cfg_gemm_m_tiles_i,
    input  logic [15:0]                         cfg_gemm_k_blocks_i,

    // Base addresses & strides
    input  logic [ADDR_WIDTH-1:0]               cfg_tcdm_ring_base_i,
    input  logic [ADDR_WIDTH-1:0]               cfg_tcdm_q_base_i,
    input  logic [ADDR_WIDTH-1:0]               cfg_tcdm_out_base_i,
    input  logic [ADDR_WIDTH-1:0]               cfg_dram_kv_base_i,
    input  logic [31:0]                         cfg_stride_tile_i,

    // V-Transpose / Readback descriptors
    input  logic [ADDR_WIDTH-1:0]               cfg_v_base_i,
    input  logic [31:0]                         cfg_v_num_batches_i,
    input  logic [ADDR_WIDTH-1:0]               cfg_v_out_base_i,
    input  logic [ADDR_WIDTH-1:0]               cfg_rd_base_i,
    input  logic [31:0]                         cfg_rd_num_vectors_i,
    input  logic [31:0]                         cfg_rd_line_stride_i,
    input  logic [ADDR_WIDTH-1:0]               cfg_rd_out_base_i,

    // Macro-Tiling & Attention Extended DMA controls
    input  logic                                cfg_gemm_macro_dma_i     = 1'b0,
    input  logic [7:0]                          cfg_gemm_m_sub_tiles_i   = 8'd0,
    input  logic [7:0]                          cfg_gemm_n_sub_tiles_i   = 8'd0,
    input  logic [ADDR_WIDTH-1:0]               cfg_dram_act_base_i      = '0,
    input  logic [ADDR_WIDTH-1:0]               cfg_dram_out_base_i      = '0,
    input  logic [31:0]                         cfg_macro_a_buf_stride_i = 32'd0,
    input  logic [31:0]                         cfg_macro_o_buf_stride_i = 32'd0,
    input  logic                                cfg_attn_kv_pin_i        = 1'b0,
    input  logic                                cfg_attn_sep_kv_i        = 1'b0,
    input  logic [ADDR_WIDTH-1:0]               cfg_dram_v_base_i        = '0,
    input  logic [13:0]                         cfg_attn_kv_past_tiles_i = 14'd0,

    // =========================================================================
    // 2. DMA MASTER INTERFACE
    // =========================================================================
    output logic                                dma_req_valid_o,
    input  logic                                dma_req_ready_i,
    output logic [1:0]                          dma_req_dir_o,
    output logic [ADDR_WIDTH-1:0]               dma_req_dram_addr_o,
    output logic [ADDR_WIDTH-1:0]               dma_req_tcdm_addr_o,
    output logic [31:0]                         dma_req_size_bytes_o,
    input  logic                                dma_done_i,

    // =========================================================================
    // 3. ACTIVATION STREAMER INTERFACE
    // =========================================================================
    output logic                                act_start_o,
    output logic [ADDR_WIDTH-1:0]               act_base_o,
    output logic [31:0]                         act_k_blocks_o,
    output logic [31:0]                         act_k_stride_o,
    output logic [31:0]                         act_tokens_o,
    output logic [31:0]                         act_token_stride_o,
    input  logic                                act_busy_i,
    input  logic                                act_valid_i,
    output logic                                act_ready_o,

    // =========================================================================
    // 4. 16-CHANNEL WEIGHT / KV STREAMER INTERFACE
    // =========================================================================
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

    // =========================================================================
    // 5. WRITE-OUT STREAMER INTERFACE
    // =========================================================================
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

    // =========================================================================
    // 6. V-TRANSPOSE & TCDM FP16 READBACK STREAMER INTERFACES
    // =========================================================================
    output logic                                v_start_o,
    output logic [31:0]                         v_num_batches_o,
    input  logic                                v_busy_i,
    input  logic                                v_done_i,
    output logic                                rd_start_o,
    output logic [ADDR_WIDTH-1:0]               rd_base_o,
    output logic [31:0]                         rd_num_vectors_o,
    output logic [31:0]                         rd_line_stride_o,
    output logic [31:0]                         rd_vec_stride_o,
    output logic [31:0]                         rd_num_batches_o,
    output logic [31:0]                         rd_batch_stride_o,
    input  logic                                rd_busy_i,
    input  logic                                rd_done_i,

    // =========================================================================
    // 7. COMPUTE CORE CONTROL INTERFACE
    // =========================================================================
    output logic [1:0]                          cfg_mode_o,
    output logic                                cfg_row_start_o,
    output logic [NUM_LANES-1:0]                cfg_key_mask_o,
    output logic                                cfg_enable_slq_o,
    output logic                                tile_drain_req_o,
    output logic [3:0]                          tile_drain_idx_o,
    output logic                                tile_clear_on_drain_o,
    output logic [NUM_LANES-1:0]                lane_valid_o,
    output logic [3:0]                          lane_acc_idx_o,
    input  logic                                p_fifo_empty_i,
    input  logic                                p_fifo_almost_full_i,
    input  logic                                core_fp16_valid_i,
    input  logic                                core_out_fifo_ready_i,
    input  logic                                core_fp16_ready_i = 1'b0
);

    // =========================================================================
    // SECTION 0: PARAMETERS, STATE TYPES & SHARED GEOMETRY
    // =========================================================================
    localparam int unsigned PTR_W = (NUM_RING_SLOTS > 1) ? $clog2(NUM_RING_SLOTS) : 1;
    localparam int unsigned CNT_W = $clog2(NUM_RING_SLOTS + 1);

    typedef enum logic [4:0] {
        CMP_IDLE            = 5'd0,
        CMP_O_INIT          = 5'd1,
        CMP_GEMM_LAUNCH     = 5'd2,
        CMP_GEMM_RUN        = 5'd3,
        CMP_ATTN_QK_LAUNCH  = 5'd4,
        CMP_ATTN_QK_RUN     = 5'd5,
        CMP_ATTN_SM_DRAIN   = 5'd6,
        CMP_ATTN_PV_LAUNCH  = 5'd7,
        CMP_ATTN_PV_RUN     = 5'd8,
        CMP_ATTN_PV_DRAIN   = 5'd9,
        CMP_O_COMMIT        = 5'd10,
        CMP_V_LAUNCH        = 5'd11,
        CMP_V_RUN           = 5'd12,
        CMP_RD_LAUNCH       = 5'd13,
        CMP_RD_RUN          = 5'd14,
        CMP_DMA_WB_ISSUE    = 5'd15,
        CMP_DMA_WB_WAIT     = 5'd16,
        CMP_WAIT_DONE       = 5'd17,
        CMP_DONE            = 5'd18
    } cmp_state_t;

    typedef enum logic [1:0] {
        DR_IDLE    = 2'd0,
        DR_WAIT_D1 = 2'd1,
        DR_RUN     = 2'd2
    } dr_state_t;

    typedef enum logic {
        DMA_IDLE     = 1'b0,
        DMA_INFLIGHT = 1'b1
    } dma_state_t;

    typedef enum logic [1:0] {
        DMA_TAG_W  = 2'd0,
        DMA_TAG_A  = 2'd1,
        DMA_TAG_O  = 2'd2,
        DMA_TAG_WB = 2'd3
    } dma_tag_t;

    // Mode classification
    wire is_dma_wb      = cfg_dma_wb_i;
    wire is_gemm        = !cfg_dma_wb_i && (cfg_op_mode_i == 2'b00);
    wire is_attn        = !cfg_dma_wb_i && (cfg_op_mode_i == 2'b01);
    wire is_vtrans      = !cfg_dma_wb_i && (cfg_op_mode_i == 2'b10);
    wire is_rdback      = !cfg_dma_wb_i && (cfg_op_mode_i == 2'b11);

    wire macro_dma_en        = cfg_gemm_macro_dma_i && cfg_enable_dma_i && (is_gemm || is_attn);
    wire attn_kv_pin_en      = cfg_attn_kv_pin_i && cfg_enable_dma_i && is_attn;
    wire w_ring_reuse_active = is_gemm && cfg_enable_dma_i && (cfg_gemm_k_tiles_m1_i == 8'd0);

    // Sub-FSM state registers
    cmp_state_t cmp_state_q, cmp_state_d;
    dr_state_t  dr_state_q,  dr_state_d;
    dma_state_t dma_state_q, dma_state_d;

    // Inter-FSM handshakes
    logic       drain_start_pulse;
    logic       dr_busy;
    logic [4:0] dr_slot_q, dr_slot_d;
    logic [3:0] inflight_out_cnt_q;
    logic       gemm_beat_ok;

    logic       dma_window_active;
    logic       w_release_credit_pulse;
    logic       o_flush_commit_pulse;
    logic       macro_a_rel_pulse;
    logic       kv_pin_rel_hkv_pulse;
    logic       w_buf_ready;
    logic       a_buf_ready;
    logic       dma_busy_q;

    // Macro / Ping-Pong geometry and tracking
    logic [15:0] cmp_m_base_q, o_col_start_q;
    logic        a_cmp_rd_sel_q, o_cmp_sel_q;
    logic [1:0]  a_buf_ready_q, o_buf_wb_busy_q;
    logic [CNT_W-1:0] w_ready_tiles_q;
    logic [PTR_W-1:0] ring_rd_ptr_q;

    // KV-Pin synchronization tracking
    logic [7:0]  pin_active_hkv_q;
    logic [15:0] k_pinned_cnt_q;
    logic [31:0] v_pinned_cnt_q;
    logic        kv_pin_hkv_done_q;

    // Macro M / Q_macro bounds
    wire [15:0] macro_m_total = is_attn ? cfg_attn_q_tiles_i : cfg_gemm_m_tiles_i;
    wire [15:0] eff_m_sub     = (!macro_dma_en || (cfg_gemm_m_sub_tiles_i == 8'd0) ||
                                 ({8'd0, cfg_gemm_m_sub_tiles_i} >= macro_m_total))
                                ? macro_m_total : {8'd0, cfg_gemm_m_sub_tiles_i};
    wire [15:0] cmp_m_rem     = macro_m_total - cmp_m_base_q;
    wire [15:0] cur_m_sub     = (cmp_m_rem > eff_m_sub) ? eff_m_sub : cmp_m_rem;
    wire        last_macro_m  = ((cmp_m_base_q + cur_m_sub) >= macro_m_total);

    // GEMM O-buffer chunking bounds
    wire [15:0] auto_o_cols   = (cur_m_sub <= 16'd1) ? 16'(CAP_O_TILES)     :
                                (cur_m_sub <= 16'd2) ? 16'(CAP_O_TILES / 2) :
                                (cur_m_sub <= 16'd4) ? 16'(CAP_O_TILES / 4) :
                                (cur_m_sub <= 16'd8) ? 16'(CAP_O_TILES / 8) : 16'd1;
    wire [15:0] o_max_cols    = !macro_dma_en                    ? cfg_gemm_n_tiles_i :
                                (cfg_gemm_n_sub_tiles_i != 8'd0) ? {8'd0, cfg_gemm_n_sub_tiles_i} :
                                                                   auto_o_cols;
    wire [15:0] rem_n_cols    = cfg_gemm_n_tiles_i - o_col_start_q;
    wire [15:0] cur_o_cols    = (rem_n_cols > o_max_cols) ? o_max_cols : rem_n_cols;
    wire        last_o_chunk  = ((o_col_start_q + cur_o_cols) >= cfg_gemm_n_tiles_i);

    wire [31:0] a_stride_eff   = (cfg_macro_a_buf_stride_i != 32'd0) ? cfg_macro_a_buf_stride_i : 32'h0002_0000;
    wire [31:0] o_stride_eff   = (cfg_macro_o_buf_stride_i != 32'd0) ? cfg_macro_o_buf_stride_i : 32'h0002_0000;
    wire [31:0] total_k_blocks = (32'(cfg_gemm_k_tiles_m1_i) + 32'd1) * 32'(cfg_gemm_k_blocks_i);

    wire [31:0] a_cmp_base     = cfg_tcdm_q_base_i + (macro_dma_en ? (a_cmp_rd_sel_q ? a_stride_eff : 32'd0) : 32'd0);
    wire [31:0] o_cmp_base_0   = (is_gemm && (cfg_out_mode_i == 2'd3)) ? cfg_v_out_base_i : cfg_tcdm_out_base_i;
    wire [31:0] o_cmp_base     = o_cmp_base_0 + (macro_dma_en ? (o_cmp_sel_q ? o_stride_eff : 32'd0) : 32'd0);

    wire [31:0] gemm_tile_out_bytes = cfg_gemm_decode_i
                                      ? ((cfg_out_mode_i == 2'd0) ? 32'd36  : 32'd9)
                                      : ((cfg_out_mode_i == 2'd0) ? 32'd576 : 32'd144);
    wire [31:0] attn_vec_bytes      = (cfg_out_mode_i == 2'd0) ? 32'd36 : 32'd9;
    wire        o_dma_is_16bank     = is_gemm &&
                                      ((cfg_out_mode_i == 2'd2) || (cfg_out_mode_i == 2'd3) ||
                                       ((cfg_out_mode_i == 2'd0) && cfg_out_bank_layout_i && !cfg_gemm_decode_i));

    wire [31:0] v_nb_eff = (cfg_v_num_batches_i == 32'd0) ? 32'd1 : cfg_v_num_batches_i;
    assign v_num_batches_o = is_vtrans ? v_nb_eff : (32'(cur_m_sub) * 32'(cur_o_cols));

    // =========================================================================
    // SECTION 1: SUB-FSM 1 -- COMPUTE, ATTENTION, V-TRANSPOSE & READBACK
    // =========================================================================
    // GEMM counters (4-Loop: M_macro -> N_16 -> M_sub -> K_TILES)
    logic [15:0] cnt_gn_q, cnt_gm_q, cnt_gk_q;
    logic [7:0]  cnt_kt_q;
    logic [3:0]  cnt_gbeat_q;

    // Attention counters (h_kv -> h_grp -> Q_macro -> qi -> dvi -> kj)
    logic [7:0]  cnt_h_q, cnt_hkv_q, cnt_hgrp_q;
    logic [15:0] cnt_qi_q, cnt_dvi_q, cnt_kj_q;
    logic [3:0]  cnt_m_q, cnt_dr_q;
    logic [3:0]  causal_q_idx_q, causal_m;
    logic        kv_first_q;
    logic        attn_chunk_done_q, attn_all_done_q;
    logic        kv_pin_hkv_end_q;
    logic        v_done_seen_q, rd_done_seen_q;
    int          mask_l;

    // GEMM terminal flags
    wire last_gbeat  = cfg_gemm_decode_i ? (cnt_gbeat_q == 4'd0) : (cnt_gbeat_q == 4'd15);
    wire last_gk     = (cnt_gk_q == (cfg_gemm_k_blocks_i - 16'd1));
    wire last_kt     = (cnt_kt_q == cfg_gemm_k_tiles_m1_i);
    wire last_gm     = (cnt_gm_q == (cur_m_sub - 16'd1));
    wire last_o_col  = (cnt_gn_q == (cur_o_cols - 16'd1));
    wire [15:0] abs_gn = o_col_start_q + cnt_gn_q;
    wire [15:0] abs_gm = cmp_m_base_q + cnt_gm_q;

    // Attention terminal flags
    wire [15:0] eff_qi_bound = macro_dma_en ? cur_m_sub : cfg_attn_q_tiles_i;
    wire        last_m_beat  = (cnt_m_q == (cfg_attn_m_q_i - 4'd1));
    wire        last_kj      = (cnt_kj_q == (cfg_attn_kv_tiles_i - 16'd1));
    wire        last_dvi     = (cnt_dvi_q == (cfg_attn_dv_tiles_i - 16'd1));
    wire        last_qi      = (cnt_qi_q == (eff_qi_bound - 16'd1));
    wire        last_h       = (cnt_h_q == (cfg_attn_h_i - 8'd1));
    wire        is_last_h_of_hkv = (cnt_hgrp_q == cfg_gqa_group_m1_i) || last_h;

    // TCDM address calculation: Ring vs Preloaded/KV-Pin
    wire [31:0] ring_rd_tcdm_addr = cfg_tcdm_ring_base_i + ({16'd0, ring_rd_ptr_q} * SLOT_SIZE_BYTES);

    // GEMM TCDM addresses
    wire [31:0] preload_wt_addr = cfg_tcdm_ring_base_i +
                                  ((32'(abs_gn) * total_k_blocks +
                                    32'(cnt_kt_q) * 32'(cfg_gemm_k_blocks_i)) * 32'h200);
    wire [31:0] core_wt_addr = cfg_enable_dma_i
                               ? ring_rd_tcdm_addr
                               : (preload_wt_addr + (32'(cnt_gk_q) * 32'h200));

    wire [31:0] act_tile_tcdm_addr = a_cmp_base +
                                     (((cfg_act_bank_layout_i
                                        ? (32'(cnt_gm_q) * total_k_blocks)
                                        : (32'(cnt_gm_q) * 32'd16 * total_k_blocks)) +
                                       32'(cnt_kt_q) * 32'(cfg_gemm_k_blocks_i) +
                                       32'(cnt_gk_q)) * 32'h200);

    // Attention TCDM addresses
    wire [31:0] q_blk_per_head = 32'(cfg_attn_q_tiles_i) * 32'(cfg_attn_m_q_i) * 32'(cfg_attn_k_blocks_i);
    wire [31:0] q_blk_per_tile = 32'(cfg_attn_m_q_i) * 32'(cfg_attn_k_blocks_i);
    wire [31:0] k_blk_per_head = 32'(cfg_attn_kv_tiles_i) * 32'(cfg_attn_k_blocks_i);
    wire [31:0] v_blk_per_head = 32'(cfg_attn_dv_tiles_i) * 32'(cfg_attn_kv_tiles_i);
    wire [31:0] auto_v_base    = 32'(cfg_attn_h_i) * k_blk_per_head;
    wire [31:0] v_base_block   = (auto_v_base > 32'd16) ? auto_v_base : 32'd16;

    wire [31:0] attn_q_addr = macro_dma_en
                              ? (a_cmp_base + (32'(cnt_qi_q) * q_blk_per_tile * 32'h200))
                              : (cfg_tcdm_q_base_i + ((32'(cnt_h_q) * q_blk_per_head + 32'(cnt_qi_q) * q_blk_per_tile) * 32'h200));

    wire [31:0] attn_k_pin_addr = cfg_attn_sep_kv_i
                                  ? (cfg_tcdm_ring_base_i + (32'(cnt_kj_q) * 32'(cfg_attn_k_blocks_i) * 32'h200))
                                  : (cfg_tcdm_ring_base_i + (32'(cnt_kj_q) * (32'(cfg_attn_k_blocks_i) + 32'd1) * 32'h200));

    wire [31:0] attn_v_pin_addr = cfg_attn_sep_kv_i
                                  ? (cfg_tcdm_ring_base_i + ((k_blk_per_head + 32'(cnt_dvi_q) * 32'(cfg_attn_kv_tiles_i) + 32'(cnt_kj_q)) * 32'h200))
                                  : ((cnt_dvi_q == 16'd0)
                                     ? (attn_k_pin_addr + 32'(cfg_attn_k_blocks_i) * 32'h200)
                                     : (cfg_tcdm_ring_base_i + ((32'(cfg_attn_kv_tiles_i) * (32'(cfg_attn_k_blocks_i) + 32'd1) +
                                        (32'(cnt_dvi_q) - 32'd1) * 32'(cfg_attn_kv_tiles_i) + 32'(cnt_kj_q)) * 32'h200)));

    wire [31:0] attn_k_preload_addr = cfg_tcdm_ring_base_i +
                                      ((32'(cnt_hkv_q) * k_blk_per_head + 32'(cnt_kj_q) * 32'(cfg_attn_k_blocks_i)) * 32'h200);
    wire [31:0] attn_v_preload_addr = cfg_tcdm_ring_base_i +
                                      ((v_base_block + 32'(cnt_hkv_q) * v_blk_per_head +
                                        32'(cnt_dvi_q) * 32'(cfg_attn_kv_tiles_i) + 32'(cnt_kj_q)) * 32'h200);

    wire [31:0] attn_k_addr = !cfg_enable_dma_i ? attn_k_preload_addr :
                              attn_kv_pin_en    ? attn_k_pin_addr     :
                                                  ring_rd_tcdm_addr;
    wire [31:0] attn_v_addr = !cfg_enable_dma_i ? attn_v_preload_addr :
                              attn_kv_pin_en    ? attn_v_pin_addr     :
                                                  (ring_rd_tcdm_addr + (32'(cfg_attn_k_blocks_i) * 32'h200));

    // Attention KV availability qualifiers
    wire [31:0] v_pin_need_idx = 32'(cnt_dvi_q) * 32'(cfg_attn_kv_tiles_i) + 32'(cnt_kj_q);
    wire        attn_k_ready   = !cfg_enable_dma_i ? 1'b1 :
                                 attn_kv_pin_en    ? ((cnt_hkv_q == pin_active_hkv_q) && (cnt_kj_q < k_pinned_cnt_q)) :
                                                     (w_ready_tiles_q > '0);
    wire        attn_v_ready   = !attn_kv_pin_en ? 1'b1 :
                                                   ((cnt_hkv_q == pin_active_hkv_q) && (v_pin_need_idx < v_pinned_cnt_q));

    // Compute beat fire signals
    wire qk_compute_fire   = act_valid_i && (&wt_valid_i);
    wire pv_compute_fire   = !p_fifo_empty_i && (&wt_valid_i);
    wire gemm_compute_fire = act_valid_i && (&wt_valid_i) && gemm_beat_ok;

    // Causal mask generator
    logic [NUM_LANES-1:0] auto_causal_mask;
    wire [15:0] q_abs_tile = cmp_m_base_q + cnt_qi_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            causal_q_idx_q <= 4'd0;
        else if (start_i || (cmp_state_q == CMP_ATTN_QK_LAUNCH))
            causal_q_idx_q <= 4'd0;
        else if (cmp_state_q == CMP_ATTN_SM_DRAIN)
            causal_q_idx_q <= cnt_dr_q;
    end

    always_comb begin
        causal_m = (cmp_state_q == CMP_ATTN_QK_LAUNCH || cmp_state_q == CMP_ATTN_QK_RUN)
                   ? cnt_m_q : causal_q_idx_q;
        if (!cfg_is_causal_i) begin
            auto_causal_mask = 16'hFFFF;
        end else begin
            for (mask_l = 0; mask_l < 16; mask_l = mask_l + 1) begin
                auto_causal_mask[mask_l] = ((32'(cnt_kj_q) * 32'd16 + 32'(mask_l)) <=
                                            (32'(cfg_attn_kv_past_tiles_i) * 32'd16 +
                                             32'(q_abs_tile) * 32'(cfg_attn_m_q_i) + 32'(causal_m)));
            end
        end
    end

    // Expose legacy TB-observable status wires
    wire [4:0] state_q = cmp_state_q;
    wire       nrm_busy    = dr_busy;
    wire [1:0] nrm_state_q = dr_state_q;
    wire [3:0] nrm_dr_q    = dr_slot_q[3:0];

    // -------------------------------------------------------------------------
    // 1.1 Compute State Register
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cmp_state_q <= CMP_IDLE;
        else
            cmp_state_q <= cmp_state_d;
    end

    // -------------------------------------------------------------------------
    // 1.2 Compute Next-State Logic
    // -------------------------------------------------------------------------
    always_comb begin
        cmp_state_d = cmp_state_q;

        case (cmp_state_q)
            CMP_IDLE: begin
                if (start_i) begin
                    if (is_dma_wb)       cmp_state_d = CMP_DMA_WB_ISSUE;
                    else if (is_vtrans)  cmp_state_d = CMP_V_LAUNCH;
                    else if (is_rdback)  cmp_state_d = CMP_RD_LAUNCH;
                    else                 cmp_state_d = CMP_O_INIT;
                end
            end

            CMP_O_INIT: begin
                if (is_gemm) begin
                    if (!macro_dma_en || (!wo_busy_i && !v_busy_i && !dr_busy &&
                                          (inflight_out_cnt_q == 4'd0) && !o_buf_wb_busy_q[o_cmp_sel_q])) begin
                        cmp_state_d = CMP_GEMM_LAUNCH;
                    end
                end else begin
                    // FlashAttention
                    if (macro_dma_en) begin
                        if (!wo_busy_i && !v_busy_i && !dr_busy &&
                            (inflight_out_cnt_q == 4'd0) && !o_buf_wb_busy_q[o_cmp_sel_q]) begin
                            cmp_state_d = CMP_ATTN_QK_LAUNCH;
                        end
                    end else if (!wo_busy_i) begin
                        cmp_state_d = CMP_ATTN_QK_LAUNCH;
                    end
                end
            end

            // ---------------- 4-Loop OS GEMM ----------------
            CMP_GEMM_LAUNCH: begin
                if (w_buf_ready && a_buf_ready && !act_busy_i && !wt_busy_i)
                    cmp_state_d = CMP_GEMM_RUN;
            end

            CMP_GEMM_RUN: begin
                if (gemm_compute_fire && last_gbeat && last_gk) begin
                    if (last_kt && last_gm && last_o_col)
                        cmp_state_d = macro_dma_en ? CMP_O_COMMIT : CMP_WAIT_DONE;
                    else
                        cmp_state_d = CMP_GEMM_LAUNCH;
                end
            end

            // ---------------- FlashAttention ----------------
            CMP_ATTN_QK_LAUNCH: begin
                if (attn_k_ready && a_buf_ready && !act_busy_i && !wt_busy_i)
                    cmp_state_d = CMP_ATTN_QK_RUN;
            end

            CMP_ATTN_QK_RUN: begin
                if (!act_busy_i && !wt_busy_i && !dr_busy && (inflight_out_cnt_q == 4'd0))
                    cmp_state_d = CMP_ATTN_SM_DRAIN;
            end

            CMP_ATTN_SM_DRAIN: begin
                if (cnt_dr_q == (cfg_attn_m_q_i - 4'd1))
                    cmp_state_d = CMP_ATTN_PV_LAUNCH;
            end

            CMP_ATTN_PV_LAUNCH: begin
                if (!dr_busy && !wt_busy_i && attn_v_ready)
                    cmp_state_d = CMP_ATTN_PV_RUN;
            end

            CMP_ATTN_PV_RUN: begin
                if (pv_compute_fire && last_m_beat)
                    cmp_state_d = CMP_ATTN_PV_DRAIN;
            end

            CMP_ATTN_PV_DRAIN: begin
                if (!wt_busy_i) begin
                    if (cnt_kj_q == 16'd0) begin
                        if (macro_dma_en && attn_chunk_done_q)
                            cmp_state_d = CMP_O_COMMIT;
                        else if (attn_all_done_q)
                            cmp_state_d = CMP_WAIT_DONE;
                        else
                            cmp_state_d = CMP_ATTN_QK_LAUNCH;
                    end else begin
                        cmp_state_d = CMP_ATTN_QK_LAUNCH;
                    end
                end
            end

            CMP_O_COMMIT: begin
                if (!wo_busy_i && !dr_busy && (inflight_out_cnt_q == 4'd0) &&
                    !v_busy_i && !act_busy_i && !wt_busy_i) begin
                    if (is_gemm)
                        cmp_state_d = (last_o_chunk && last_macro_m) ? CMP_WAIT_DONE : CMP_O_INIT;
                    else
                        cmp_state_d = attn_all_done_q ? CMP_WAIT_DONE : CMP_O_INIT;
                end
            end

            // ---------------- Hardware Decode / Standalone V-Transpose ----------------
            CMP_V_LAUNCH: begin
                if (!v_busy_i && !rd_busy_i && !wo_busy_i)
                    cmp_state_d = CMP_V_RUN;
            end

            CMP_V_RUN: begin
                if ((v_done_seen_q || !v_busy_i) && (rd_done_seen_q || !rd_busy_i) && !wo_busy_i)
                    cmp_state_d = CMP_WAIT_DONE;
            end

            // ---------------- TCDM Readback + Requantize ----------------
            CMP_RD_LAUNCH: begin
                if (!rd_busy_i && !wo_busy_i)
                    cmp_state_d = CMP_RD_RUN;
            end

            CMP_RD_RUN: begin
                if ((rd_done_seen_q || !rd_busy_i) && !wo_busy_i)
                    cmp_state_d = CMP_WAIT_DONE;
            end

            // ---------------- Standalone DMA Transfer ----------------
            CMP_DMA_WB_ISSUE: begin
                if ((dma_state_q == DMA_IDLE) && dma_req_ready_i)
                    cmp_state_d = CMP_DMA_WB_WAIT;
            end

            CMP_DMA_WB_WAIT: begin
                if (dma_done_i)
                    cmp_state_d = CMP_DONE;
            end

            // ---------------- Drain & Completion Barrier ----------------
            CMP_WAIT_DONE: begin
                if (!dr_busy && (inflight_out_cnt_q == 4'd0) && !wo_busy_i && !v_busy_i && !rd_busy_i &&
                    (!macro_dma_en || (!o_buf_wb_busy_q[0] && !o_buf_wb_busy_q[1])) &&
                    (!cfg_enable_dma_i || !dma_busy_q)) begin
                    cmp_state_d = CMP_DONE;
                end
            end

            CMP_DONE: begin
                if (start_i) begin
                    if (is_dma_wb)       cmp_state_d = CMP_DMA_WB_ISSUE;
                    else if (is_vtrans)  cmp_state_d = CMP_V_LAUNCH;
                    else if (is_rdback)  cmp_state_d = CMP_RD_LAUNCH;
                    else                 cmp_state_d = CMP_O_INIT;
                end
            end

            default: cmp_state_d = CMP_IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // 1.3 Compute Combinational Output & Inter-FSM Pulse Logic
    // -------------------------------------------------------------------------
    logic sm_drain_req;
    logic sm_row_start;
    logic [3:0] sm_drain_idx;

    always_comb begin
        drain_start_pulse      = 1'b0;
        w_release_credit_pulse = 1'b0;
        o_flush_commit_pulse   = 1'b0;
        macro_a_rel_pulse      = 1'b0;
        kv_pin_rel_hkv_pulse   = 1'b0;

        sm_drain_req           = 1'b0;
        sm_row_start           = 1'b0;
        sm_drain_idx           = 4'd0;

        // Activation streamer defaults
        act_start_o            = 1'b0;
        act_base_o             = act_tile_tcdm_addr;
        act_k_blocks_o         = is_attn ? 32'(cfg_attn_k_blocks_i) : 32'(cfg_gemm_k_blocks_i);
        act_k_stride_o         = 32'h200;
        act_tokens_o           = is_attn ? 32'(cfg_attn_m_q_i) : (cfg_gemm_decode_i ? 32'd1 : 32'd16);
        act_token_stride_o     = is_attn ? (32'(cfg_attn_k_blocks_i) * 32'h200) :
                                 (cfg_act_bank_layout_i ? 32'h10 : (total_k_blocks * 32'h200));
        act_ready_o            = 1'b0;

        // Weight / KV streamer defaults
        wt_start_o             = 1'b0;
        wt_base_o              = core_wt_addr;
        wt_repeat_o            = is_attn ? 32'(cfg_attn_m_q_i) : (cfg_gemm_decode_i ? 32'd1 : 32'd16);
        wt_k_blocks_o          = is_attn ? 32'(cfg_attn_k_blocks_i) : 32'(cfg_gemm_k_blocks_i);
        wt_k_stride_o          = 32'h200;
        wt_num_tiles_o         = 32'd1;
        wt_tile_step_o         = 32'd0;
        wt_ready_o             = 16'h0000;

        // Write-out streamer defaults
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

        // V-transpose & FP16 readback defaults
        v_start_o              = 1'b0;
        rd_start_o             = 1'b0;
        rd_base_o              = cfg_rd_base_i;
        rd_num_vectors_o       = cfg_rd_num_vectors_i;
        rd_line_stride_o       = cfg_rd_line_stride_i;
        rd_vec_stride_o        = 32'd0;
        rd_num_batches_o       = 32'd1;
        rd_batch_stride_o      = 32'd0;

        // Important: Never drive cfg_mode_o == 2'b00 during FlashAttention,
        // because norm_bypass_inv_l = (cfg_mode_i == 2'b00) in core.
        cfg_mode_o             = is_attn ? 2'b10 : 2'b00;
        cfg_enable_slq_o       = 1'b1;
        lane_valid_o           = 16'h0000;
        lane_acc_idx_o         = 4'd0;

        case (cmp_state_q)
            CMP_O_INIT: begin
                if (is_gemm) begin
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
                                2'd0: begin
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
                                2'd1: begin
                                    wo_mode_o        = 2'd0;
                                    wo_num_vectors_o = cfg_gemm_decode_i ? 32'(cur_o_cols) :
                                                       (32'(cur_m_sub) * 32'(cur_o_cols) * 32'd16);
                                end
                                2'd2: begin
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
                end else if (is_attn) begin
                    if (macro_dma_en ? (!wo_busy_i && !v_busy_i && !dr_busy &&
                                        (inflight_out_cnt_q == 4'd0) && !o_buf_wb_busy_q[o_cmp_sel_q])
                                     : !wo_busy_i) begin
                        wo_start_o = 1'b1;
                        wo_base_o  = o_cmp_base;
                        if (cfg_out_mode_i == 2'd2) begin
                            wo_mode_o         = 2'd3;
                            wo_num_vectors_o  = 32'(cfg_attn_m_q_i);
                            wo_line_stride_o  = (macro_dma_en ? 32'd1 : 32'(cfg_attn_h_i)) *
                                                32'(cfg_attn_dv_tiles_i) * 32'h200;
                            wo_num_batches_o  = 32'(cfg_attn_dv_tiles_i);
                            wo_batch_stride_o = 32'h200;
                            wo_dim2_cnt_o     = 32'(cur_m_sub);
                            wo_dim2_stride_o  = 32'(cfg_attn_m_q_i) *
                                                (macro_dma_en ? 32'd1 : 32'(cfg_attn_h_i)) *
                                                32'(cfg_attn_dv_tiles_i) * 32'h200;
                            wo_dim3_cnt_o     = macro_dma_en ? 32'd1 : 32'(cfg_attn_h_i);
                            wo_dim3_stride_o  = 32'(cfg_attn_dv_tiles_i) * 32'h200;
                        end else begin
                            wo_mode_o        = (cfg_out_mode_i == 2'd1) ? 2'd0 : 2'd1;
                            wo_num_vectors_o = (macro_dma_en ? 32'd1 : 32'(cfg_attn_h_i)) *
                                               32'(cur_m_sub) *
                                               32'(cfg_attn_dv_tiles_i) *
                                               32'(cfg_attn_m_q_i);
                            wo_line_stride_o  = 32'h200;
                            wo_num_batches_o  = 32'd1;
                            wo_batch_stride_o = 32'd0;
                        end
                    end
                end
            end

            // ---------------- GEMM ----------------
            CMP_GEMM_LAUNCH: begin
                if (w_buf_ready && a_buf_ready && !act_busy_i && !wt_busy_i) begin
                    wt_start_o  = 1'b1;
                    act_start_o = 1'b1;
                end
            end

            CMP_GEMM_RUN: begin
                if (gemm_compute_fire) begin
                    lane_valid_o   = 16'hFFFF;
                    lane_acc_idx_o = cnt_gbeat_q;
                    act_ready_o    = 1'b1;
                    wt_ready_o     = 16'hFFFF;

                    if (last_gbeat && last_gk) begin
                        w_release_credit_pulse = cfg_enable_dma_i &&
                                                 (w_ring_reuse_active ? last_gm : 1'b1);
                        if (last_kt)
                            drain_start_pulse = 1'b1;
                    end
                end
            end

            // ---------------- FlashAttention ----------------
            CMP_ATTN_QK_LAUNCH: begin
                cfg_mode_o = 2'b01;
                if (attn_k_ready && a_buf_ready && !act_busy_i && !wt_busy_i) begin
                    wt_start_o         = 1'b1;
                    wt_base_o          = attn_k_addr;
                    wt_repeat_o        = 32'(cfg_attn_m_q_i);
                    wt_k_blocks_o      = 32'(cfg_attn_k_blocks_i);
                    wt_k_stride_o      = 32'h200;

                    act_start_o        = 1'b1;
                    act_base_o         = attn_q_addr;
                    act_tokens_o       = 32'(cfg_attn_m_q_i);
                    act_token_stride_o = 32'(cfg_attn_k_blocks_i) * 32'h200;
                    act_k_blocks_o     = 32'(cfg_attn_k_blocks_i);
                    act_k_stride_o     = 32'h200;
                end
            end

            CMP_ATTN_QK_RUN: begin
                cfg_mode_o = 2'b01;
                if (qk_compute_fire) begin
                    lane_valid_o   = 16'hFFFF;
                    lane_acc_idx_o = {1'b1, cnt_m_q[2:0]};
                    act_ready_o    = 1'b1;
                    wt_ready_o     = 16'hFFFF;
                end
            end

            CMP_ATTN_SM_DRAIN: begin
                cfg_mode_o   = 2'b01;
                sm_drain_req = 1'b1;
                sm_drain_idx = {1'b1, cnt_dr_q[2:0]};
                sm_row_start = kv_first_q;
            end

            CMP_ATTN_PV_LAUNCH: begin
                cfg_mode_o = 2'b01;
                if (!dr_busy && !wt_busy_i && attn_v_ready) begin
                    wt_start_o    = 1'b1;
                    wt_base_o     = attn_v_addr;
                    wt_repeat_o   = 32'(cfg_attn_m_q_i);
                    wt_k_blocks_o = 32'd1;
                    wt_k_stride_o = 32'd0;
                end
            end

            CMP_ATTN_PV_RUN: begin
                cfg_mode_o = 2'b10;
                if (pv_compute_fire) begin
                    lane_valid_o   = 16'hFFFF;
                    lane_acc_idx_o = {1'b0, cnt_m_q[2:0]};
                    wt_ready_o     = 16'hFFFF;
                end
            end

            CMP_ATTN_PV_DRAIN: begin
                cfg_mode_o = 2'b10;
                wt_ready_o = 16'hFFFF;
                if (!wt_busy_i) begin
                    w_release_credit_pulse = cfg_enable_dma_i && !attn_kv_pin_en;
                    kv_pin_rel_hkv_pulse   = attn_kv_pin_en && kv_pin_hkv_end_q;
                    if (cnt_kj_q == 16'd0)
                        drain_start_pulse = 1'b1;
                end
            end

            // ---------------- Macro-DMA O Commit ----------------
            CMP_O_COMMIT: begin
                if (!wo_busy_i && !dr_busy && (inflight_out_cnt_q == 4'd0) &&
                    !v_busy_i && !act_busy_i && !wt_busy_i) begin
                    o_flush_commit_pulse = 1'b1;
                    macro_a_rel_pulse    = is_attn ? 1'b1 : last_o_chunk;
                end
            end

            // ---------------- Hardware Decode / Standalone V-Transpose ----------------
            CMP_V_LAUNCH: begin
                rd_base_o         = cfg_tcdm_q_base_i;
                rd_num_vectors_o  = 32'd16;
                rd_line_stride_o  = 32'h200;
                rd_vec_stride_o   = cfg_act_bank_layout_i
                                    ? 32'h10
                                    : ((cfg_rd_line_stride_i != 32'd0) ? cfg_rd_line_stride_i
                                                                       : (v_nb_eff * 32'h800));
                rd_num_batches_o  = v_nb_eff;
                rd_batch_stride_o = 32'h800;

                wo_mode_o         = 2'd2;
                wo_base_o         = cfg_v_out_base_i;
                wo_num_vectors_o  = 32'd16;
                wo_line_stride_o  = 32'd16;
                wo_num_batches_o  = v_nb_eff;
                wo_batch_stride_o = (cfg_stride_tile_i != 32'd0) ? cfg_stride_tile_i : 32'h200;

                if (!v_busy_i && !rd_busy_i && !wo_busy_i) begin
                    v_start_o  = 1'b1;
                    rd_start_o = 1'b1;
                end
            end

            CMP_V_RUN: begin
                // Hold descriptor lines steady during active transpose
                rd_base_o         = cfg_tcdm_q_base_i;
                rd_num_vectors_o  = 32'd16;
                rd_line_stride_o  = 32'h200;
                rd_vec_stride_o   = cfg_act_bank_layout_i
                                    ? 32'h10
                                    : ((cfg_rd_line_stride_i != 32'd0) ? cfg_rd_line_stride_i
                                                                       : (v_nb_eff * 32'h800));
                rd_num_batches_o  = v_nb_eff;
                rd_batch_stride_o = 32'h800;

                wo_mode_o         = 2'd2;
                wo_base_o         = cfg_v_out_base_i;
                wo_num_vectors_o  = 32'd16;
                wo_line_stride_o  = 32'd16;
                wo_num_batches_o  = v_nb_eff;
                wo_batch_stride_o = (cfg_stride_tile_i != 32'd0) ? cfg_stride_tile_i : 32'h200;
            end

            // ---------------- TCDM Readback + Requantize ----------------
            CMP_RD_LAUNCH: begin
                rd_base_o        = cfg_rd_base_i;
                rd_num_vectors_o = cfg_rd_num_vectors_i;
                rd_line_stride_o = cfg_rd_line_stride_i;
                wo_base_o        = cfg_rd_out_base_i;
                if (cfg_out_mode_i == 2'd3) begin
                    wo_mode_o         = 2'd3;
                    wo_num_vectors_o  = {16'd0, cfg_attn_q_tiles_i};
                    wo_line_stride_o  = cfg_tcdm_ring_base_i;
                    wo_num_batches_o  = {16'd0, cfg_attn_dv_tiles_i};
                    wo_batch_stride_o = cfg_tcdm_q_base_i;
                    wo_dim2_cnt_o     = {16'd0, cfg_attn_kv_tiles_i};
                    wo_dim2_stride_o  = cfg_tcdm_out_base_i;
                    wo_dim3_cnt_o     = {16'd0, cfg_attn_k_blocks_i};
                    wo_dim3_stride_o  = cfg_dram_kv_base_i;
                end else begin
                    wo_mode_o        = 2'd0;
                    wo_num_vectors_o = cfg_rd_num_vectors_i;
                    wo_line_stride_o = cfg_rd_line_stride_i;
                end
                if (!rd_busy_i && !wo_busy_i) begin
                    rd_start_o = 1'b1;
                    wo_start_o = 1'b1;
                end
            end

            CMP_RD_RUN: begin
                rd_base_o        = cfg_rd_base_i;
                rd_num_vectors_o = cfg_rd_num_vectors_i;
                rd_line_stride_o = cfg_rd_line_stride_i;
                wo_base_o        = cfg_rd_out_base_i;
                if (cfg_out_mode_i == 2'd3) begin
                    wo_mode_o         = 2'd3;
                    wo_num_vectors_o  = {16'd0, cfg_attn_q_tiles_i};
                    wo_line_stride_o  = cfg_tcdm_ring_base_i;
                    wo_num_batches_o  = {16'd0, cfg_attn_dv_tiles_i};
                    wo_batch_stride_o = cfg_tcdm_q_base_i;
                    wo_dim2_cnt_o     = {16'd0, cfg_attn_kv_tiles_i};
                    wo_dim2_stride_o  = cfg_tcdm_out_base_i;
                    wo_dim3_cnt_o     = {16'd0, cfg_attn_k_blocks_i};
                    wo_dim3_stride_o  = cfg_dram_kv_base_i;
                end else begin
                    wo_mode_o        = 2'd0;
                    wo_num_vectors_o = cfg_rd_num_vectors_i;
                    wo_line_stride_o = cfg_rd_line_stride_i;
                end
            end

            default: ;
        endcase
    end

    // -------------------------------------------------------------------------
    // 1.4 Registered Loop Counters & Sticky Done Flags
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmp_m_base_q      <= 16'd0;
            o_col_start_q     <= 16'd0;
            a_cmp_rd_sel_q    <= 1'b0;
            o_cmp_sel_q       <= 1'b0;

            cnt_gn_q          <= 16'd0;
            cnt_gm_q          <= 16'd0;
            cnt_kt_q          <= 8'd0;
            cnt_gk_q          <= 16'd0;
            cnt_gbeat_q       <= 4'd0;

            cnt_h_q           <= 8'd0;
            cnt_hkv_q         <= 8'd0;
            cnt_hgrp_q        <= 8'd0;
            cnt_qi_q          <= 16'd0;
            cnt_dvi_q         <= 16'd0;
            cnt_kj_q          <= 16'd0;
            cnt_m_q           <= 4'd0;
            cnt_dr_q          <= 4'd0;
            kv_first_q        <= 1'b1;
            attn_chunk_done_q <= 1'b0;
            attn_all_done_q   <= 1'b0;
            kv_pin_hkv_end_q  <= 1'b0;
            v_done_seen_q     <= 1'b0;
            rd_done_seen_q    <= 1'b0;
        end else if (start_i) begin
            cmp_m_base_q      <= 16'd0;
            o_col_start_q     <= 16'd0;
            a_cmp_rd_sel_q    <= 1'b0;
            o_cmp_sel_q       <= 1'b0;

            cnt_gn_q          <= 16'd0;
            cnt_gm_q          <= 16'd0;
            cnt_kt_q          <= 8'd0;
            cnt_gk_q          <= 16'd0;
            cnt_gbeat_q       <= 4'd0;

            cnt_h_q           <= 8'd0;
            cnt_hkv_q         <= 8'd0;
            cnt_hgrp_q        <= 8'd0;
            cnt_qi_q          <= 16'd0;
            cnt_dvi_q         <= 16'd0;
            cnt_kj_q          <= 16'd0;
            cnt_m_q           <= 4'd0;
            cnt_dr_q          <= 4'd0;
            kv_first_q        <= 1'b1;
            attn_chunk_done_q <= 1'b0;
            attn_all_done_q   <= 1'b0;
            kv_pin_hkv_end_q  <= 1'b0;
            v_done_seen_q     <= 1'b0;
            rd_done_seen_q    <= 1'b0;
        end else begin
            // 4-Loop GEMM Compute counters
            if (cmp_state_q == CMP_GEMM_RUN && gemm_compute_fire) begin
                if (last_gbeat) begin
                    cnt_gbeat_q <= 4'd0;
                    if (last_gk) begin
                        cnt_gk_q <= 16'd0;
                        if (last_kt) begin
                            cnt_kt_q <= 8'd0;
                            if (last_gm) begin
                                cnt_gm_q <= 16'd0;
                                cnt_gn_q <= last_o_col ? 16'd0 : (cnt_gn_q + 16'd1);
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

            // Attention QK/PV beat & drain slot counters
            if (cmp_state_q == CMP_ATTN_QK_LAUNCH || cmp_state_q == CMP_ATTN_PV_LAUNCH) begin
                cnt_m_q <= 4'd0;
            end else if (cmp_state_q == CMP_ATTN_QK_RUN && qk_compute_fire) begin
                cnt_m_q <= last_m_beat ? 4'd0 : (cnt_m_q + 4'd1);
            end else if (cmp_state_q == CMP_ATTN_PV_RUN && pv_compute_fire) begin
                if (last_m_beat) begin
                    cnt_m_q <= 4'd0;
                    if (!last_kj) begin
                        cnt_kj_q <= cnt_kj_q + 16'd1;
                    end else begin
                        cnt_kj_q         <= 16'd0;
                        kv_pin_hkv_end_q <= attn_kv_pin_en && last_dvi && last_qi && (!macro_dma_en || last_macro_m) && is_last_h_of_hkv;
                        if (macro_dma_en) begin
                            if (last_dvi) begin
                                cnt_dvi_q <= 16'd0;
                                if (last_qi) begin
                                    cnt_qi_q          <= 16'd0;
                                    attn_chunk_done_q <= 1'b1;
                                    attn_all_done_q   <= last_h && last_macro_m;
                                end else begin
                                    cnt_qi_q <= cnt_qi_q + 16'd1;
                                end
                            end else begin
                                cnt_dvi_q <= cnt_dvi_q + 16'd1;
                            end
                        end else begin
                            attn_all_done_q <= last_dvi && last_qi && last_h;
                            if (last_dvi) begin
                                cnt_dvi_q <= 16'd0;
                                if (last_qi) begin
                                    cnt_qi_q <= 16'd0;
                                    if (last_h) begin
                                        cnt_h_q    <= 8'd0;
                                        cnt_hkv_q  <= 8'd0;
                                        cnt_hgrp_q <= 8'd0;
                                    end else begin
                                        cnt_h_q <= cnt_h_q + 8'd1;
                                        if (cnt_hgrp_q == cfg_gqa_group_m1_i) begin
                                            cnt_hgrp_q <= 8'd0;
                                            cnt_hkv_q  <= cnt_hkv_q + 8'd1;
                                        end else begin
                                            cnt_hgrp_q <= cnt_hgrp_q + 8'd1;
                                        end
                                    end
                                end else begin
                                    cnt_qi_q <= cnt_qi_q + 16'd1;
                                end
                            end else begin
                                cnt_dvi_q <= cnt_dvi_q + 16'd1;
                            end
                        end
                    end
                end else begin
                    cnt_m_q <= cnt_m_q + 4'd1;
                end
            end

            if (cmp_state_q == CMP_ATTN_QK_RUN && !act_busy_i && !wt_busy_i && !dr_busy && (inflight_out_cnt_q == 4'd0))
                cnt_dr_q <= 4'd0;
            else if (cmp_state_q == CMP_ATTN_SM_DRAIN)
                cnt_dr_q <= cnt_dr_q + 4'd1;

            if (cmp_state_q == CMP_ATTN_PV_DRAIN && !wt_busy_i) begin
                kv_first_q       <= (cnt_kj_q == 16'd0);
                kv_pin_hkv_end_q <= 1'b0;
            end

            // Macro-DMA chunk commit & Ping-Pong bank toggle
            if (o_flush_commit_pulse) begin
                o_cmp_sel_q <= ~o_cmp_sel_q;
                if (is_gemm) begin
                    if (!last_o_chunk) begin
                        o_col_start_q <= o_col_start_q + cur_o_cols;
                    end else begin
                        o_col_start_q  <= 16'd0;
                        a_cmp_rd_sel_q <= ~a_cmp_rd_sel_q;
                        cmp_m_base_q   <= last_macro_m ? 16'd0 : (cmp_m_base_q + eff_m_sub);
                    end
                end else if (is_attn) begin
                    attn_chunk_done_q <= 1'b0;
                    a_cmp_rd_sel_q    <= ~a_cmp_rd_sel_q;
                    if (!last_macro_m) begin
                        cmp_m_base_q <= cmp_m_base_q + eff_m_sub;
                    end else begin
                        cmp_m_base_q <= 16'd0;
                        if (last_h) begin
                            cnt_h_q    <= 8'd0;
                            cnt_hkv_q  <= 8'd0;
                            cnt_hgrp_q <= 8'd0;
                        end else begin
                            cnt_h_q <= cnt_h_q + 8'd1;
                            if (cnt_hgrp_q == cfg_gqa_group_m1_i) begin
                                cnt_hgrp_q <= 8'd0;
                                cnt_hkv_q  <= cnt_hkv_q + 8'd1;
                            end else begin
                                cnt_hgrp_q <= cnt_hgrp_q + 8'd1;
                            end
                        end
                    end
                end
            end

            // Sticky completion capture for V-transpose / Readback
            if (cmp_state_q == CMP_V_LAUNCH)
                v_done_seen_q <= 1'b0;
            else if (cmp_state_q == CMP_V_RUN)
                v_done_seen_q <= v_done_seen_q | v_done_i;

            if (cmp_state_q == CMP_RD_LAUNCH)
                rd_done_seen_q <= 1'b0;
            else if (cmp_state_q == CMP_RD_RUN)
                rd_done_seen_q <= rd_done_seen_q | rd_done_i;

            if (cmp_state_q == CMP_V_RUN || cmp_state_q == CMP_RD_RUN) begin
                if (rd_done_i)
                    rd_done_seen_q <= 1'b1;
            end
        end
    end

    // =========================================================================
    // SECTION 2: SUB-FSM 2 -- SHARED PORT-B DRAIN & OUT_PIPE_CREDITS GUARD
    // =========================================================================
    wire [4:0] dr_last_slot    = is_attn ? {1'b0, cfg_attn_m_q_i - 4'd1} :
                                           (cfg_gemm_decode_i ? 5'd0 : 5'd15);
    wire       dr_needs_wait   = is_gemm && cfg_gemm_decode_i;
    wire       out_retire_fire = (is_gemm || is_attn) &&
                                 core_fp16_valid_i && core_fp16_ready_i &&
                                 (inflight_out_cnt_q != 4'd0);
    wire       dr_can_issue    = (inflight_out_cnt_q < 4'(OUT_PIPE_CREDITS)) && core_out_fifo_ready_i;

    logic      dr_req, dr_clear;
    logic [3:0] dr_idx;

    // -------------------------------------------------------------------------
    // 2.1 Drain State Register
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
    // 2.2 Drain Next-State Logic
    // -------------------------------------------------------------------------
    always_comb begin
        dr_state_d = dr_state_q;

        case (dr_state_q)
            DR_IDLE: begin
                if (drain_start_pulse)
                    dr_state_d = dr_needs_wait ? DR_WAIT_D1 : DR_RUN;
            end

            DR_WAIT_D1: begin
                dr_state_d = DR_RUN;
            end

            DR_RUN: begin
                if (dr_can_issue && (dr_slot_q == dr_last_slot))
                    dr_state_d = DR_IDLE;
            end

            default: dr_state_d = DR_IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // 2.3 Drain Combinational Outputs & WAR Hazard Guard
    // -------------------------------------------------------------------------
    always_comb begin
        dr_slot_d    = dr_slot_q;
        dr_req       = 1'b0;
        dr_idx       = 4'd0;
        dr_clear     = 1'b0;
        dr_busy      = (dr_state_q != DR_IDLE);
        gemm_beat_ok = (dr_state_q == DR_IDLE) ||
                       ((dr_state_q == DR_RUN) && ({1'b0, cnt_gbeat_q} < dr_slot_q));

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
                    dr_req    = 1'b1;
                    dr_idx    = dr_slot_q[3:0];
                    dr_clear  = 1'b1;
                    dr_slot_d = (dr_slot_q == dr_last_slot) ? 5'd16 : (dr_slot_q + 5'd1);
                end
            end

            default: dr_slot_d = 5'd0;
        endcase

        // Port-B drain mux (Attention SM drain and Background O/GEMM drain are mutually exclusive)
        if (sm_drain_req) begin
            tile_drain_req_o      = 1'b1;
            tile_drain_idx_o      = sm_drain_idx;
            tile_clear_on_drain_o = 1'b1;
            cfg_row_start_o       = sm_row_start;
        end else begin
            tile_drain_req_o      = dr_req;
            tile_drain_idx_o      = dr_idx;
            tile_clear_on_drain_o = dr_clear;
            cfg_row_start_o       = 1'b0;
        end
        cfg_key_mask_o = auto_causal_mask;
    end

    // -------------------------------------------------------------------------
    // 2.4 Registered Drain Slot & Output Pipeline Credit Counter
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
                                + (dr_req          ? 4'd1 : 4'd0)
                                - (out_retire_fire ? 4'd1 : 4'd0);
        end
    end

    // =========================================================================
    // SECTION 3: SUB-FSM 3 -- DMA ARBITER, RING, PING-PONG & KV-PIN CONTROLLER
    // =========================================================================
    logic [PTR_W-1:0] ring_wr_ptr_q;
    logic [CNT_W-1:0] w_free_credits_q;

    dma_tag_t dma_inflight_tag_q;
    logic     dma_inflight_buf_q;
    logic     dma_inflight_kv_phase_q;
    logic     dma_inflight_kv_dvi0_q;

    // GEMM / Attention W & KV prefetch counters
    logic [15:0] dma_w_m_base_q, dma_gn_q, dma_gm_q;
    logic [7:0]  dma_kt_q;
    logic [7:0]  dma_h_q, dma_hkv_q, dma_hgrp_q;
    logic [15:0] dma_qi_q, dma_dvi_q, dma_kj_q;
    logic        dma_kv_phase_q; // 0: K phase (or packed KV), 1: V phase (when sep_kv == 1)
    logic        dma_w_pending_q;

    // Macro-DMA A / Q prefetch counters
    logic [7:0]  dma_a_h_q;
    logic [15:0] dma_a_m_base_q;
    logic        dma_a_pending_q;
    logic        a_dma_wr_sel_q;

    // Ping-Pong O writeback descriptors (packed arrays)
    logic [1:0][ADDR_WIDTH-1:0] o_buf_wb_dram_q;
    logic [1:0][31:0]           o_buf_wb_bytes_q;
    logic                       o_dma_sel_q;

    assign dma_window_active = (cmp_state_q != CMP_IDLE) && (cmp_state_q != CMP_DONE) && !is_dma_wb && !is_vtrans && !is_rdback;
    assign dma_busy_q        = (dma_state_q == DMA_INFLIGHT);
    assign w_buf_ready       = !cfg_enable_dma_i || (w_ready_tiles_q > '0);
    assign a_buf_ready       = !macro_dma_en || a_buf_ready_q[a_cmp_rd_sel_q];

    wire [15:0] dma_w_m_rem     = macro_m_total - dma_w_m_base_q;
    wire [15:0] dma_w_cur_m_sub = (dma_w_m_rem > eff_m_sub) ? eff_m_sub : dma_w_m_rem;
    wire [15:0] dma_a_m_rem     = macro_m_total - dma_a_m_base_q;
    wire [15:0] dma_a_cur_m_sub = (dma_a_m_rem > eff_m_sub) ? eff_m_sub : dma_a_m_rem;

    wire [ADDR_WIDTH-1:0] dma_w_tcdm_addr = cfg_tcdm_ring_base_i + ({16'd0, ring_wr_ptr_q} * SLOT_SIZE_BYTES);
    wire                  dma_arb_en      = (dma_state_q == DMA_IDLE) && dma_window_active;

    // Arbitration requests
    wire dma_wb_req     = (cmp_state_q == CMP_DMA_WB_ISSUE) && (dma_state_q == DMA_IDLE);
    wire dma_o_wb_req   = dma_arb_en && macro_dma_en && o_buf_wb_busy_q[o_dma_sel_q];
    wire dma_a_can_req  = dma_arb_en && macro_dma_en && dma_a_pending_q && !a_buf_ready_q[a_dma_wr_sel_q];
    wire dma_a_urg_req  = dma_a_can_req && !a_buf_ready_q[a_cmp_rd_sel_q];

    // Weight / KV arbitration qualifier
    wire dma_kv_pin_can_req = dma_arb_en && attn_kv_pin_en && dma_w_pending_q &&
                              !kv_pin_hkv_done_q && (dma_hkv_q == pin_active_hkv_q);
    wire dma_ring_space_ok  = is_attn && cfg_attn_sep_kv_i && dma_kv_phase_q
                              ? 1'b1
                              : (w_free_credits_q > '0);
    wire dma_w_ring_can_req = dma_arb_en && cfg_enable_dma_i && !attn_kv_pin_en && dma_w_pending_q && dma_ring_space_ok;
    wire dma_w_can_req      = dma_kv_pin_can_req || dma_w_ring_can_req;
    wire dma_w_urg_req      = dma_w_can_req && (is_attn ? (!attn_k_ready || !attn_v_ready) : (w_ready_tiles_q == '0));

    logic     dma_issue_fire;
    logic     dma_ring_slot_reserve_fire;
    logic     dma_ring_slot_commit_fire;
    dma_tag_t dma_issue_tag;
    logic     dma_issue_buf;
    logic     dma_issue_kv_phase;
    logic     dma_issue_kv_dvi0;

    wire dma_ring_slot_done = dma_done_i && !is_dma_wb && !attn_kv_pin_en &&
                              (dma_inflight_tag_q == DMA_TAG_W) &&
                              (!is_attn || !cfg_attn_sep_kv_i || dma_inflight_kv_phase_q);

    // -------------------------------------------------------------------------
    // 3.1 DMA State Register
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
    // 3.2 DMA Next-State Logic
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
    // 3.3 DMA Combinational Arbiter & Descriptor Mux
    // -------------------------------------------------------------------------
    always_comb begin
        dma_req_valid_o            = 1'b0;
        dma_req_dir_o              = 2'd0;
        dma_req_dram_addr_o        = cfg_dram_kv_base_i;
        dma_req_tcdm_addr_o        = dma_w_tcdm_addr;
        dma_req_size_bytes_o       = SLOT_SIZE_BYTES;
        dma_issue_fire             = 1'b0;
        dma_ring_slot_reserve_fire = 1'b0;
        dma_ring_slot_commit_fire  = 1'b0;
        dma_issue_tag              = DMA_TAG_W;
        dma_issue_buf              = 1'b0;
        dma_issue_kv_phase         = 1'b0;
        dma_issue_kv_dvi0          = 1'b0;

        if (dma_wb_req) begin
            dma_req_valid_o      = 1'b1;
            dma_req_dir_o        = cfg_out_mode_i;
            dma_req_dram_addr_o  = cfg_dram_kv_base_i;
            dma_req_tcdm_addr_o  = cfg_tcdm_out_base_i;
            dma_req_size_bytes_o = cfg_stride_tile_i;
            dma_issue_tag        = DMA_TAG_WB;
            dma_issue_fire       = dma_req_ready_i;
        end else if (dma_o_wb_req) begin
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
            dma_req_tcdm_addr_o  = cfg_tcdm_q_base_i + (a_dma_wr_sel_q ? a_stride_eff : 32'd0);
            if (is_attn) begin
                dma_req_dram_addr_o  = cfg_dram_act_base_i +
                                       ((32'(dma_a_h_q) * 32'(cfg_attn_q_tiles_i) + 32'(dma_a_m_base_q)) *
                                        32'(cfg_attn_m_q_i) * 32'(cfg_attn_k_blocks_i) * 32'd9);
                dma_req_size_bytes_o = 32'(dma_a_cur_m_sub) * 32'(cfg_attn_m_q_i) * 32'(cfg_attn_k_blocks_i) * 32'd9;
            end else begin
                dma_req_dram_addr_o  = cfg_dram_act_base_i +
                                       (32'(dma_a_m_base_q) * total_k_blocks * (cfg_gemm_decode_i ? 32'd9 : 32'd144));
                dma_req_size_bytes_o = 32'(dma_a_cur_m_sub) * total_k_blocks * (cfg_gemm_decode_i ? 32'd9 : 32'd144);
            end
            dma_issue_tag  = DMA_TAG_A;
            dma_issue_buf  = a_dma_wr_sel_q;
            dma_issue_fire = dma_req_ready_i;
        end else if (dma_w_can_req) begin
            dma_req_valid_o = 1'b1;
            dma_req_dir_o   = 2'd0;
            dma_issue_tag   = DMA_TAG_W;
            dma_issue_fire  = dma_req_ready_i;

            if (is_gemm) begin
                dma_req_dram_addr_o        = cfg_dram_kv_base_i +
                                             (32'(dma_gn_q) * cfg_stride_tile_i) +
                                             (32'(dma_kt_q) * 32'(cfg_gemm_k_blocks_i) * 32'd144);
                dma_req_tcdm_addr_o        = dma_w_tcdm_addr;
                dma_req_size_bytes_o       = 32'(cfg_gemm_k_blocks_i) * 32'd144;
                dma_ring_slot_reserve_fire = dma_req_ready_i;
                dma_ring_slot_commit_fire  = dma_req_ready_i;
            end else if (attn_kv_pin_en) begin
                dma_issue_kv_phase = dma_kv_phase_q;
                dma_issue_kv_dvi0  = (dma_dvi_q == 16'd0);
                if (cfg_attn_sep_kv_i) begin
                    if (!dma_kv_phase_q) begin
                        // Separate K phase
                        dma_req_dram_addr_o  = cfg_dram_kv_base_i +
                                               ((32'(dma_hkv_q) * 32'(cfg_attn_kv_tiles_i) + 32'(dma_kj_q)) *
                                                32'(cfg_attn_k_blocks_i) * 32'd144);
                        dma_req_tcdm_addr_o  = cfg_tcdm_ring_base_i +
                                               (32'(dma_kj_q) * 32'(cfg_attn_k_blocks_i) * 32'h200);
                        dma_req_size_bytes_o = 32'(cfg_attn_k_blocks_i) * 32'd144;
                    end else begin
                        // Separate V phase
                        dma_req_dram_addr_o  = cfg_dram_v_base_i +
                                               (((32'(dma_hkv_q) * 32'(cfg_attn_dv_tiles_i) + 32'(dma_dvi_q)) *
                                                 32'(cfg_attn_kv_tiles_i) + 32'(dma_kj_q)) * 32'd144);
                        dma_req_tcdm_addr_o  = cfg_tcdm_ring_base_i +
                                               ((k_blk_per_head + 32'(dma_dvi_q) * 32'(cfg_attn_kv_tiles_i) +
                                                 32'(dma_kj_q)) * 32'h200);
                        dma_req_size_bytes_o = 32'd144;
                    end
                end else begin
                    // Packed (K, V) KV-Pin: dvi==0 loads K+V0 once; dvi>=1 loads only V[dvi]
                    if (dma_dvi_q == 16'd0) begin
                        dma_req_dram_addr_o  = cfg_dram_kv_base_i +
                                               (((32'(dma_hkv_q) * 32'(cfg_attn_dv_tiles_i) + 32'd0) *
                                                 32'(cfg_attn_kv_tiles_i) + 32'(dma_kj_q)) * cfg_stride_tile_i);
                        dma_req_tcdm_addr_o  = cfg_tcdm_ring_base_i +
                                               (32'(dma_kj_q) * (32'(cfg_attn_k_blocks_i) + 32'd1) * 32'h200);
                        dma_req_size_bytes_o = (32'(cfg_attn_k_blocks_i) + 32'd1) * 32'd144;
                    end else begin
                        dma_req_dram_addr_o  = cfg_dram_kv_base_i +
                                               (((32'(dma_hkv_q) * 32'(cfg_attn_dv_tiles_i) + 32'(dma_dvi_q)) *
                                                 32'(cfg_attn_kv_tiles_i) + 32'(dma_kj_q)) * cfg_stride_tile_i) +
                                               (32'(cfg_attn_k_blocks_i) * 32'd144);
                        dma_req_tcdm_addr_o  = cfg_tcdm_ring_base_i +
                                               ((32'(cfg_attn_kv_tiles_i) * (32'(cfg_attn_k_blocks_i) + 32'd1) +
                                                 (32'(dma_dvi_q) - 32'd1) * 32'(cfg_attn_kv_tiles_i) + 32'(dma_kj_q)) * 32'h200);
                        dma_req_size_bytes_o = 32'd144;
                    end
                end
            end else begin
                // Ring mode Attention (K, V)
                dma_issue_kv_phase = dma_kv_phase_q;
                if (cfg_attn_sep_kv_i) begin
                    if (!dma_kv_phase_q) begin
                        dma_req_dram_addr_o        = cfg_dram_kv_base_i +
                                                     ((32'(dma_hkv_q) * 32'(cfg_attn_kv_tiles_i) + 32'(dma_kj_q)) *
                                                      32'(cfg_attn_k_blocks_i) * 32'd144);
                        dma_req_tcdm_addr_o        = dma_w_tcdm_addr;
                        dma_req_size_bytes_o       = 32'(cfg_attn_k_blocks_i) * 32'd144;
                        dma_ring_slot_reserve_fire = dma_req_ready_i;
                    end else begin
                        dma_req_dram_addr_o        = cfg_dram_v_base_i +
                                                     (((32'(dma_hkv_q) * 32'(cfg_attn_dv_tiles_i) + 32'(dma_dvi_q)) *
                                                       32'(cfg_attn_kv_tiles_i) + 32'(dma_kj_q)) * 32'd144);
                        dma_req_tcdm_addr_o        = dma_w_tcdm_addr + (32'(cfg_attn_k_blocks_i) * 32'h200);
                        dma_req_size_bytes_o       = 32'd144;
                        dma_ring_slot_commit_fire  = dma_req_ready_i;
                    end
                end else begin
                    dma_req_dram_addr_o        = cfg_dram_kv_base_i +
                                                 (((32'(dma_hkv_q) * 32'(cfg_attn_dv_tiles_i) + 32'(dma_dvi_q)) *
                                                   32'(cfg_attn_kv_tiles_i) + 32'(dma_kj_q)) * cfg_stride_tile_i);
                    dma_req_tcdm_addr_o        = dma_w_tcdm_addr;
                    dma_req_size_bytes_o       = (32'(cfg_attn_k_blocks_i) + 32'd1) * 32'd144;
                    dma_ring_slot_reserve_fire = dma_req_ready_i;
                    dma_ring_slot_commit_fire  = dma_req_ready_i;
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // 3.4 Registered DMA Pointers, Ring Credits, Ping-Pong & KV-Pin Books
    // -------------------------------------------------------------------------
    wire attn_next_hkv_done = ((32'(dma_hkv_q) + 32'd1) * (32'(cfg_gqa_group_m1_i) + 32'd1)) >= 32'(cfg_attn_h_i);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ring_wr_ptr_q          <= '0;
            ring_rd_ptr_q          <= '0;
            w_free_credits_q       <= NUM_RING_SLOTS[CNT_W-1:0];
            w_ready_tiles_q        <= '0;

            dma_inflight_tag_q     <= DMA_TAG_W;
            dma_inflight_buf_q     <= 1'b0;
            dma_inflight_kv_phase_q<= 1'b0;
            dma_inflight_kv_dvi0_q <= 1'b0;

            dma_w_m_base_q         <= 16'd0;
            dma_gn_q               <= 16'd0;
            dma_gm_q               <= 16'd0;
            dma_kt_q               <= 8'd0;

            dma_h_q                <= 8'd0;
            dma_hkv_q              <= 8'd0;
            dma_hgrp_q             <= 8'd0;
            dma_qi_q               <= 16'd0;
            dma_dvi_q              <= 16'd0;
            dma_kj_q               <= 16'd0;
            dma_kv_phase_q         <= 1'b0;
            dma_w_pending_q        <= 1'b0;

            dma_a_h_q              <= 8'd0;
            dma_a_m_base_q         <= 16'd0;
            dma_a_pending_q        <= 1'b0;
            a_buf_ready_q          <= 2'b00;
            a_dma_wr_sel_q         <= 1'b0;

            o_buf_wb_busy_q        <= 2'b00;
            o_buf_wb_dram_q        <= '0;
            o_buf_wb_bytes_q       <= '0;
            o_dma_sel_q            <= 1'b0;

            pin_active_hkv_q       <= 8'd0;
            k_pinned_cnt_q         <= 16'd0;
            v_pinned_cnt_q         <= 32'd0;
            kv_pin_hkv_done_q      <= 1'b0;
        end else if (start_i) begin
            ring_wr_ptr_q          <= '0;
            ring_rd_ptr_q          <= '0;
            w_free_credits_q       <= NUM_RING_SLOTS[CNT_W-1:0];
            w_ready_tiles_q        <= '0;

            dma_inflight_tag_q     <= DMA_TAG_W;
            dma_inflight_buf_q     <= 1'b0;
            dma_inflight_kv_phase_q<= 1'b0;
            dma_inflight_kv_dvi0_q <= 1'b0;

            dma_w_m_base_q         <= 16'd0;
            dma_gn_q               <= 16'd0;
            dma_gm_q               <= 16'd0;
            dma_kt_q               <= 8'd0;

            dma_h_q                <= 8'd0;
            dma_hkv_q              <= 8'd0;
            dma_hgrp_q             <= 8'd0;
            dma_qi_q               <= 16'd0;
            dma_dvi_q              <= 16'd0;
            dma_kj_q               <= 16'd0;
            dma_kv_phase_q         <= 1'b0;
            dma_w_pending_q        <= cfg_enable_dma_i && (is_gemm || is_attn);

            dma_a_h_q              <= 8'd0;
            dma_a_m_base_q         <= 16'd0;
            dma_a_pending_q        <= macro_dma_en;
            a_buf_ready_q          <= 2'b00;
            a_dma_wr_sel_q         <= 1'b0;

            o_buf_wb_busy_q        <= 2'b00;
            o_dma_sel_q            <= 1'b0;

            pin_active_hkv_q       <= 8'd0;
            k_pinned_cnt_q         <= 16'd0;
            v_pinned_cnt_q         <= 32'd0;
            kv_pin_hkv_done_q      <= 1'b0;
        end else begin
            // Ring credits & pointers
            if (dma_ring_slot_reserve_fire && !w_release_credit_pulse)
                w_free_credits_q <= w_free_credits_q - 1'b1;
            else if (!dma_ring_slot_reserve_fire && w_release_credit_pulse)
                w_free_credits_q <= w_free_credits_q + 1'b1;

            if (dma_ring_slot_commit_fire)
                ring_wr_ptr_q <= (ring_wr_ptr_q == PTR_W'(NUM_RING_SLOTS - 1)) ? '0 : ring_wr_ptr_q + 1'b1;

            if (dma_ring_slot_done && !w_release_credit_pulse)
                w_ready_tiles_q <= w_ready_tiles_q + 1'b1;
            else if (!dma_ring_slot_done && w_release_credit_pulse)
                w_ready_tiles_q <= w_ready_tiles_q - 1'b1;

            if (w_release_credit_pulse)
                ring_rd_ptr_q <= (ring_rd_ptr_q == PTR_W'(NUM_RING_SLOTS - 1)) ? '0 : ring_rd_ptr_q + 1'b1;

            // In-flight capture & completion
            if (dma_issue_fire) begin
                dma_inflight_tag_q      <= dma_issue_tag;
                dma_inflight_buf_q      <= dma_issue_buf;
                dma_inflight_kv_phase_q <= dma_issue_kv_phase;
                dma_inflight_kv_dvi0_q  <= dma_issue_kv_dvi0;
            end else if (dma_done_i && !is_dma_wb) begin
                if (macro_dma_en && (dma_inflight_tag_q == DMA_TAG_A)) begin
                    a_buf_ready_q[dma_inflight_buf_q] <= 1'b1;
                end else if (macro_dma_en && (dma_inflight_tag_q == DMA_TAG_O)) begin
                    o_buf_wb_busy_q[dma_inflight_buf_q] <= 1'b0;
                    o_dma_sel_q                         <= ~o_dma_sel_q;
                end else if (attn_kv_pin_en && (dma_inflight_tag_q == DMA_TAG_W)) begin
                    if (cfg_attn_sep_kv_i) begin
                        if (!dma_inflight_kv_phase_q)
                            k_pinned_cnt_q <= k_pinned_cnt_q + 16'd1;
                        else
                            v_pinned_cnt_q <= v_pinned_cnt_q + 32'd1;
                    end else begin
                        if (dma_inflight_kv_dvi0_q)
                            k_pinned_cnt_q <= k_pinned_cnt_q + 16'd1;
                        v_pinned_cnt_q <= v_pinned_cnt_q + 32'd1;
                    end
                end
            end

            // Release active KV-pin head to allow loading hkv + 1
            if (kv_pin_rel_hkv_pulse) begin
                pin_active_hkv_q  <= pin_active_hkv_q + 8'd1;
                k_pinned_cnt_q    <= 16'd0;
                v_pinned_cnt_q    <= 32'd0;
                kv_pin_hkv_done_q <= 1'b0;
            end

            // Inter-FSM handoffs from [CMP]
            if (macro_a_rel_pulse)
                a_buf_ready_q[a_cmp_rd_sel_q] <= 1'b0;

            if (o_flush_commit_pulse) begin
                o_buf_wb_busy_q[o_cmp_sel_q] <= 1'b1;
                if (is_gemm) begin
                    o_buf_wb_dram_q[o_cmp_sel_q] <= cfg_dram_out_base_i +
                        ((32'(cmp_m_base_q) * 32'(cfg_gemm_n_tiles_i) +
                          32'(o_col_start_q) * 32'(cur_m_sub)) * gemm_tile_out_bytes);
                    o_buf_wb_bytes_q[o_cmp_sel_q] <= 32'(cur_m_sub) * 32'(cur_o_cols) * gemm_tile_out_bytes;
                end else begin
                    o_buf_wb_dram_q[o_cmp_sel_q] <= cfg_dram_out_base_i +
                        ((32'(cnt_h_q) * 32'(cfg_attn_q_tiles_i) + 32'(cmp_m_base_q)) *
                         32'(cfg_attn_dv_tiles_i) * 32'(cfg_attn_m_q_i) * attn_vec_bytes);
                    o_buf_wb_bytes_q[o_cmp_sel_q] <= 32'(cur_m_sub) * 32'(cfg_attn_dv_tiles_i) *
                                                     32'(cfg_attn_m_q_i) * attn_vec_bytes;
                end
            end

            // DMA Activation / Query Prefetch Progression
            if (dma_issue_fire && macro_dma_en && (dma_issue_tag == DMA_TAG_A) && dma_a_pending_q) begin
                a_dma_wr_sel_q <= ~a_dma_wr_sel_q;
                if ((dma_a_m_base_q + dma_a_cur_m_sub) >= macro_m_total) begin
                    dma_a_m_base_q <= 16'd0;
                    if (is_attn && (dma_a_h_q < (cfg_attn_h_i - 8'd1)))
                        dma_a_h_q <= dma_a_h_q + 8'd1;
                    else
                        dma_a_pending_q <= 1'b0;
                end else begin
                    dma_a_m_base_q <= dma_a_m_base_q + eff_m_sub;
                end
            end

            // DMA Weight / KV Prefetch Progression
            if (dma_issue_fire && (dma_issue_tag == DMA_TAG_W) && dma_w_pending_q) begin
                if (is_gemm) begin
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
                end else if (attn_kv_pin_en) begin
                    if (cfg_attn_sep_kv_i && !dma_kv_phase_q) begin
                        if (dma_kj_q == (cfg_attn_kv_tiles_i - 16'd1)) begin
                            dma_kj_q       <= 16'd0;
                            dma_dvi_q      <= 16'd0;
                            dma_kv_phase_q <= 1'b1;
                        end else begin
                            dma_kj_q <= dma_kj_q + 16'd1;
                        end
                    end else begin
                        if (dma_kj_q == (cfg_attn_kv_tiles_i - 16'd1)) begin
                            dma_kj_q <= 16'd0;
                            if (dma_dvi_q == (cfg_attn_dv_tiles_i - 16'd1)) begin
                                dma_dvi_q      <= 16'd0;
                                dma_kv_phase_q <= 1'b0;
                                if (attn_next_hkv_done) begin
                                    dma_hkv_q       <= 8'd0;
                                    dma_w_pending_q <= 1'b0;
                                end else begin
                                    dma_hkv_q         <= dma_hkv_q + 8'd1;
                                    kv_pin_hkv_done_q <= 1'b1;
                                end
                            end else begin
                                dma_dvi_q <= dma_dvi_q + 16'd1;
                            end
                        end else begin
                            dma_kj_q <= dma_kj_q + 16'd1;
                        end
                    end
                end else begin
                    // Ring mode Attention
                    if (cfg_attn_sep_kv_i && !dma_kv_phase_q) begin
                        dma_kv_phase_q <= 1'b1;
                    end else begin
                        dma_kv_phase_q <= 1'b0;
                        if (dma_kj_q == (cfg_attn_kv_tiles_i - 16'd1)) begin
                            dma_kj_q <= 16'd0;
                            if (dma_dvi_q == (cfg_attn_dv_tiles_i - 16'd1)) begin
                                dma_dvi_q <= 16'd0;
                                if (dma_qi_q == (cfg_attn_q_tiles_i - 16'd1)) begin
                                    dma_qi_q <= 16'd0;
                                    if (dma_h_q == (cfg_attn_h_i - 8'd1)) begin
                                        dma_h_q         <= 8'd0;
                                        dma_hkv_q       <= 8'd0;
                                        dma_hgrp_q      <= 8'd0;
                                        dma_w_pending_q <= 1'b0;
                                    end else begin
                                        dma_h_q <= dma_h_q + 8'd1;
                                        if (dma_hgrp_q == cfg_gqa_group_m1_i) begin
                                            dma_hgrp_q <= 8'd0;
                                            dma_hkv_q  <= dma_hkv_q + 8'd1;
                                        end else begin
                                            dma_hgrp_q <= dma_hgrp_q + 8'd1;
                                        end
                                    end
                                end else begin
                                    dma_qi_q <= dma_qi_q + 16'd1;
                                end
                            end else begin
                                dma_dvi_q <= dma_dvi_q + 16'd1;
                            end
                        end else begin
                            dma_kj_q <= dma_kj_q + 16'd1;
                        end
                    end
                end
            end
        end
    end

    // =========================================================================
    // SECTION 4: TOP-LEVEL STATUS
    // =========================================================================
    wire cmp_active    = (cmp_state_q != CMP_IDLE) && (cmp_state_q != CMP_DONE);
    wire macro_wb_busy = macro_dma_en && ((o_buf_wb_busy_q != 2'b00) || dma_busy_q);

    assign busy_o = cmp_active || dr_busy || (inflight_out_cnt_q != 4'd0) ||
                    v_busy_i || rd_busy_i || wo_busy_i || macro_wb_busy;
    assign done_o = (cmp_state_q == CMP_DONE) && !dr_busy && (inflight_out_cnt_q == 4'd0) &&
                    !v_busy_i && !rd_busy_i && !wo_busy_i && !macro_wb_busy;

endmodule
