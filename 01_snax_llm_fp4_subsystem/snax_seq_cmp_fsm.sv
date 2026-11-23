// =============================================================================
// Module: snax_seq_cmp_fsm.sv
// Description: Sub-FSM 1 of snax_llm_sequencer -- Compute, Shared 16x16 Multiplier
//              (CMP_PARAM_INIT / CMP_O_INIT), FlashAttention, V-Transpose,
//              Readback, Compute TCDM Pointers, and Causal Mask Generator.
// =============================================================================

`timescale 1ns/1ps

module snax_seq_cmp_fsm #(
    parameter int unsigned ADDR_WIDTH     = 32,
    parameter int unsigned NUM_LANES      = 16,
    parameter int unsigned NUM_RING_SLOTS = 4,
    parameter int unsigned CAP_O_TILES    = 16,
    parameter int unsigned CNT_W          = $clog2(NUM_RING_SLOTS + 1)
)(
    input  logic                        clk,
    input  logic                        rst_n,

    input  logic                        start_i,
    input  logic                        is_dma_wb_i,
    input  logic                        is_gemm_i,
    input  logic                        is_attn_i,
    input  logic                        is_vtrans_i,
    input  logic                        is_rdback_i,
    input  logic                        cfg_enable_dma_i,
    input  logic                        macro_dma_en_i,
    input  logic                        attn_kv_pin_en_i,
    input  logic                        w_ring_reuse_active_i,

    input  logic                        cfg_gemm_decode_i,
    input  logic [1:0]                  cfg_out_mode_i,
    input  logic [7:0]                  cfg_gemm_k_tiles_m1_i,
    input  logic [7:0]                  cfg_gqa_group_m1_i,
    input  logic                        cfg_act_bank_layout_i,
    input  logic                        cfg_out_bank_layout_i,
    input  logic                        cfg_is_causal_i,

    input  logic [7:0]                  cfg_attn_h_i,
    input  logic [15:0]                 cfg_attn_q_tiles_i,
    input  logic [3:0]                  cfg_attn_m_q_i,
    input  logic [15:0]                 cfg_attn_dv_tiles_i,
    input  logic [15:0]                 cfg_attn_kv_tiles_i,
    input  logic [15:0]                 cfg_attn_k_blocks_i,
    input  logic                        cfg_attn_sep_kv_i,
    input  logic [13:0]                 cfg_attn_kv_past_tiles_i,

    input  logic [15:0]                 cfg_gemm_n_tiles_i,
    input  logic [15:0]                 cfg_gemm_m_tiles_i,
    input  logic [15:0]                 cfg_gemm_k_blocks_i,
    input  logic [7:0]                  cfg_gemm_n_sub_tiles_i,

    input  logic [ADDR_WIDTH-1:0]       cfg_tcdm_ring_base_i,
    input  logic [ADDR_WIDTH-1:0]       cfg_tcdm_q_base_i,
    input  logic [ADDR_WIDTH-1:0]       cfg_tcdm_out_base_i,
    input  logic [ADDR_WIDTH-1:0]       cfg_dram_kv_base_i,
    input  logic [31:0]                 cfg_stride_tile_i,
    input  logic [ADDR_WIDTH-1:0]       cfg_v_base_i,
    input  logic [31:0]                 cfg_v_num_batches_i,
    input  logic [ADDR_WIDTH-1:0]       cfg_v_out_base_i,
    input  logic [ADDR_WIDTH-1:0]       cfg_rd_base_i,
    input  logic [31:0]                 cfg_rd_num_vectors_i,
    input  logic [31:0]                 cfg_rd_line_stride_i,
    input  logic [ADDR_WIDTH-1:0]       cfg_rd_out_base_i,

    input  logic [15:0]                 macro_m_total_i,
    input  logic [15:0]                 eff_m_sub_i,
    input  logic [31:0]                 a_stride_eff_i,
    input  logic [31:0]                 o_stride_eff_i,

    // From Drain FSM
    input  logic                        dr_busy_i,
    input  logic                        gemm_beat_ok_i,
    input  logic [3:0]                  inflight_out_cnt_i,

    // From DMA FSM / External DMA handshake for WB wait
    input  logic                        dma_state_i,
    input  logic                        dma_busy_i,
    input  logic                        dma_req_ready_i,
    input  logic                        dma_done_i,
    input  logic                        w_buf_ready_i,
    input  logic                        a_buf_ready_i,
    input  logic [1:0]                  o_buf_wb_busy_i,
    input  logic [CNT_W-1:0]            w_ready_tiles_i,
    input  logic [19:0]                 ring_rd_off_i,
    input  logic [7:0]                  pin_active_hkv_i,
    input  logic [15:0]                 k_pinned_cnt_i,
    input  logic [31:0]                 v_pinned_cnt_i,

    // To Drain FSM
    output logic                        drain_start_pulse_o,
    output logic                        sm_drain_req_o,
    output logic [3:0]                  sm_drain_idx_o,
    output logic                        sm_row_start_o,
    output logic [NUM_LANES-1:0]        auto_causal_mask_o,

    // To DMA FSM
    output logic [4:0]                  cmp_state_o,
    output logic                        pinit_sync_o,
    output logic                        a_cmp_rd_sel_o,
    output logic                        o_cmp_sel_o,
    output logic                        last_o_chunk_o,
    output logic                        last_macro_m_o,
    output logic                        w_release_credit_pulse_o,
    output logic                        macro_a_rel_pulse_o,
    output logic                        o_flush_commit_pulse_o,
    output logic                        kv_pin_rel_hkv_pulse_o,
    output logic                        attn_k_ready_o,
    output logic                        attn_v_ready_o,

    output logic [31:0]                 o_wb_bytes_o,
    output logic [31:0]                 attn_wb_bytes_o,
    output logic [31:0]                 o_wb_macro_step_o,
    output logic [31:0]                 a_step_h_o,
    output logic [31:0]                 a_full_bytes_o,
    output logic [31:0]                 kt_stride_o,
    output logic [31:0]                 kt_stride_attn_o,
    output logic [31:0]                 sz_kv0_o,
    output logic [10:0]                 k_blk_head_blk_o,
    output logic [10:0]                 k_preload_step_blk_o,
    output logic [10:0]                 packed_k_step_blk_o,
    output logic [15:0]                 packed_v_off_idx_o,

    // TB-observable counters / flags
    output logic [15:0]                 cnt_gn_o,
    output logic [15:0]                 cnt_gm_o,
    output logic [15:0]                 cnt_gk_o,
    output logic [7:0]                  cnt_kt_o,
    output logic [3:0]                  cnt_gbeat_o,
    output logic [7:0]                  cnt_h_o,
    output logic [15:0]                 cnt_qi_o,
    output logic [15:0]                 cnt_dvi_o,
    output logic [15:0]                 cnt_kj_o,
    output logic [3:0]                  cnt_m_o,
    output logic [3:0]                  cnt_dr_o,
    output logic                        qk_compute_fire_o,
    output logic                        pv_compute_fire_o,

    // Activation Streamer
    output logic                        act_start_o,
    output logic [ADDR_WIDTH-1:0]       act_base_o,
    output logic [31:0]                 act_k_blocks_o,
    output logic [31:0]                 act_k_stride_o,
    output logic [31:0]                 act_tokens_o,
    output logic [31:0]                 act_token_stride_o,
    input  logic                        act_busy_i,
    input  logic                        act_valid_i,
    output logic                        act_ready_o,

    // 16-Channel Weight / KV Streamer
    output logic                        wt_start_o,
    output logic [ADDR_WIDTH-1:0]       wt_base_o,
    output logic [31:0]                 wt_repeat_o,
    output logic [31:0]                 wt_k_blocks_o,
    output logic [31:0]                 wt_k_stride_o,
    output logic [31:0]                 wt_num_tiles_o,
    output logic [31:0]                 wt_tile_step_o,
    input  logic                        wt_busy_i,
    input  logic [NUM_LANES-1:0]        wt_valid_i,
    output logic [NUM_LANES-1:0]        wt_ready_o,

    // Write-Out Streamer
    output logic                        wo_start_o,
    output logic [1:0]                  wo_mode_o,
    output logic [ADDR_WIDTH-1:0]       wo_base_o,
    output logic [31:0]                 wo_num_vectors_o,
    output logic [31:0]                 wo_line_stride_o,
    output logic [31:0]                 wo_num_batches_o,
    output logic [31:0]                 wo_batch_stride_o,
    output logic [31:0]                 wo_dim2_cnt_o,
    output logic [31:0]                 wo_dim2_stride_o,
    output logic [31:0]                 wo_dim3_cnt_o,
    output logic [31:0]                 wo_dim3_stride_o,
    input  logic                        wo_busy_i,
    input  logic                        wo_done_i,

    // V-Transpose & TCDM FP16 Readback Streamers
    output logic                        v_start_o,
    output logic [31:0]                 v_num_batches_o,
    input  logic                        v_busy_i,
    input  logic                        v_done_i,
    output logic                        rd_start_o,
    output logic [ADDR_WIDTH-1:0]       rd_base_o,
    output logic [31:0]                 rd_num_vectors_o,
    output logic [31:0]                 rd_line_stride_o,
    output logic [31:0]                 rd_vec_stride_o,
    output logic [31:0]                 rd_num_batches_o,
    output logic [31:0]                 rd_batch_stride_o,
    input  logic                        rd_busy_i,
    input  logic                        rd_done_i,

    // Compute Core Control Interface
    output logic [1:0]                  cfg_mode_o,
    output logic                        cfg_enable_slq_o,
    output logic [NUM_LANES-1:0]        lane_valid_o,
    output logic [3:0]                  lane_acc_idx_o,
    input  logic                        p_fifo_empty_i,
    input  logic                        p_fifo_almost_full_i
);

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
        CMP_DONE            = 5'd18,
        CMP_PARAM_INIT      = 5'd19
    } cmp_state_t;

    localparam logic DMA_IDLE = 1'b0;

    cmp_state_t  cmp_state_q, cmp_state_d;
    logic [15:0] cmp_m_base_q, o_col_start_q;
    logic        a_cmp_rd_sel_q, o_cmp_sel_q;
    logic        drain_start_pulse;
    logic        w_release_credit_pulse;
    logic        o_flush_commit_pulse;
    logic        macro_a_rel_pulse;
    logic        kv_pin_rel_hkv_pulse;

    // 32-bit x 5-bit shift-add (cur_m_sub / eff_m_sub / last_m are <= 16).
    function automatic [31:0] mul_u32_u5(input [31:0] a, input [4:0] b);
        logic [31:0] s;
        integer k;
        s = 32'd0;
        for (k = 0; k < 5; k = k + 1)
            if (b[k]) s = s + (a << k);
        return s;
    endfunction

    // 32-bit x 3-bit shift-add for tiny loop multiplicands.
    function automatic [31:0] mul_u32_u3(input [31:0] a, input [2:0] b);
        logic [31:0] s;
        integer k;
        s = 32'd0;
        for (k = 0; k < 3; k = k + 1)
            if (b[k]) s = s + (a << k);
        return s;
    endfunction

    // Pipeline byte counts expressed as shifts, never general multipliers.
    function automatic [31:0] tile_out_bytes_f(input [1:0] out_mode, input logic decode);
        logic [31:0] b;
        if (decode) b = (out_mode == 2'd0) ? 32'd36 : 32'd9;
        else        b = (out_mode == 2'd0) ? 32'd576 : 32'd144;
        return b;
    endfunction

    function automatic [31:0] mul_tile_bytes(input [31:0] a, input [1:0] out_mode, input logic decode);
        logic [31:0] r;
        if (decode)
            r = (out_mode == 2'd0) ? ((a << 5) + (a << 2)) : ((a << 3) + a);
        else
            r = (out_mode == 2'd0) ? ((a << 9) + (a << 6)) : ((a << 7) + (a << 4));
        return r;
    endfunction

    function automatic [31:0] attn_vec_bytes_f(input [1:0] out_mode);
        return (out_mode == 2'd0) ? 32'd36 : 32'd9;
    endfunction

    wire [15:0] cmp_m_rem     = macro_m_total_i - cmp_m_base_q;
    wire [15:0] cur_m_sub     = (cmp_m_rem > eff_m_sub_i) ? eff_m_sub_i : cmp_m_rem;
    wire        last_macro_m  = ((cmp_m_base_q + cur_m_sub) >= macro_m_total_i);

    wire [15:0] auto_o_cols   = (cur_m_sub <= 16'd1) ? 16'(CAP_O_TILES)     :
                                (cur_m_sub <= 16'd2) ? 16'(CAP_O_TILES / 2) :
                                (cur_m_sub <= 16'd4) ? 16'(CAP_O_TILES / 4) :
                                (cur_m_sub <= 16'd8) ? 16'(CAP_O_TILES / 8) : 16'd1;
    wire [15:0] o_max_cols    = !macro_dma_en_i                    ? cfg_gemm_n_tiles_i :
                                (cfg_gemm_n_sub_tiles_i != 8'd0) ? {8'd0, cfg_gemm_n_sub_tiles_i} :
                                                                   auto_o_cols;
    wire [15:0] rem_n_cols    = cfg_gemm_n_tiles_i - o_col_start_q;
    wire [15:0] cur_o_cols    = (rem_n_cols > o_max_cols) ? o_max_cols : rem_n_cols;
    wire        last_o_chunk  = ((o_col_start_q + cur_o_cols) >= cfg_gemm_n_tiles_i);

    // -------------------------------------------------------------------------
    // Static job parameters: computed once by a small shared-multiplier FSM.
    // -------------------------------------------------------------------------
    localparam int unsigned PINIT_N = 12;

    logic [15:0] total_k_blocks_q;
    logic [15:0] q_blk_per_tile_q;
    logic [15:0] k_blk_per_head_q;
    logic [15:0] v_blk_per_head_q;
    logic [15:0] auto_v_base_q;
    logic [15:0] packed_v_off_idx_q;
    logic [15:0] attn_wo_line_factor_q;
    logic [31:0] attn_wo_vec_factor_q;
    logic [15:0] attn_wb_dv_m_q;
    logic [15:0] o_wb_macro_inner_q;
    logic [31:0] a_step_h_q;
    logic [31:0] a_full_bytes_q, a_rem_bytes_q;
    logic [31:0] kt_stride_q, kt_stride_attn_q, sz_kv0_q;
    logic [19:0] q_tile_row_base_q;

    logic [4:0]  pinit_idx_q;
    logic        pinit_phase_q;
    logic        pinit_done_q;
    logic [15:0] mul_a_q, mul_b_q;
    logic [31:0] mul_p;
    assign mul_p = mul_a_q * mul_b_q;

    // Chunk-level products (cur_m_sub / cur_o_cols are stable inside CMP_O_INIT).
    logic [2:0]  ccalc_idx_q;
    logic        ccalc_phase_q;
    logic        ccalc_done_q;
    logic [31:0] cur_m_x_cols_q;
    logic [31:0] o_wb_bytes_q;
    logic [31:0] attn_wo_vectors_q;
    logic [31:0] attn_wb_bytes_q;
    logic [31:0] wt_chunk_inner_q;

    // Compute-side compact TCDM pointers (11-bit 512B block offsets)
    logic [10:0] act_q_blk_q;
    logic [10:0] wk_tcdm_blk_q, wk_base_blk_q;
    logic [10:0] v_tcdm_blk_q, v_base_blk_q;

    wire [31:0] auto_v_base    = {16'd0, auto_v_base_q};
    wire [10:0] v_base_blk     = (auto_v_base_q > 16'd16) ? auto_v_base_q[10:0] : 11'd16;

    wire [31:0] attn_wb_per_m    = (cfg_out_mode_i == 2'd0)
                                    ? (({16'd0, attn_wb_dv_m_q} << 5) + ({16'd0, attn_wb_dv_m_q} << 2))
                                    : (({16'd0, attn_wb_dv_m_q} << 3) + {16'd0, attn_wb_dv_m_q});
    wire [31:0] attn_wo_line_stride = {16'd0, attn_wo_line_factor_q} << 9;
    wire [31:0] attn_wo_dim2_stride = attn_wo_vec_factor_q << 9;
    wire [31:0] attn_wo_dim3_stride = {16'd0, cfg_attn_dv_tiles_i} << 9;
    wire [31:0] o_wb_macro_step     = mul_tile_bytes({16'd0, o_wb_macro_inner_q}, cfg_out_mode_i, cfg_gemm_decode_i);

    wire [10:0] wt_gn_step_blk     = total_k_blocks_q[10:0];
    wire [10:0] k_kt_step_blk      = cfg_gemm_k_blocks_i[10:0];
    wire [10:0] q_step_blk         = q_blk_per_tile_q[10:0];
    wire [10:0] k_pin_step_blk     = cfg_attn_k_blocks_i[10:0] + (cfg_attn_sep_kv_i ? 11'd0 : 11'd1);
    wire [10:0] packed_k_step_blk  = cfg_attn_k_blocks_i[10:0] + 11'd1;
    wire [10:0] k_preload_step_blk = cfg_attn_k_blocks_i[10:0];
    wire [10:0] k_blk_head_blk     = k_blk_per_head_q[10:0];
    wire [10:0] v_blk_head_blk     = v_blk_per_head_q[10:0];
    wire [10:0] act_gm_step_blk    = cfg_act_bank_layout_i
                                     ? cfg_gemm_k_blocks_i[10:0]
                                     : ((total_k_blocks_q[10:0] << 4) - total_k_blocks_q[10:0] + cfg_gemm_k_blocks_i[10:0]);

    wire [31:0] tile_out_bytes_w = tile_out_bytes_f(cfg_out_mode_i, cfg_gemm_decode_i);
    wire [31:0] pinit_a_bytes_w  = (is_attn_i || cfg_gemm_decode_i)
                                   ? ((mul_p << 3) + mul_p)
                                   : ((mul_p << 7) + (mul_p << 4));

    always_comb begin
        mul_a_q = 16'd0;
        mul_b_q = 16'd0;
        if (!pinit_done_q) begin
        case (pinit_idx_q)
            5'd0:  begin mul_a_q = 16'(cfg_gemm_k_tiles_m1_i) + 16'd1; mul_b_q = cfg_gemm_k_blocks_i; end
            5'd1:  begin mul_a_q = {12'd0, cfg_attn_m_q_i};            mul_b_q = cfg_attn_k_blocks_i; end
            5'd2:  begin mul_a_q = cfg_attn_kv_tiles_i;                mul_b_q = cfg_attn_k_blocks_i; end
            5'd3:  begin mul_a_q = cfg_attn_dv_tiles_i;                mul_b_q = cfg_attn_kv_tiles_i; end
            5'd4:  begin mul_a_q = {8'd0, cfg_attn_h_i};               mul_b_q = k_blk_per_head_q; end
            5'd5:  begin mul_a_q = cfg_attn_kv_tiles_i;                mul_b_q = cfg_attn_k_blocks_i + 16'd1; end
            5'd6:  begin mul_a_q = macro_dma_en_i ? 16'd1 : {8'd0, cfg_attn_h_i}; mul_b_q = cfg_attn_dv_tiles_i; end
            5'd7:  begin mul_a_q = attn_wo_line_factor_q;              mul_b_q = {12'd0, cfg_attn_m_q_i}; end
            5'd8:  begin mul_a_q = cfg_attn_dv_tiles_i;                mul_b_q = {12'd0, cfg_attn_m_q_i}; end
            5'd9:  begin mul_a_q = eff_m_sub_i;                          mul_b_q = cfg_gemm_n_tiles_i; end
            5'd10: begin mul_a_q = macro_m_total_i;                      mul_b_q = is_attn_i ? q_blk_per_tile_q : total_k_blocks_q; end
            5'd11: begin mul_a_q = eff_m_sub_i;                          mul_b_q = is_attn_i ? q_blk_per_tile_q : total_k_blocks_q; end
            default: begin mul_a_q = 16'd0; mul_b_q = 16'd0; end
        endcase
        end else if ((cmp_state_q == CMP_O_INIT) && !ccalc_done_q) begin
            case (ccalc_idx_q)
                3'd0: begin mul_a_q = cur_m_sub[15:0];       mul_b_q = cur_o_cols[15:0]; end
                3'd1: begin mul_a_q = cur_m_x_cols_q[15:0];  mul_b_q = tile_out_bytes_w[15:0]; end
                3'd2: begin mul_a_q = cur_m_sub[15:0];       mul_b_q = attn_wo_vec_factor_q[15:0]; end
                3'd3: begin mul_a_q = cur_m_sub[15:0];       mul_b_q = attn_wb_per_m[15:0]; end
                3'd4: begin mul_a_q = cur_o_cols[15:0];      mul_b_q = total_k_blocks_q; end
                default: ;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pinit_idx_q   <= 5'd0;
            pinit_phase_q <= 1'b0;
            pinit_done_q  <= 1'b0;
            total_k_blocks_q     <= 16'd0;
            q_blk_per_tile_q     <= 16'd0;
            k_blk_per_head_q     <= 16'd0;
            v_blk_per_head_q     <= 16'd0;
            auto_v_base_q        <= 16'd0;
            packed_v_off_idx_q   <= 16'd0;
            attn_wo_line_factor_q<= 16'd0;
            attn_wo_vec_factor_q <= 32'd0;
            attn_wb_dv_m_q       <= 16'd0;
            o_wb_macro_inner_q   <= 16'd0;
            a_step_h_q           <= 32'd0;
            a_full_bytes_q       <= 32'd0;
            kt_stride_q          <= 32'd0;
            kt_stride_attn_q     <= 32'd0;
            sz_kv0_q             <= 32'd0;
        end else if (start_i) begin
            pinit_idx_q   <= 5'd0;
            pinit_phase_q <= 1'b0;
            pinit_done_q  <= 1'b0;
        end else if (!pinit_done_q) begin
            if (!pinit_phase_q) begin
                pinit_phase_q <= 1'b1;
                if (pinit_idx_q == 5'd0) begin
                    kt_stride_q      <= ({16'd0, cfg_gemm_k_blocks_i} << 7) + ({16'd0, cfg_gemm_k_blocks_i} << 4);
                    kt_stride_attn_q <= ({16'd0, cfg_attn_k_blocks_i} << 7) + ({16'd0, cfg_attn_k_blocks_i} << 4);
                    sz_kv0_q         <= (({16'd0, cfg_attn_k_blocks_i} + 16'd1) << 7) +
                                        (({16'd0, cfg_attn_k_blocks_i} + 16'd1) << 4);
                end
            end else begin
                case (pinit_idx_q)
                    5'd0:  total_k_blocks_q      <= mul_p[15:0];
                    5'd1:  q_blk_per_tile_q      <= mul_p[15:0];
                    5'd2:  k_blk_per_head_q      <= mul_p[15:0];
                    5'd3:  v_blk_per_head_q      <= mul_p[15:0];
                    5'd4:  auto_v_base_q         <= mul_p[15:0];
                    5'd5:  packed_v_off_idx_q    <= mul_p[15:0];
                    5'd6:  attn_wo_line_factor_q <= mul_p[15:0];
                    5'd7:  attn_wo_vec_factor_q  <= mul_p;
                    5'd8:  attn_wb_dv_m_q        <= mul_p[15:0];
                    5'd9:  o_wb_macro_inner_q    <= mul_p[15:0];
                    5'd10: a_step_h_q            <= pinit_a_bytes_w;
                    5'd11: a_full_bytes_q        <= pinit_a_bytes_w;
                    default: ;
                endcase
                if (pinit_idx_q == 5'(PINIT_N - 1)) begin
                    pinit_done_q  <= 1'b1;
                    pinit_phase_q <= 1'b0;
                end else begin
                    pinit_idx_q   <= pinit_idx_q + 5'd1;
                    pinit_phase_q <= 1'b0;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ccalc_idx_q  <= 3'd0;
            ccalc_phase_q<= 1'b0;
            ccalc_done_q <= 1'b0;
            cur_m_x_cols_q     <= 32'd0;
            o_wb_bytes_q       <= 32'd0;
            attn_wo_vectors_q  <= 32'd0;
            attn_wb_bytes_q    <= 32'd0;
            wt_chunk_inner_q   <= 32'd0;
        end else if (cmp_state_q != CMP_O_INIT) begin
            ccalc_idx_q  <= 3'd0;
            ccalc_phase_q<= 1'b0;
            ccalc_done_q <= 1'b0;
        end else if (!ccalc_done_q) begin
            if (!ccalc_phase_q) begin
                ccalc_phase_q <= 1'b1;
            end else begin
                case (ccalc_idx_q)
                    3'd0: cur_m_x_cols_q    <= mul_p;
                    3'd1: o_wb_bytes_q      <= mul_p;
                    3'd2: attn_wo_vectors_q <= mul_p;
                    3'd3: attn_wb_bytes_q   <= mul_p;
                    3'd4: wt_chunk_inner_q  <= mul_p;
                    default: ;
                endcase
                if (ccalc_idx_q == 3'd4) begin
                    ccalc_done_q  <= 1'b1;
                    ccalc_phase_q <= 1'b0;
                end else begin
                    ccalc_idx_q   <= ccalc_idx_q + 3'd1;
                    ccalc_phase_q <= 1'b0;
                end
            end
        end
    end

    wire [31:0] gemm_wo_vectors = cfg_gemm_decode_i ? {16'd0, cur_o_cols} : (cur_m_x_cols_q << 4);
    wire [31:0] wt_chunk_step = wt_chunk_inner_q << 9;

    wire [31:0] a_cmp_base     = cfg_tcdm_q_base_i + (macro_dma_en_i ? (a_cmp_rd_sel_q ? a_stride_eff_i : 32'd0) : 32'd0);
    wire [31:0] o_cmp_base_0   = (is_gemm_i && (cfg_out_mode_i == 2'd3)) ? cfg_v_out_base_i : cfg_tcdm_out_base_i;
    wire [31:0] o_cmp_base     = o_cmp_base_0 + (macro_dma_en_i ? (o_cmp_sel_q ? o_stride_eff_i : 32'd0) : 32'd0);

    wire [31:0] gemm_tile_out_bytes = cfg_gemm_decode_i
                                      ? ((cfg_out_mode_i == 2'd0) ? 32'd36  : 32'd9)
                                      : ((cfg_out_mode_i == 2'd0) ? 32'd576 : 32'd144);
    wire [31:0] attn_vec_bytes      = (cfg_out_mode_i == 2'd0) ? 32'd36 : 32'd9;
    wire        o_dma_is_16bank     = is_gemm_i &&
                                      ((cfg_out_mode_i == 2'd2) || (cfg_out_mode_i == 2'd3) ||
                                       ((cfg_out_mode_i == 2'd0) && cfg_out_bank_layout_i && !cfg_gemm_decode_i));

    wire [31:0] v_nb_eff = (cfg_v_num_batches_i == 32'd0) ? 32'd1 : cfg_v_num_batches_i;
    assign v_num_batches_o = is_vtrans_i ? v_nb_eff : cur_m_x_cols_q;

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
    wire [15:0] eff_qi_bound = macro_dma_en_i ? cur_m_sub : cfg_attn_q_tiles_i;
    wire        last_m_beat  = (cnt_m_q == (cfg_attn_m_q_i - 4'd1));
    wire        last_kj      = (cnt_kj_q == (cfg_attn_kv_tiles_i - 16'd1));
    wire        last_dvi     = (cnt_dvi_q == (cfg_attn_dv_tiles_i - 16'd1));
    wire        last_qi      = (cnt_qi_q == (eff_qi_bound - 16'd1));
    wire        last_h       = (cnt_h_q == (cfg_attn_h_i - 8'd1));
    wire        is_last_h_of_hkv = (cnt_hgrp_q == cfg_gqa_group_m1_i) || last_h;

    // Compact TCDM address generation (no multiplier trees)
    wire [31:0] ring_rd_tcdm_addr  = cfg_tcdm_ring_base_i + {12'd0, ring_rd_off_i};

    wire [31:0] preload_wt_addr    = cfg_tcdm_ring_base_i + {12'd0, wk_tcdm_blk_q, 9'd0};
    wire [31:0] core_wt_addr       = cfg_enable_dma_i
                                     ? ring_rd_tcdm_addr
                                     : (preload_wt_addr + ({16'd0, cnt_gk_q} << 9));

    wire [31:0] act_tile_tcdm_addr = a_cmp_base + {12'd0, act_q_blk_q, 9'd0};

    wire [31:0] attn_q_addr        = a_cmp_base + {12'd0, act_q_blk_q, 9'd0};
    wire [31:0] attn_k_pin_addr    = cfg_tcdm_ring_base_i + {12'd0, wk_tcdm_blk_q, 9'd0};
    wire [31:0] attn_v_pin_addr    = cfg_attn_sep_kv_i
                                     ? (cfg_tcdm_ring_base_i + {12'd0, v_tcdm_blk_q, 9'd0})
                                     : ((cnt_dvi_q == 16'd0)
                                        ? (attn_k_pin_addr + ({16'd0, cfg_attn_k_blocks_i} << 9))
                                        : (cfg_tcdm_ring_base_i + {12'd0, v_tcdm_blk_q, 9'd0}));
    wire [31:0] attn_k_preload_addr = cfg_tcdm_ring_base_i + {12'd0, wk_tcdm_blk_q, 9'd0};
    wire [31:0] attn_v_preload_addr = cfg_tcdm_ring_base_i + {12'd0, v_tcdm_blk_q, 9'd0};

    wire [31:0] attn_k_addr = !cfg_enable_dma_i ? attn_k_preload_addr :
                              attn_kv_pin_en_i    ? attn_k_pin_addr     :
                                                  ring_rd_tcdm_addr;
    wire [31:0] attn_v_addr = !cfg_enable_dma_i ? attn_v_preload_addr :
                              attn_kv_pin_en_i    ? attn_v_pin_addr     :
                                                  (ring_rd_tcdm_addr + ({16'd0, cfg_attn_k_blocks_i} << 9));

    // Attention KV availability qualifiers
    wire [31:0] v_pin_need_idx = mul_u32_u3({16'd0, cfg_attn_kv_tiles_i}, cnt_dvi_q[2:0]) + {16'd0, cnt_kj_q};
    wire        attn_k_ready   = !cfg_enable_dma_i ? 1'b1 :
                                 attn_kv_pin_en_i    ? ((cnt_hkv_q == pin_active_hkv_i) && (cnt_kj_q < k_pinned_cnt_i)) :
                                                     (w_ready_tiles_i > '0);
    wire        attn_v_ready   = !attn_kv_pin_en_i ? 1'b1 :
                                                   ((cnt_hkv_q == pin_active_hkv_i) && (v_pin_need_idx < v_pinned_cnt_i));

    // Compute beat fire signals
    wire qk_compute_fire   = act_valid_i && (&wt_valid_i);
    wire pv_compute_fire   = !p_fifo_empty_i && (&wt_valid_i);
    wire gemm_compute_fire = act_valid_i && (&wt_valid_i) && gemm_beat_ok_i;

    // Causal mask generator (20-bit incremental row counter + 16-bit comparator)
    logic [NUM_LANES-1:0] auto_causal_mask;
    logic [19:0] q_row_abs;
    logic [15:0] q_row_blk;
    logic [3:0]  q_row_rem;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            causal_q_idx_q <= 4'd0;
        else if (start_i || (cmp_state_q == CMP_ATTN_QK_LAUNCH))
            causal_q_idx_q <= 4'd0;
        else if (cmp_state_q == CMP_ATTN_SM_DRAIN)
            causal_q_idx_q <= cnt_dr_q;
    end

    always_comb begin
        causal_m  = (cmp_state_q == CMP_ATTN_QK_LAUNCH || cmp_state_q == CMP_ATTN_QK_RUN)
                    ? cnt_m_q : causal_q_idx_q;
        q_row_abs = q_tile_row_base_q + {16'd0, causal_m};
        q_row_blk = q_row_abs[19:4];
        q_row_rem = q_row_abs[3:0];
        if (!cfg_is_causal_i) begin
            auto_causal_mask = 16'hFFFF;
        end else if (cnt_kj_q < q_row_blk) begin
            auto_causal_mask = 16'hFFFF;
        end else if (cnt_kj_q > q_row_blk) begin
            auto_causal_mask = 16'h0000;
        end else begin
            auto_causal_mask = 16'hFFFF >> (4'd15 - q_row_rem);
        end
    end

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
                if (start_i)
                    cmp_state_d = CMP_PARAM_INIT;
            end

            CMP_PARAM_INIT: begin
                if (pinit_done_q) begin
                    if (is_dma_wb_i)       cmp_state_d = CMP_DMA_WB_ISSUE;
                    else if (is_vtrans_i)  cmp_state_d = CMP_V_LAUNCH;
                    else if (is_rdback_i)  cmp_state_d = CMP_RD_LAUNCH;
                    else                 cmp_state_d = CMP_O_INIT;
                end
            end

            CMP_O_INIT: begin
                if (is_gemm_i) begin
                    if (ccalc_done_q && (!macro_dma_en_i || (!wo_busy_i && !v_busy_i && !dr_busy_i &&
                                          (inflight_out_cnt_i == 4'd0) && !o_buf_wb_busy_i[o_cmp_sel_q]))) begin
                        cmp_state_d = CMP_GEMM_LAUNCH;
                    end
                end else begin
                    // FlashAttention
                    if (macro_dma_en_i) begin
                        if (ccalc_done_q && !wo_busy_i && !v_busy_i && !dr_busy_i &&
                            (inflight_out_cnt_i == 4'd0) && !o_buf_wb_busy_i[o_cmp_sel_q]) begin
                            cmp_state_d = CMP_ATTN_QK_LAUNCH;
                        end
                    end else if (ccalc_done_q && !wo_busy_i) begin
                        cmp_state_d = CMP_ATTN_QK_LAUNCH;
                    end
                end
            end

            // ---------------- 4-Loop OS GEMM ----------------
            CMP_GEMM_LAUNCH: begin
                if (w_buf_ready_i && a_buf_ready_i && !act_busy_i && !wt_busy_i)
                    cmp_state_d = CMP_GEMM_RUN;
            end

            CMP_GEMM_RUN: begin
                if (gemm_compute_fire && last_gbeat && last_gk) begin
                    if (last_kt && last_gm && last_o_col)
                        cmp_state_d = macro_dma_en_i ? CMP_O_COMMIT : CMP_WAIT_DONE;
                    else
                        cmp_state_d = CMP_GEMM_LAUNCH;
                end
            end

            // ---------------- FlashAttention ----------------
            CMP_ATTN_QK_LAUNCH: begin
                if (attn_k_ready && a_buf_ready_i && !act_busy_i && !wt_busy_i)
                    cmp_state_d = CMP_ATTN_QK_RUN;
            end

            CMP_ATTN_QK_RUN: begin
                if (!act_busy_i && !wt_busy_i && !dr_busy_i && (inflight_out_cnt_i == 4'd0))
                    cmp_state_d = CMP_ATTN_SM_DRAIN;
            end

            CMP_ATTN_SM_DRAIN: begin
                if (cnt_dr_q == (cfg_attn_m_q_i - 4'd1))
                    cmp_state_d = CMP_ATTN_PV_LAUNCH;
            end

            CMP_ATTN_PV_LAUNCH: begin
                if (!dr_busy_i && !wt_busy_i && attn_v_ready)
                    cmp_state_d = CMP_ATTN_PV_RUN;
            end

            CMP_ATTN_PV_RUN: begin
                if (pv_compute_fire && last_m_beat)
                    cmp_state_d = CMP_ATTN_PV_DRAIN;
            end

            CMP_ATTN_PV_DRAIN: begin
                if (!wt_busy_i) begin
                    if (cnt_kj_q == 16'd0) begin
                        if (macro_dma_en_i && attn_chunk_done_q)
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
                if (!wo_busy_i && !dr_busy_i && (inflight_out_cnt_i == 4'd0) &&
                    !v_busy_i && !act_busy_i && !wt_busy_i) begin
                    if (is_gemm_i)
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
                if ((dma_state_i == DMA_IDLE) && dma_req_ready_i)
                    cmp_state_d = CMP_DMA_WB_WAIT;
            end

            CMP_DMA_WB_WAIT: begin
                if (dma_done_i)
                    cmp_state_d = CMP_DONE;
            end

            // ---------------- Drain & Completion Barrier ----------------
            CMP_WAIT_DONE: begin
                if (!dr_busy_i && (inflight_out_cnt_i == 4'd0) && !wo_busy_i && !v_busy_i && !rd_busy_i &&
                    (!macro_dma_en_i || (!o_buf_wb_busy_i[0] && !o_buf_wb_busy_i[1])) &&
                    (!cfg_enable_dma_i || !dma_busy_i)) begin
                    cmp_state_d = CMP_DONE;
                end
            end

            CMP_DONE: begin
                if (start_i)
                    cmp_state_d = CMP_PARAM_INIT;
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
        act_k_blocks_o         = is_attn_i ? 32'(cfg_attn_k_blocks_i) : 32'(cfg_gemm_k_blocks_i);
        act_k_stride_o         = 32'h200;
        act_tokens_o           = is_attn_i ? 32'(cfg_attn_m_q_i) : (cfg_gemm_decode_i ? 32'd1 : 32'd16);
        act_token_stride_o     = is_attn_i ? ({16'd0, cfg_attn_k_blocks_i} << 9) :
                                 (cfg_act_bank_layout_i ? 32'h10 : ({16'd0, total_k_blocks_q} << 9));
        act_ready_o            = 1'b0;

        // Weight / KV streamer defaults
        wt_start_o             = 1'b0;
        wt_base_o              = core_wt_addr;
        wt_repeat_o            = is_attn_i ? 32'(cfg_attn_m_q_i) : (cfg_gemm_decode_i ? 32'd1 : 32'd16);
        wt_k_blocks_o          = is_attn_i ? 32'(cfg_attn_k_blocks_i) : 32'(cfg_gemm_k_blocks_i);
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
        cfg_mode_o             = is_attn_i ? 2'b10 : 2'b00;
        cfg_enable_slq_o       = 1'b1;
        lane_valid_o           = 16'h0000;
        lane_acc_idx_o         = 4'd0;

        case (cmp_state_q)
            CMP_O_INIT: begin
                if (is_gemm_i) begin
                    if (ccalc_done_q && (!macro_dma_en_i || (!wo_busy_i && !v_busy_i && !dr_busy_i &&
                                          (inflight_out_cnt_i == 4'd0) && !o_buf_wb_busy_i[o_cmp_sel_q]))) begin
                        wo_base_o = o_cmp_base;
                        if (cfg_out_mode_i == 2'd3) begin
                            v_start_o         = 1'b1;
                            wo_mode_o         = 2'd2;
                            wo_num_vectors_o  = 32'd16;
                            wo_line_stride_o  = 32'd16;
                            wo_num_batches_o  = cur_m_x_cols_q;
                            wo_batch_stride_o = 32'h200;
                        end else begin
                            wo_start_o = 1'b1;
                            case (cfg_out_mode_i)
                                2'd0: begin
                                    wo_mode_o = 2'd1;
                                    if (cfg_out_bank_layout_i && !cfg_gemm_decode_i) begin
                                        wo_num_vectors_o  = 32'd16;
                                        wo_line_stride_o  = 32'd16;
                                        wo_num_batches_o  = cur_m_x_cols_q;
                                        wo_batch_stride_o = 32'h800;
                                    end else begin
                                        wo_num_vectors_o  = cfg_gemm_decode_i ? {16'd0, cur_o_cols} :
                                                            gemm_wo_vectors;
                                    end
                                end
                                2'd1: begin
                                    wo_mode_o        = 2'd0;
                                    wo_num_vectors_o = cfg_gemm_decode_i ? {16'd0, cur_o_cols} :
                                                       gemm_wo_vectors;
                                end
                                2'd2: begin
                                    wo_mode_o         = 2'd2;
                                    wo_num_vectors_o  = cfg_gemm_decode_i ? 32'd1 : 32'd16;
                                    wo_line_stride_o  = 32'd16;
                                    wo_num_batches_o  = cur_m_x_cols_q;
                                    wo_batch_stride_o = 32'h200;
                                end
                                default: begin
                                    wo_mode_o        = 2'd1;
                                    wo_num_vectors_o = cfg_gemm_decode_i ? {16'd0, cur_o_cols} :
                                                       gemm_wo_vectors;
                                end
                            endcase
                        end
                    end
                end else if (is_attn_i) begin
                    if (ccalc_done_q && (macro_dma_en_i ? (!wo_busy_i && !v_busy_i && !dr_busy_i &&
                                        (inflight_out_cnt_i == 4'd0) && !o_buf_wb_busy_i[o_cmp_sel_q])
                                     : !wo_busy_i)) begin
                        wo_start_o = 1'b1;
                        wo_base_o  = o_cmp_base;
                        if (cfg_out_mode_i == 2'd2) begin
                            wo_mode_o         = 2'd3;
                            wo_num_vectors_o  = {16'd0, cfg_attn_m_q_i};
                            wo_line_stride_o  = attn_wo_line_stride;
                            wo_num_batches_o  = {16'd0, cfg_attn_dv_tiles_i};
                            wo_batch_stride_o = 32'h200;
                            wo_dim2_cnt_o     = {16'd0, cur_m_sub};
                            wo_dim2_stride_o  = attn_wo_dim2_stride;
                            wo_dim3_cnt_o     = macro_dma_en_i ? 32'd1 : {16'd0, cfg_attn_h_i};
                            wo_dim3_stride_o  = attn_wo_dim3_stride;
                        end else begin
                            wo_mode_o         = (cfg_out_mode_i == 2'd1) ? 2'd0 : 2'd1;
                            wo_num_vectors_o  = attn_wo_vectors_q;
                            wo_line_stride_o  = 32'h200;
                            wo_num_batches_o  = 32'd1;
                            wo_batch_stride_o = 32'd0;
                        end
                    end
                end
            end

            // ---------------- GEMM ----------------
            CMP_GEMM_LAUNCH: begin
                if (w_buf_ready_i && a_buf_ready_i && !act_busy_i && !wt_busy_i) begin
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
                                                 (w_ring_reuse_active_i ? last_gm : 1'b1);
                        if (last_kt)
                            drain_start_pulse = 1'b1;
                    end
                end
            end

            // ---------------- FlashAttention ----------------
            CMP_ATTN_QK_LAUNCH: begin
                cfg_mode_o = 2'b01;
                if (attn_k_ready && a_buf_ready_i && !act_busy_i && !wt_busy_i) begin
                    wt_start_o         = 1'b1;
                    wt_base_o          = attn_k_addr;
                    wt_repeat_o        = 32'(cfg_attn_m_q_i);
                    wt_k_blocks_o      = 32'(cfg_attn_k_blocks_i);
                    wt_k_stride_o      = 32'h200;

                    act_start_o        = 1'b1;
                    act_base_o         = attn_q_addr;
                    act_tokens_o       = 32'(cfg_attn_m_q_i);
                    act_token_stride_o = {16'd0, cfg_attn_k_blocks_i} << 9;
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
                if (!dr_busy_i && !wt_busy_i && attn_v_ready) begin
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
                    w_release_credit_pulse = cfg_enable_dma_i && !attn_kv_pin_en_i;
                    kv_pin_rel_hkv_pulse   = attn_kv_pin_en_i && kv_pin_hkv_end_q;
                    if (cnt_kj_q == 16'd0)
                        drain_start_pulse = 1'b1;
                end
            end

            // ---------------- Macro-DMA O Commit ----------------
            CMP_O_COMMIT: begin
                if (!wo_busy_i && !dr_busy_i && (inflight_out_cnt_i == 4'd0) &&
                    !v_busy_i && !act_busy_i && !wt_busy_i) begin
                    o_flush_commit_pulse = 1'b1;
                    macro_a_rel_pulse    = is_attn_i ? 1'b1 : last_o_chunk;
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
                                                                       : (v_nb_eff << 11));
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
                                                                       : (v_nb_eff << 11));
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
                        kv_pin_hkv_end_q <= attn_kv_pin_en_i && last_dvi && last_qi && (!macro_dma_en_i || last_macro_m) && is_last_h_of_hkv;
                        if (macro_dma_en_i) begin
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

            if (cmp_state_q == CMP_ATTN_QK_RUN && !act_busy_i && !wt_busy_i && !dr_busy_i && (inflight_out_cnt_i == 4'd0))
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
                if (is_gemm_i) begin
                    if (!last_o_chunk) begin
                        o_col_start_q <= o_col_start_q + cur_o_cols;
                    end else begin
                        o_col_start_q  <= 16'd0;
                        a_cmp_rd_sel_q <= ~a_cmp_rd_sel_q;
                        cmp_m_base_q   <= last_macro_m ? 16'd0 : (cmp_m_base_q + eff_m_sub_i);
                    end
                end else if (is_attn_i) begin
                    attn_chunk_done_q <= 1'b0;
                    a_cmp_rd_sel_q    <= ~a_cmp_rd_sel_q;
                    if (!last_macro_m) begin
                        cmp_m_base_q <= cmp_m_base_q + eff_m_sub_i;
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


    wire gemm_tile_end = (cmp_state_q == CMP_GEMM_RUN) && gemm_compute_fire && last_gbeat && last_gk;
    wire attn_pv_end   = (cmp_state_q == CMP_ATTN_PV_RUN) && pv_compute_fire && last_m_beat;
    wire attn_last_kj  = attn_pv_end && last_kj;
    wire attn_kj_inc   = attn_pv_end && !last_kj;
    wire attn_dvi_inc  = attn_last_kj && !last_dvi;
    wire attn_qi_inc   = attn_last_kj && last_dvi && !last_qi;
    wire attn_head_end = attn_last_kj && last_dvi && last_qi;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_q_blk_q       <= 11'd0;
            wk_tcdm_blk_q     <= 11'd0;
            wk_base_blk_q     <= 11'd0;
            v_tcdm_blk_q      <= 11'd0;
            v_base_blk_q      <= 11'd0;
            q_tile_row_base_q <= 20'd0;
        end else if (start_i || ((cmp_state_q == CMP_PARAM_INIT) && pinit_done_q)) begin
            act_q_blk_q       <= 11'd0;
            wk_tcdm_blk_q     <= 11'd0;
            wk_base_blk_q     <= 11'd0;
            v_tcdm_blk_q      <= (is_attn_i && !cfg_enable_dma_i) ? v_base_blk
                                 : (cfg_attn_sep_kv_i ? k_blk_head_blk : packed_v_off_idx_q[10:0]);
            v_base_blk_q      <= v_base_blk;
            q_tile_row_base_q <= {2'd0, cfg_attn_kv_past_tiles_i, 4'd0};
        end else begin
            if (gemm_tile_end) begin
                if (last_kt) begin
                    if (last_gm) begin
                        if (!last_o_col) begin
                            wk_base_blk_q <= wk_base_blk_q + wt_gn_step_blk;
                            wk_tcdm_blk_q <= wk_base_blk_q + wt_gn_step_blk;
                            act_q_blk_q   <= 11'd0;
                        end
                    end else begin
                        wk_tcdm_blk_q <= wk_base_blk_q;
                        act_q_blk_q   <= act_q_blk_q + act_gm_step_blk;
                    end
                end else begin
                    wk_tcdm_blk_q <= wk_tcdm_blk_q + k_kt_step_blk;
                    act_q_blk_q   <= act_q_blk_q + k_kt_step_blk;
                end
            end

            if (o_flush_commit_pulse) begin
                if (is_gemm_i) begin
                    if (!last_o_chunk) begin
                        wk_base_blk_q <= wk_base_blk_q + wt_chunk_inner_q[10:0];
                        wk_tcdm_blk_q <= wk_base_blk_q + wt_chunk_inner_q[10:0];
                        act_q_blk_q   <= 11'd0;
                    end else begin
                        wk_base_blk_q <= 11'd0;
                        wk_tcdm_blk_q <= 11'd0;
                        act_q_blk_q   <= 11'd0;
                    end
                end else begin
                    act_q_blk_q       <= 11'd0;
                    wk_tcdm_blk_q     <= !cfg_enable_dma_i ? wk_base_blk_q : 11'd0;
                    v_tcdm_blk_q      <= !cfg_enable_dma_i ? v_base_blk_q
                                         : (cfg_attn_sep_kv_i ? k_blk_head_blk : packed_v_off_idx_q[10:0]);
                    q_tile_row_base_q <= last_macro_m
                                         ? {2'd0, cfg_attn_kv_past_tiles_i, 4'd0}
                                         : (q_tile_row_base_q + {16'd0, cfg_attn_m_q_i});
                end
            end

            if (attn_kj_inc) begin
                if (!cfg_enable_dma_i)
                    wk_tcdm_blk_q <= wk_tcdm_blk_q + k_preload_step_blk;
                else if (attn_kv_pin_en_i)
                    wk_tcdm_blk_q <= wk_tcdm_blk_q + (cfg_attn_sep_kv_i ? k_pin_step_blk : packed_k_step_blk);

                if (!cfg_enable_dma_i || cfg_attn_sep_kv_i || (cnt_dvi_q != 16'd0))
                    v_tcdm_blk_q <= v_tcdm_blk_q + 11'd1;
            end

            if (attn_dvi_inc) begin
                wk_tcdm_blk_q <= !cfg_enable_dma_i ? wk_base_blk_q : 11'd0;
                if (!cfg_enable_dma_i || cfg_attn_sep_kv_i || (cnt_dvi_q != 16'd0))
                    v_tcdm_blk_q <= v_tcdm_blk_q + 11'd1;
            end

            if (attn_qi_inc) begin
                wk_tcdm_blk_q     <= !cfg_enable_dma_i ? wk_base_blk_q : 11'd0;
                v_tcdm_blk_q      <= !cfg_enable_dma_i ? v_base_blk_q
                                     : (cfg_attn_sep_kv_i ? k_blk_head_blk : packed_v_off_idx_q[10:0]);
                q_tile_row_base_q <= q_tile_row_base_q + {16'd0, cfg_attn_m_q_i};
            end

            if (attn_qi_inc || (attn_head_end && !macro_dma_en_i))
                act_q_blk_q <= act_q_blk_q + q_step_blk;

            if (attn_head_end) begin
                if (!macro_dma_en_i)
                    q_tile_row_base_q <= {2'd0, cfg_attn_kv_past_tiles_i, 4'd0};
                if (!cfg_enable_dma_i && !macro_dma_en_i) begin
                    if (last_h) begin
                        wk_base_blk_q <= 11'd0;
                        v_base_blk_q  <= v_base_blk;
                        wk_tcdm_blk_q <= 11'd0;
                        v_tcdm_blk_q  <= v_base_blk;
                    end else if (cnt_hgrp_q == cfg_gqa_group_m1_i) begin
                        wk_base_blk_q <= wk_base_blk_q + k_blk_head_blk;
                        v_base_blk_q  <= v_base_blk_q + v_blk_head_blk;
                        wk_tcdm_blk_q <= wk_base_blk_q + k_blk_head_blk;
                        v_tcdm_blk_q  <= v_base_blk_q + v_blk_head_blk;
                    end else begin
                        wk_tcdm_blk_q <= wk_base_blk_q;
                        v_tcdm_blk_q  <= v_base_blk_q;
                    end
                end else begin
                    wk_tcdm_blk_q <= 11'd0;
                    v_tcdm_blk_q  <= cfg_attn_sep_kv_i ? k_blk_head_blk : packed_v_off_idx_q[10:0];
                end
            end
        end
    end

    assign cmp_state_o              = cmp_state_q;
    assign pinit_sync_o             = (cmp_state_q == CMP_PARAM_INIT) && pinit_done_q;
    assign a_cmp_rd_sel_o           = a_cmp_rd_sel_q;
    assign o_cmp_sel_o              = o_cmp_sel_q;
    assign last_o_chunk_o           = last_o_chunk;
    assign last_macro_m_o           = last_macro_m;
    assign drain_start_pulse_o      = drain_start_pulse;
    assign sm_drain_req_o           = sm_drain_req;
    assign sm_drain_idx_o           = sm_drain_idx;
    assign sm_row_start_o           = sm_row_start;
    assign auto_causal_mask_o       = auto_causal_mask;
    assign w_release_credit_pulse_o = w_release_credit_pulse;
    assign macro_a_rel_pulse_o      = macro_a_rel_pulse;
    assign o_flush_commit_pulse_o   = o_flush_commit_pulse;
    assign kv_pin_rel_hkv_pulse_o   = kv_pin_rel_hkv_pulse;
    assign attn_k_ready_o           = attn_k_ready;
    assign attn_v_ready_o           = attn_v_ready;
    assign o_wb_bytes_o             = o_wb_bytes_q;
    assign attn_wb_bytes_o          = attn_wb_bytes_q;
    assign o_wb_macro_step_o        = o_wb_macro_step;
    assign a_step_h_o               = a_step_h_q;
    assign a_full_bytes_o           = a_full_bytes_q;
    assign kt_stride_o              = kt_stride_q;
    assign kt_stride_attn_o         = kt_stride_attn_q;
    assign sz_kv0_o                 = sz_kv0_q;
    assign k_blk_head_blk_o         = k_blk_head_blk;
    assign k_preload_step_blk_o     = k_preload_step_blk;
    assign packed_k_step_blk_o      = packed_k_step_blk;
    assign packed_v_off_idx_o       = packed_v_off_idx_q;
    assign cnt_gn_o                 = cnt_gn_q;
    assign cnt_gm_o                 = cnt_gm_q;
    assign cnt_gk_o                 = cnt_gk_q;
    assign cnt_kt_o                 = cnt_kt_q;
    assign cnt_gbeat_o              = cnt_gbeat_q;
    assign cnt_h_o                  = cnt_h_q;
    assign cnt_qi_o                 = cnt_qi_q;
    assign cnt_dvi_o                = cnt_dvi_q;
    assign cnt_kj_o                 = cnt_kj_q;
    assign cnt_m_o                  = cnt_m_q;
    assign cnt_dr_o                 = cnt_dr_q;
    assign qk_compute_fire_o        = qk_compute_fire;
    assign pv_compute_fire_o        = pv_compute_fire;

endmodule
