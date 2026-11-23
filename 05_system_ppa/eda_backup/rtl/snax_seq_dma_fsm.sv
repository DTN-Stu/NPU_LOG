// =============================================================================
// Module: snax_seq_dma_fsm.sv
// Description: Sub-FSM 3 of snax_llm_sequencer -- DMA Arbiter, Ring Credit,
//              Ping-Pong A/O, KV-Pin Synchronization & DRAM Address Pointers.
// =============================================================================

`timescale 1ns/1ps

module snax_seq_dma_fsm #(
    parameter int unsigned ADDR_WIDTH     = 32,
    parameter int unsigned NUM_RING_SLOTS = 4,
    parameter int unsigned SLOT_SIZE_BYTES= 128 * 1024,
    parameter int unsigned PTR_W          = (NUM_RING_SLOTS > 1) ? $clog2(NUM_RING_SLOTS) : 1,
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
    input  logic                        cfg_act_bank_layout_i,
    input  logic                        cfg_out_bank_layout_i,
    input  logic [15:0]                 cfg_gemm_n_tiles_i,
    input  logic [15:0]                 cfg_gemm_m_tiles_i,
    input  logic [7:0]                  cfg_gqa_group_m1_i,
    input  logic [7:0]                  cfg_attn_h_i,
    input  logic [15:0]                 cfg_attn_q_tiles_i,
    input  logic [15:0]                 cfg_attn_dv_tiles_i,
    input  logic [15:0]                 cfg_attn_kv_tiles_i,
    input  logic [15:0]                 cfg_attn_k_blocks_i,
    input  logic                        cfg_attn_sep_kv_i,

    input  logic [ADDR_WIDTH-1:0]       cfg_tcdm_ring_base_i,
    input  logic [ADDR_WIDTH-1:0]       cfg_tcdm_q_base_i,
    input  logic [ADDR_WIDTH-1:0]       cfg_tcdm_out_base_i,
    input  logic [ADDR_WIDTH-1:0]       cfg_dram_kv_base_i,
    input  logic [31:0]                 cfg_stride_tile_i,
    input  logic [ADDR_WIDTH-1:0]       cfg_dram_act_base_i,
    input  logic [ADDR_WIDTH-1:0]       cfg_dram_out_base_i,
    input  logic [ADDR_WIDTH-1:0]       cfg_dram_v_base_i,

    input  logic [15:0]                 macro_m_total_i,
    input  logic [15:0]                 eff_m_sub_i,
    input  logic [31:0]                 a_stride_eff_i,
    input  logic [31:0]                 o_stride_eff_i,
    input  logic [31:0]                 o_cmp_base_0_i,
    input  logic                        o_dma_is_16bank_i,

    // From Compute FSM
    input  logic [4:0]                  cmp_state_i,
    input  logic                        pinit_sync_i,
    input  logic                        a_cmp_rd_sel_i,
    input  logic                        o_cmp_sel_i,
    input  logic                        last_o_chunk_i,
    input  logic                        last_macro_m_i,
    input  logic                        w_release_credit_pulse_i,
    input  logic                        macro_a_rel_pulse_i,
    input  logic                        o_flush_commit_pulse_i,
    input  logic                        kv_pin_rel_hkv_pulse_i,
    input  logic                        attn_k_ready_i,
    input  logic                        attn_v_ready_i,

    input  logic [31:0]                 o_wb_bytes_i,
    input  logic [31:0]                 attn_wb_bytes_i,
    input  logic [31:0]                 o_wb_macro_step_i,
    input  logic [31:0]                 a_step_h_i,
    input  logic [31:0]                 a_full_bytes_i,
    input  logic [31:0]                 kt_stride_i,
    input  logic [31:0]                 kt_stride_attn_i,
    input  logic [31:0]                 sz_kv0_i,
    input  logic [10:0]                 k_blk_head_blk_i,
    input  logic [10:0]                 k_preload_step_blk_i,
    input  logic [10:0]                 packed_k_step_blk_i,
    input  logic [15:0]                 packed_v_off_idx_i,

    // To Compute / Top
    output logic                        dma_state_o,
    output logic                        dma_busy_o,
    output logic                        w_buf_ready_o,
    output logic                        a_buf_ready_o,
    output logic [1:0]                  o_buf_wb_busy_o,
    output logic [CNT_W-1:0]            w_ready_tiles_o,
    output logic [19:0]                 ring_rd_off_o,
    output logic [7:0]                  pin_active_hkv_o,
    output logic [15:0]                 k_pinned_cnt_o,
    output logic [31:0]                 v_pinned_cnt_o,

    // DMA Master Interface
    output logic                        dma_req_valid_o,
    input  logic                        dma_req_ready_i,
    output logic [1:0]                  dma_req_dir_o,
    output logic [ADDR_WIDTH-1:0]       dma_req_dram_addr_o,
    output logic [ADDR_WIDTH-1:0]       dma_req_tcdm_addr_o,
    output logic [31:0]                 dma_req_size_bytes_o,
    input  logic                        dma_done_i
);

    localparam logic [4:0] CMP_IDLE         = 5'd0;
    localparam logic [4:0] CMP_DMA_WB_ISSUE = 5'd15;
    localparam logic [4:0] CMP_DONE         = 5'd18;
    localparam logic [4:0] CMP_PARAM_INIT   = 5'd19;

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

    function automatic [31:0] mul_u32_u5(input [31:0] a, input [4:0] b);
        logic [31:0] s;
        integer k;
        s = 32'd0;
        for (k = 0; k < 5; k = k + 1)
            if (b[k]) s = s + (a << k);
        return s;
    endfunction

    dma_state_t       dma_state_q, dma_state_d;
    logic [CNT_W-1:0] w_ready_tiles_q;
    logic [PTR_W-1:0] ring_rd_ptr_q;
    logic [1:0]       a_buf_ready_q, o_buf_wb_busy_q;
    logic [7:0]       pin_active_hkv_q;
    logic [15:0]      k_pinned_cnt_q;
    logic [31:0]      v_pinned_cnt_q;
    logic             kv_pin_hkv_done_q;
    logic             dma_window_active;
    logic             dma_busy_q;
    logic             w_buf_ready, a_buf_ready;

    logic [19:0] ring_wr_off_q, ring_rd_off_q;
    logic [10:0] dma_k_blk_q, dma_v_blk_q;
    logic [31:0] dma_a_dram_ptr_q, a_head_base_q;
    logic [31:0] dma_wk_dram_ptr_q, dma_wg_v_dram_ptr_q;
    logic [31:0] o_wb_ptr_q, o_wb_macro_base_q;
    logic [31:0] a_rem_bytes_q;

    assign dma_state_o      = dma_state_q;
    assign dma_busy_o       = dma_busy_q;
    assign w_buf_ready_o    = w_buf_ready;
    assign a_buf_ready_o    = a_buf_ready;
    assign o_buf_wb_busy_o  = o_buf_wb_busy_q;
    assign w_ready_tiles_o  = w_ready_tiles_q;
    assign ring_rd_off_o    = ring_rd_off_q;
    assign pin_active_hkv_o = pin_active_hkv_q;
    assign k_pinned_cnt_o   = k_pinned_cnt_q;
    assign v_pinned_cnt_o   = v_pinned_cnt_q;

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

    assign dma_window_active = (cmp_state_i != CMP_IDLE) && (cmp_state_i != CMP_DONE) &&
                               (cmp_state_i != CMP_PARAM_INIT) && !is_dma_wb_i && !is_vtrans_i && !is_rdback_i;
    assign dma_busy_q        = (dma_state_q == DMA_INFLIGHT);
    assign w_buf_ready       = !cfg_enable_dma_i || (w_ready_tiles_q > '0);
    assign a_buf_ready       = !macro_dma_en_i || a_buf_ready_q[a_cmp_rd_sel_i];

    wire [15:0] dma_w_m_rem     = macro_m_total_i - dma_w_m_base_q;
    wire [15:0] dma_w_cur_m_sub = (dma_w_m_rem > eff_m_sub_i) ? eff_m_sub_i : dma_w_m_rem;
    wire [15:0] dma_a_m_rem     = macro_m_total_i - dma_a_m_base_q;
    wire [15:0] dma_a_cur_m_sub = (dma_a_m_rem > eff_m_sub_i) ? eff_m_sub_i : dma_a_m_rem;

    wire [ADDR_WIDTH-1:0] dma_w_tcdm_addr = cfg_tcdm_ring_base_i + {12'd0, ring_wr_off_q};
    wire                  dma_arb_en      = (dma_state_q == DMA_IDLE) && dma_window_active;

    // Arbitration requests
    wire dma_wb_req     = (cmp_state_i == CMP_DMA_WB_ISSUE) && (dma_state_q == DMA_IDLE);
    wire dma_o_wb_req   = dma_arb_en && macro_dma_en_i && o_buf_wb_busy_q[o_dma_sel_q];
    wire dma_a_can_req  = dma_arb_en && macro_dma_en_i && dma_a_pending_q && !a_buf_ready_q[a_dma_wr_sel_q];
    wire dma_a_urg_req  = dma_a_can_req && !a_buf_ready_q[a_cmp_rd_sel_i];

    // Weight / KV arbitration qualifier
    wire dma_kv_pin_can_req = dma_arb_en && attn_kv_pin_en_i && dma_w_pending_q &&
                              !kv_pin_hkv_done_q && (dma_hkv_q == pin_active_hkv_q);
    wire dma_ring_space_ok  = is_attn_i && cfg_attn_sep_kv_i && dma_kv_phase_q
                              ? 1'b1
                              : (w_free_credits_q > '0);
    wire dma_w_ring_can_req = dma_arb_en && cfg_enable_dma_i && !attn_kv_pin_en_i && dma_w_pending_q && dma_ring_space_ok;
    wire dma_w_can_req      = dma_kv_pin_can_req || dma_w_ring_can_req;
    wire dma_w_urg_req      = dma_w_can_req && (is_attn_i ? (!attn_k_ready_i || !attn_v_ready_i) : (w_ready_tiles_q == '0));

    logic     dma_issue_fire;
    logic     dma_ring_slot_reserve_fire;
    logic     dma_ring_slot_commit_fire;
    dma_tag_t dma_issue_tag;
    logic     dma_issue_buf;
    logic     dma_issue_kv_phase;
    logic     dma_issue_kv_dvi0;

    wire dma_ring_slot_done = dma_done_i && !is_dma_wb_i && !attn_kv_pin_en_i &&
                              (dma_inflight_tag_q == DMA_TAG_W) &&
                              (!is_attn_i || !cfg_attn_sep_kv_i || dma_inflight_kv_phase_q);

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
            dma_req_dir_o        = o_dma_is_16bank_i ? 2'd1 : 2'd3;
            dma_req_dram_addr_o  = o_buf_wb_dram_q[o_dma_sel_q];
            dma_req_tcdm_addr_o  = o_cmp_base_0_i + (o_dma_sel_q ? o_stride_eff_i : 32'd0);
            dma_req_size_bytes_o = o_buf_wb_bytes_q[o_dma_sel_q];
            dma_issue_tag        = DMA_TAG_O;
            dma_issue_buf        = o_dma_sel_q;
            dma_issue_fire       = dma_req_ready_i;
        end else if (dma_a_urg_req || (dma_a_can_req && !dma_w_urg_req)) begin
            dma_req_valid_o      = 1'b1;
            dma_req_dir_o        = cfg_act_bank_layout_i ? 2'd0 : 2'd2;
            dma_req_tcdm_addr_o  = cfg_tcdm_q_base_i + (a_dma_wr_sel_q ? a_stride_eff_i : 32'd0);
            dma_req_dram_addr_o  = dma_a_dram_ptr_q;
            dma_req_size_bytes_o = ((dma_a_m_base_q + eff_m_sub_i) >= macro_m_total_i)
                                   ? a_rem_bytes_q : a_full_bytes_i;
            dma_issue_tag  = DMA_TAG_A;
            dma_issue_buf  = a_dma_wr_sel_q;
            dma_issue_fire = dma_req_ready_i;
        end else if (dma_w_can_req) begin
            dma_req_valid_o = 1'b1;
            dma_req_dir_o   = 2'd0;
            dma_issue_tag   = DMA_TAG_W;
            dma_issue_fire  = dma_req_ready_i;

            if (is_gemm_i) begin
                dma_req_dram_addr_o        = dma_wk_dram_ptr_q;
                dma_req_tcdm_addr_o        = dma_w_tcdm_addr;
                dma_req_size_bytes_o       = kt_stride_i;
                dma_ring_slot_reserve_fire = dma_req_ready_i;
                dma_ring_slot_commit_fire  = dma_req_ready_i;
            end else if (attn_kv_pin_en_i) begin
                dma_issue_kv_phase = dma_kv_phase_q;
                dma_issue_kv_dvi0  = (dma_dvi_q == 16'd0);
                if (cfg_attn_sep_kv_i) begin
                    if (!dma_kv_phase_q) begin
                        dma_req_dram_addr_o  = dma_wk_dram_ptr_q;
                        dma_req_tcdm_addr_o  = cfg_tcdm_ring_base_i + {12'd0, dma_k_blk_q, 9'd0};
                        dma_req_size_bytes_o = kt_stride_attn_i;
                    end else begin
                        dma_req_dram_addr_o  = dma_wg_v_dram_ptr_q;
                        dma_req_tcdm_addr_o  = cfg_tcdm_ring_base_i + {12'd0, dma_v_blk_q, 9'd0};
                        dma_req_size_bytes_o = 32'd144;
                    end
                end else begin
                    if (dma_dvi_q == 16'd0) begin
                        dma_req_dram_addr_o  = dma_wk_dram_ptr_q;
                        dma_req_tcdm_addr_o  = cfg_tcdm_ring_base_i + {12'd0, dma_k_blk_q, 9'd0};
                        dma_req_size_bytes_o = sz_kv0_i;
                    end else begin
                        dma_req_dram_addr_o  = dma_wk_dram_ptr_q + kt_stride_attn_i;
                        dma_req_tcdm_addr_o  = cfg_tcdm_ring_base_i + {12'd0, dma_v_blk_q, 9'd0};
                        dma_req_size_bytes_o = 32'd144;
                    end
                end
            end else begin
                dma_issue_kv_phase = dma_kv_phase_q;
                if (cfg_attn_sep_kv_i) begin
                    if (!dma_kv_phase_q) begin
                        dma_req_dram_addr_o        = dma_wk_dram_ptr_q;
                        dma_req_tcdm_addr_o        = dma_w_tcdm_addr;
                        dma_req_size_bytes_o       = kt_stride_attn_i;
                        dma_ring_slot_reserve_fire = dma_req_ready_i;
                    end else begin
                        dma_req_dram_addr_o        = dma_wg_v_dram_ptr_q;
                        dma_req_tcdm_addr_o        = dma_w_tcdm_addr + ({16'd0, cfg_attn_k_blocks_i} << 9);
                        dma_req_size_bytes_o       = 32'd144;
                        dma_ring_slot_commit_fire  = dma_req_ready_i;
                    end
                end else begin
                    dma_req_dram_addr_o        = dma_wk_dram_ptr_q;
                    dma_req_tcdm_addr_o        = dma_w_tcdm_addr;
                    dma_req_size_bytes_o       = sz_kv0_i;
                    dma_ring_slot_reserve_fire = dma_req_ready_i;
                    dma_ring_slot_commit_fire  = dma_req_ready_i;
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // 3.4 Registered DMA Pointers, Ring Credits, Ping-Pong & KV-Pin Books
    // -------------------------------------------------------------------------
    wire attn_next_hkv_done = mul_u32_u5({24'd0, dma_hkv_q} + 32'd1,
                                         {2'b0, cfg_gqa_group_m1_i} + 5'd1) >= {24'd0, cfg_attn_h_i};

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
            dma_w_pending_q        <= cfg_enable_dma_i && (is_gemm_i || is_attn_i);

            dma_a_h_q              <= 8'd0;
            dma_a_m_base_q         <= 16'd0;
            dma_a_pending_q        <= macro_dma_en_i;
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
            if (dma_ring_slot_reserve_fire && !w_release_credit_pulse_i)
                w_free_credits_q <= w_free_credits_q - 1'b1;
            else if (!dma_ring_slot_reserve_fire && w_release_credit_pulse_i)
                w_free_credits_q <= w_free_credits_q + 1'b1;

            if (dma_ring_slot_commit_fire)
                ring_wr_ptr_q <= (ring_wr_ptr_q == PTR_W'(NUM_RING_SLOTS - 1)) ? '0 : ring_wr_ptr_q + 1'b1;

            if (dma_ring_slot_done && !w_release_credit_pulse_i)
                w_ready_tiles_q <= w_ready_tiles_q + 1'b1;
            else if (!dma_ring_slot_done && w_release_credit_pulse_i)
                w_ready_tiles_q <= w_ready_tiles_q - 1'b1;

            if (w_release_credit_pulse_i)
                ring_rd_ptr_q <= (ring_rd_ptr_q == PTR_W'(NUM_RING_SLOTS - 1)) ? '0 : ring_rd_ptr_q + 1'b1;

            // In-flight capture & completion
            if (dma_issue_fire) begin
                dma_inflight_tag_q      <= dma_issue_tag;
                dma_inflight_buf_q      <= dma_issue_buf;
                dma_inflight_kv_phase_q <= dma_issue_kv_phase;
                dma_inflight_kv_dvi0_q  <= dma_issue_kv_dvi0;
            end else if (dma_done_i && !is_dma_wb_i) begin
                if (macro_dma_en_i && (dma_inflight_tag_q == DMA_TAG_A)) begin
                    a_buf_ready_q[dma_inflight_buf_q] <= 1'b1;
                end else if (macro_dma_en_i && (dma_inflight_tag_q == DMA_TAG_O)) begin
                    o_buf_wb_busy_q[dma_inflight_buf_q] <= 1'b0;
                    o_dma_sel_q                         <= ~o_dma_sel_q;
                end else if (attn_kv_pin_en_i && (dma_inflight_tag_q == DMA_TAG_W)) begin
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
            if (kv_pin_rel_hkv_pulse_i) begin
                pin_active_hkv_q  <= pin_active_hkv_q + 8'd1;
                k_pinned_cnt_q    <= 16'd0;
                v_pinned_cnt_q    <= 32'd0;
                kv_pin_hkv_done_q <= 1'b0;
            end

            // Inter-FSM handoffs from [CMP]
            if (macro_a_rel_pulse_i)
                a_buf_ready_q[a_cmp_rd_sel_i] <= 1'b0;

            if (o_flush_commit_pulse_i) begin
                o_buf_wb_busy_q[o_cmp_sel_i]  <= 1'b1;
                o_buf_wb_dram_q[o_cmp_sel_i]  <= o_wb_ptr_q;
                o_buf_wb_bytes_q[o_cmp_sel_i] <= is_gemm_i ? o_wb_bytes_i : attn_wb_bytes_i;
            end

            // DMA Activation / Query Prefetch Progression
            if (dma_issue_fire && macro_dma_en_i && (dma_issue_tag == DMA_TAG_A) && dma_a_pending_q) begin
                a_dma_wr_sel_q <= ~a_dma_wr_sel_q;
                if ((dma_a_m_base_q + dma_a_cur_m_sub) >= macro_m_total_i) begin
                    dma_a_m_base_q <= 16'd0;
                    if (is_attn_i && (dma_a_h_q < (cfg_attn_h_i - 8'd1)))
                        dma_a_h_q <= dma_a_h_q + 8'd1;
                    else
                        dma_a_pending_q <= 1'b0;
                end else begin
                    dma_a_m_base_q <= dma_a_m_base_q + eff_m_sub_i;
                end
            end

            // DMA Weight / KV Prefetch Progression
            if (dma_issue_fire && (dma_issue_tag == DMA_TAG_W) && dma_w_pending_q) begin
                if (is_gemm_i) begin
                    if (dma_kt_q == cfg_gemm_k_tiles_m1_i) begin
                        dma_kt_q <= 8'd0;
                        if (w_ring_reuse_active_i || (dma_gm_q == (dma_w_cur_m_sub - 16'd1))) begin
                            dma_gm_q <= 16'd0;
                            if (dma_gn_q == (cfg_gemm_n_tiles_i - 16'd1)) begin
                                dma_gn_q <= 16'd0;
                                if (!macro_dma_en_i || ((dma_w_m_base_q + dma_w_cur_m_sub) >= cfg_gemm_m_tiles_i)) begin
                                    dma_w_m_base_q  <= 16'd0;
                                    dma_w_pending_q <= 1'b0;
                                end else begin
                                    dma_w_m_base_q <= dma_w_m_base_q + eff_m_sub_i;
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
                end else if (attn_kv_pin_en_i) begin
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

    wire dma_a_last    = ((dma_a_m_base_q + eff_m_sub_i) >= macro_m_total_i);
    wire dma_w_gm_wrap = w_ring_reuse_active_i || (dma_gm_q == (dma_w_cur_m_sub - 16'd1));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ring_wr_off_q       <= 20'd0;
            ring_rd_off_q       <= 20'd0;
            dma_k_blk_q         <= 11'd0;
            dma_v_blk_q         <= 11'd0;
            dma_a_dram_ptr_q    <= 32'd0;
            a_head_base_q       <= 32'd0;
            dma_wk_dram_ptr_q   <= 32'd0;
            dma_wg_v_dram_ptr_q <= 32'd0;
            o_wb_ptr_q          <= 32'd0;
            o_wb_macro_base_q   <= 32'd0;
            a_rem_bytes_q       <= 32'd0;
        end else if (start_i || pinit_sync_i) begin
            ring_wr_off_q       <= 20'd0;
            ring_rd_off_q       <= 20'd0;
            dma_k_blk_q         <= 11'd0;
            dma_v_blk_q         <= cfg_attn_sep_kv_i ? k_blk_head_blk_i : packed_v_off_idx_i[10:0];
            dma_a_dram_ptr_q    <= cfg_dram_act_base_i;
            a_head_base_q       <= cfg_dram_act_base_i;
            dma_wk_dram_ptr_q   <= cfg_dram_kv_base_i;
            dma_wg_v_dram_ptr_q <= is_gemm_i ? cfg_dram_kv_base_i : cfg_dram_v_base_i;
            o_wb_ptr_q          <= cfg_dram_out_base_i;
            o_wb_macro_base_q   <= cfg_dram_out_base_i;
            a_rem_bytes_q       <= a_step_h_i;
        end else begin
            if (dma_ring_slot_commit_fire)
                ring_wr_off_q <= (ring_wr_ptr_q == PTR_W'(NUM_RING_SLOTS - 1))
                                 ? 20'd0 : (ring_wr_off_q + SLOT_SIZE_BYTES[19:0]);
            if (w_release_credit_pulse_i)
                ring_rd_off_q <= (ring_rd_ptr_q == PTR_W'(NUM_RING_SLOTS - 1))
                                 ? 20'd0 : (ring_rd_off_q + SLOT_SIZE_BYTES[19:0]);

            if (o_flush_commit_pulse_i) begin
                if (is_gemm_i) begin
                    if (!last_o_chunk_i) begin
                        o_wb_ptr_q <= o_wb_ptr_q + o_wb_bytes_i;
                    end else begin
                        if (!last_macro_m_i) begin
                            o_wb_macro_base_q <= o_wb_macro_base_q + o_wb_macro_step_i;
                            o_wb_ptr_q        <= o_wb_macro_base_q + o_wb_macro_step_i;
                        end else begin
                            o_wb_macro_base_q <= cfg_dram_out_base_i;
                            o_wb_ptr_q        <= cfg_dram_out_base_i;
                        end
                    end
                end else begin
                    o_wb_ptr_q <= o_wb_ptr_q + attn_wb_bytes_i;
                end
            end

            if (kv_pin_rel_hkv_pulse_i) begin
                dma_k_blk_q <= 11'd0;
                dma_v_blk_q <= cfg_attn_sep_kv_i ? k_blk_head_blk_i : packed_v_off_idx_i[10:0];
            end
            if (dma_issue_fire && (dma_issue_tag == DMA_TAG_W) && attn_kv_pin_en_i) begin
                if (cfg_attn_sep_kv_i) begin
                    if (!dma_kv_phase_q)
                        dma_k_blk_q <= dma_k_blk_q + k_preload_step_blk_i;
                    else
                        dma_v_blk_q <= dma_v_blk_q + 11'd1;
                end else begin
                    if (dma_dvi_q == 16'd0)
                        dma_k_blk_q <= dma_k_blk_q + packed_k_step_blk_i;
                    else
                        dma_v_blk_q <= dma_v_blk_q + 11'd1;
                end
            end

            if (dma_issue_fire && (dma_issue_tag == DMA_TAG_A)) begin
                if (dma_a_last) begin
                    a_rem_bytes_q <= a_step_h_i;
                    if (is_attn_i && (dma_a_h_q < (cfg_attn_h_i - 8'd1))) begin
                        a_head_base_q    <= a_head_base_q + a_step_h_i;
                        dma_a_dram_ptr_q <= a_head_base_q + a_step_h_i;
                    end else begin
                        a_head_base_q    <= cfg_dram_act_base_i;
                        dma_a_dram_ptr_q <= cfg_dram_act_base_i;
                    end
                end else begin
                    a_rem_bytes_q    <= a_rem_bytes_q - a_full_bytes_i;
                    dma_a_dram_ptr_q <= dma_a_dram_ptr_q + a_full_bytes_i;
                end
            end

            if (dma_issue_fire && (dma_issue_tag == DMA_TAG_W)) begin
                if (is_gemm_i) begin
                    if (dma_kt_q == cfg_gemm_k_tiles_m1_i) begin
                        if (dma_w_gm_wrap) begin
                            if (dma_gn_q == (cfg_gemm_n_tiles_i - 16'd1)) begin
                                dma_wg_v_dram_ptr_q <= cfg_dram_kv_base_i;
                                dma_wk_dram_ptr_q   <= cfg_dram_kv_base_i;
                            end else begin
                                dma_wg_v_dram_ptr_q <= dma_wg_v_dram_ptr_q + cfg_stride_tile_i;
                                dma_wk_dram_ptr_q   <= dma_wg_v_dram_ptr_q + cfg_stride_tile_i;
                            end
                        end else begin
                            dma_wk_dram_ptr_q <= dma_wg_v_dram_ptr_q;
                        end
                    end else begin
                        dma_wk_dram_ptr_q <= dma_wk_dram_ptr_q + kt_stride_i;
                    end
                end else begin
                    if (cfg_attn_sep_kv_i) begin
                        if (!dma_kv_phase_q)
                            dma_wk_dram_ptr_q <= dma_wk_dram_ptr_q + kt_stride_attn_i;
                        else
                            dma_wg_v_dram_ptr_q <= dma_wg_v_dram_ptr_q + 32'd144;
                    end else begin
                        dma_wk_dram_ptr_q <= dma_wk_dram_ptr_q + cfg_stride_tile_i;
                    end
                end
            end
        end
    end

endmodule
