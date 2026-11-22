`timescale 1ns/1ps

module snax_flash_csr_snax #(
    parameter bit ENABLE_LEGACY_DIRECT_CSR = 1'b0,
    parameter bit DOUBLE_BUFFER            = 1'b0
)(
    input  logic        clk,
    input  logic        rst_n,

    // SNAX CSR request / response
    input  logic        csr_req_valid_i,
    output logic        csr_req_ready_o,
    input  logic        csr_req_write_i,
    input  logic [11:0] csr_req_addr_i,
    input  logic [31:0] csr_req_data_i,
    output logic        csr_rsp_valid_o,
    output logic [31:0] csr_rsp_data_o,
    input  logic        csr_rsp_ready_i,

    input  logic        cfg_ready_i,

    input  logic        seq_busy,
    input  logic        seq_done,
    input  logic        act_busy,
    input  logic        wt_busy,
    input  logic        wo_busy,
    input  logic        v_busy,
    input  logic        v_done,
    input  logic        rd_busy,
    input  logic        rd_done,
    input  logic        p_fifo_full,
    input  logic        p_fifo_empty,
    input  logic        p_fifo_almost_full,
    input  logic        l_ready,
    input  logic        core_fp16_valid,
    input  logic        sfu_busy,
    input  logic        sfu_done,
    input  logic        sfu_seq_busy,

    output logic        start_pulse,
    output logic        src_sel,
    output logic        act_ready_force,
    output logic        wt_ready_force,

    output logic        act_start,
    output logic [31:0] act_base,
    output logic [31:0] act_k_blocks,
    output logic [31:0] act_k_stride,
    output logic [31:0] act_tokens,
    output logic [31:0] act_token_stride,

    output logic        wt_start,
    output logic [31:0] wt_base,
    output logic [31:0] wt_repeat,
    output logic [31:0] wt_k_blocks,
    output logic [31:0] wt_k_stride,
    output logic [31:0] wt_num_tiles,
    output logic [31:0] wt_tile_step,

    output logic        wo_start,
    output logic        wo_mode_fp16,
    output logic [31:0] wo_base,
    output logic [31:0] wo_num_vectors,
    output logic [31:0] wo_line_stride,

    output logic [1:0]  core_mode,
    output logic        core_row_start,
    output logic [15:0] core_key_mask,
    output logic        core_enable_slq,
    output logic        core_tile_drain_req,
    output logic [3:0]  core_tile_drain_idx,
    output logic        core_tile_clear_on_drain,
    output logic [15:0] core_lane_valid,
    output logic [3:0]  core_lane_acc_idx,

    output logic        v_start,
    output logic [31:0] v_base,
    output logic [31:0] v_num_batches,

    output logic        rd_start,
    output logic [31:0] rd_base,
    output logic [31:0] rd_num_vectors,
    output logic [31:0] rd_line_stride,

    // SFU control/status and second read stream
    output logic        sfu_start,
    output logic        sfu_en,
    output logic        sfu_out_dst,     // 0: logq FP4, 1: write-out FP16
    output logic        sfu_in_a_src,    // 0: TCDM rd_fp16, 1: direct core_fp16_packed
    output logic        sfu_seq_en,      // 1: hardware snax_sfu_sequencer batch/2D mode
    output logic [2:0]  sfu_op,
    output logic [15:0] sfu_num_vecs,
    output logic [31:0] sfu_inv_n_fp32,
    output logic [31:0] sfu_eps_fp32,
    output logic        rd_b_start,
    output logic [31:0] rd_b_base,
    output logic [31:0] rd_b_num_vectors,
    output logic [31:0] rd_b_line_stride,

    output logic [1:0]  quant_src_sel,

    // Sequencer descriptor (all CPU-programmable)
    output logic [1:0]  seq_op_mode,        // 00 GEMM, 01 FlashAttention
    output logic        seq_is_causal,
    output logic        seq_enable_dma,
    output logic [1:0]  seq_out_mode,      // 00 FP16, 01 FP4 linear, 10 FP4 bank-scatter
    output logic        seq_dma_wb,        // 1: Standalone DMA transfer (TCDM <-> DRAM)
    output logic        seq_gemm_decode,   // 1: GEMM Decode mode (M=1, slot 0 only)
    output logic [7:0]  seq_gemm_k_tiles_m1, // K-tiles - 1 (0 => 1 K-tile)
    output logic [7:0]  seq_gqa_group_m1,    // GQA group size - 1 (0 => MHA 1:1)
    output logic        seq_act_bank_layout, // 1: 1 token per bank across banks 16..31 (stride 0x10)
    output logic        seq_out_bank_layout, // 1: 1 token per bank FP16 output scatter
    output logic        wo_wrap_en,
    output logic        rd_wrap_en,
    output logic        rd_b_wrap_en,
    output logic        rd_b_wrap_bank_inc,
    output logic [15:0] wo_wrap_words_m1,
    output logic [15:0] rd_wrap_words_m1,
    output logic [15:0] rd_b_wrap_words_m1,
    output logic [7:0]  seq_attn_h,
    output logic [15:0] seq_attn_q_tiles,
    output logic [3:0]  seq_attn_m_q,
    output logic [15:0] seq_attn_dv_tiles,
    output logic [15:0] seq_attn_kv_tiles,
    output logic [15:0] seq_attn_k_blocks,
    output logic [15:0] seq_gemm_n_tiles,
    output logic [15:0] seq_gemm_m_tiles,
    output logic [15:0] seq_gemm_k_blocks,
    output logic [31:0] seq_tcdm_ring_base,
    output logic [31:0] seq_tcdm_q_base,
    output logic [31:0] seq_tcdm_out_base,
    output logic [31:0] seq_dram_kv_base,
    output logic [31:0] seq_stride_tile,
    // Opt-in Macro-Tiling & Extended Attention DMA controls
    output logic        seq_gemm_macro_dma,
    output logic        seq_use_legacy_gemm,
    output logic [7:0]  seq_gemm_m_sub_tiles,
    output logic [7:0]  seq_gemm_n_sub_tiles,
    output logic [31:0] seq_dram_act_base,
    output logic [31:0] seq_dram_out_base,
    output logic [31:0] seq_macro_a_buf_stride,
    output logic [31:0] seq_macro_o_buf_stride,
    output logic        seq_attn_kv_pin_en,
    output logic        seq_attn_sep_kv_dram,
    output logic [31:0] seq_dram_v_base,
    output logic [13:0] seq_attn_kv_past_tiles
);

    localparam int NUM_RW         = 70;
    localparam int NUM_RO         = 10;
    localparam int START_IDX      = 53;

    localparam int RW_CTRL        = 0;
    localparam int RW_ACT_BASE    = 1;
    localparam int RW_ACT_KB      = 2;
    localparam int RW_ACT_KS      = 3;
    localparam int RW_ACT_TOK     = 4;
    localparam int RW_ACT_TSTR    = 5;
    localparam int RW_WT_BASE     = 6;
    localparam int RW_WT_REP      = 7;
    localparam int RW_WT_KB       = 8;
    localparam int RW_WT_KS       = 9;
    localparam int RW_WT_NT       = 10;
    localparam int RW_WT_TS       = 11;
    localparam int RW_WO_MODE     = 12;
    localparam int RW_WO_BASE     = 13;
    localparam int RW_WO_NV       = 14;
    localparam int RW_WO_LS       = 15;
    localparam int RW_CORE_MODE   = 16;
    localparam int RW_CORE_ROW    = 17;
    localparam int RW_CORE_MASK   = 18;
    localparam int RW_CORE_SLQ    = 19;
    localparam int RW_CORE_DR     = 20;
    localparam int RW_CORE_DI     = 21;
    localparam int RW_CORE_CLR    = 22;
    localparam int RW_CORE_LV     = 23;
    localparam int RW_CORE_LA     = 24;
    localparam int RW_V_BASE      = 25;
    localparam int RW_V_NB        = 26;
    localparam int RW_RD_BASE     = 27;
    localparam int RW_RD_NV       = 28;
    localparam int RW_RD_LS       = 29;
    localparam int RW_QUANT       = 30;

    // Sequencer descriptor registers (index 31..45), start moves to 47.
    localparam int RW_SEQ_CTRL    = 31;
    localparam int RW_SEQ_ATTN_H  = 32;
    localparam int RW_SEQ_ATTN_QT = 33;
    localparam int RW_SEQ_ATTN_MQ = 34;
    localparam int RW_SEQ_ATTN_DV = 35;
    localparam int RW_SEQ_ATTN_KV = 36;
    localparam int RW_SEQ_ATTN_KB = 37;
    localparam int RW_SEQ_GEMM_NT = 38;
    localparam int RW_SEQ_GEMM_MT = 39;
    localparam int RW_SEQ_GEMM_KB = 40;
    localparam int RW_SEQ_RING    = 41;
    localparam int RW_SEQ_Q_BASE  = 42;
    localparam int RW_SEQ_OUT     = 43;
    localparam int RW_SEQ_DRAM_KV = 44;
    localparam int RW_SEQ_TILE_ST = 45;

    localparam int RW_SFU_CTRL    = 46;   // [0] en, [1] out_dst, [4:2] op, [20:5] num_vecs
    localparam int RW_SFU_INV_N   = 47;
    localparam int RW_SFU_EPS     = 48;
    localparam int RW_SFU_B_BASE  = 49;
    localparam int RW_SFU_B_NV    = 50;
    localparam int RW_SFU_B_LS    = 51;
    localparam int RW_WRAP_CFG    = 52;
    // The CSR manager reserves the slot at START_IDX as START.
    localparam int RW_START       = START_IDX;
    // Extended non-aliased Macro-Tiling & Attention DMA CSRs (past RO slots 54..63)
    localparam int RW_MACRO_SUB   = 64;   // [7:0] m_sub, [15:8] n_sub, [16] attn_kv_pin_en, [17] attn_sep_kv_dram
    localparam int RW_MACRO_ACT   = 65;   // DRAM Q/Act base
    localparam int RW_MACRO_OUT   = 66;   // DRAM O/Out base
    localparam int RW_MACRO_A_STR = 67;   // Q/A Ping-Pong TCDM buffer stride
    localparam int RW_MACRO_O_STR = 68;   // O Ping-Pong TCDM buffer stride
    localparam int RW_DRAM_V_BASE = 69;   // Separate DRAM V base for FlashAttention

    logic [NUM_RW-1:0][31:0] cfg_rw;
    logic [31:0]             start_mask;
    logic                    cfg_valid;
    logic                    cfg_commit;
    logic [NUM_RO-1:0][31:0] ro;

    assign ro[0] = {30'd0, seq_done, seq_busy};
    assign ro[1] = {31'd0, act_busy};
    assign ro[2] = {31'd0, wt_busy};
    assign ro[3] = {31'd0, wo_busy};
    assign ro[4] = {30'd0, v_done, v_busy};
    assign ro[5] = {30'd0, rd_done, rd_busy};
    assign ro[6] = {28'd0, core_fp16_valid, l_ready, p_fifo_empty, p_fifo_full};
    assign ro[7] = 32'h534E_4146;
    assign ro[8] = {31'd0, p_fifo_almost_full};
    assign ro[9] = {29'd0, sfu_seq_busy, sfu_done, sfu_busy};

    typedef struct packed {
        logic        src_sel;        // 0[0]
        logic        wo_mode_fp16;   // 12[0]
        logic [19:0] wo_base;        // 13[19:0]
        logic [15:0] wo_nv;          // 14[15:0]
        logic [31:0] wo_ls;          // 15
        logic [19:0] v_base;         // 25[19:0]
        logic [15:0] v_nb;           // 26[15:0]
        logic [19:0] rd_base;        // 27[19:0]
        logic [15:0] rd_nv;          // 28[15:0]
        logic [31:0] rd_ls;          // 29
        logic [31:0] seq_ctrl;       // 31
        logic [7:0]  attn_h;         // 32[7:0]
        logic [15:0] attn_qt;        // 33[15:0]
        logic [3:0]  attn_mq;        // 34[3:0]
        logic [15:0] attn_dv;        // 35[15:0]
        logic [15:0] attn_kv;        // 36[15:0]
        logic [15:0] attn_kb;        // 37[15:0]
        logic [15:0] gemm_nt;        // 38[15:0]
        logic [15:0] gemm_mt;        // 39[15:0]
        logic [15:0] gemm_kb;        // 40[15:0]
        logic [19:0] ring_base;      // 41[19:0]
        logic [19:0] q_base;         // 42[19:0]
        logic [19:0] out_base;       // 43[19:0]
        logic [31:0] dram_kv;        // 44
        logic [31:0] tile_st;        // 45
        logic [22:0] sfu_ctrl;       // 46[22:0]
        logic [19:0] sfu_inv_n_hi;   // 47[31:12]
        logic [19:0] sfu_eps_hi;     // 48[31:12]
        logic [19:0] sfu_b_base;     // 49[19:0]
        logic [15:0] sfu_b_nv;       // 50[15:0]
        logic [31:0] sfu_b_ls;       // 51
        logic [31:0] wrap_cfg;       // 52
        logic [31:0] macro_sub;      // 64
        logic [31:0] macro_act;      // 65
        logic [31:0] macro_out;      // 66
        logic [19:0] macro_a_str;    // 67[19:0]
        logic [19:0] macro_o_str;    // 68[19:0]
        logic [31:0] dram_v_base;    // 69
    } csr_cfg_t;

    function automatic logic [31:0] read_compact_csr(
        input csr_cfg_t    c,
        input logic [8:0]  start_m,
        input logic [11:0] addr
    );
        case (addr)
            12'(RW_CTRL):        return {31'd0, c.src_sel};
            12'(RW_WO_MODE):     return {31'd0, c.wo_mode_fp16};
            12'(RW_WO_BASE):     return {12'd0, c.wo_base};
            12'(RW_WO_NV):       return {16'd0, c.wo_nv};
            12'(RW_WO_LS):       return c.wo_ls;
            12'(RW_V_BASE):      return {12'd0, c.v_base};
            12'(RW_V_NB):        return {16'd0, c.v_nb};
            12'(RW_RD_BASE):     return {12'd0, c.rd_base};
            12'(RW_RD_NV):       return {16'd0, c.rd_nv};
            12'(RW_RD_LS):       return c.rd_ls;
            12'(RW_SEQ_CTRL):    return c.seq_ctrl;
            12'(RW_SEQ_ATTN_H):  return {24'd0, c.attn_h};
            12'(RW_SEQ_ATTN_QT): return {16'd0, c.attn_qt};
            12'(RW_SEQ_ATTN_MQ): return {28'd0, c.attn_mq};
            12'(RW_SEQ_ATTN_DV): return {16'd0, c.attn_dv};
            12'(RW_SEQ_ATTN_KV): return {16'd0, c.attn_kv};
            12'(RW_SEQ_ATTN_KB): return {16'd0, c.attn_kb};
            12'(RW_SEQ_GEMM_NT): return {16'd0, c.gemm_nt};
            12'(RW_SEQ_GEMM_MT): return {16'd0, c.gemm_mt};
            12'(RW_SEQ_GEMM_KB): return {16'd0, c.gemm_kb};
            12'(RW_SEQ_RING):    return {12'd0, c.ring_base};
            12'(RW_SEQ_Q_BASE):  return {12'd0, c.q_base};
            12'(RW_SEQ_OUT):     return {12'd0, c.out_base};
            12'(RW_SEQ_DRAM_KV): return c.dram_kv;
            12'(RW_SEQ_TILE_ST): return c.tile_st;
            12'(RW_SFU_CTRL):    return {9'd0, c.sfu_ctrl};
            12'(RW_SFU_INV_N):   return {c.sfu_inv_n_hi, 12'd0};
            12'(RW_SFU_EPS):     return {c.sfu_eps_hi, 12'd0};
            12'(RW_SFU_B_BASE):  return {12'd0, c.sfu_b_base};
            12'(RW_SFU_B_NV):    return {16'd0, c.sfu_b_nv};
            12'(RW_SFU_B_LS):    return c.sfu_b_ls;
            12'(RW_WRAP_CFG):    return c.wrap_cfg;
            12'(RW_START):       return {23'd0, start_m};
            12'(RW_MACRO_SUB):   return c.macro_sub;
            12'(RW_MACRO_ACT):   return c.macro_act;
            12'(RW_MACRO_OUT):   return c.macro_out;
            12'(RW_MACRO_A_STR): return {12'd0, c.macro_a_str};
            12'(RW_MACRO_O_STR): return {12'd0, c.macro_o_str};
            12'(RW_DRAM_V_BASE): return c.dram_v_base;
            default:             return 32'd0;
        endcase
    endfunction

    function automatic csr_cfg_t write_compact_csr(
        input csr_cfg_t    c,
        input logic [11:0] addr,
        input logic [31:0] data
    );
        csr_cfg_t nxt;
        nxt = c;
        case (addr)
            12'(RW_CTRL):        nxt.src_sel      = data[0];
            12'(RW_WO_MODE):     nxt.wo_mode_fp16 = data[0];
            12'(RW_WO_BASE):     nxt.wo_base      = data[19:0];
            12'(RW_WO_NV):       nxt.wo_nv        = data[15:0];
            12'(RW_WO_LS):       nxt.wo_ls        = data;
            12'(RW_V_BASE):      nxt.v_base       = data[19:0];
            12'(RW_V_NB):        nxt.v_nb         = data[15:0];
            12'(RW_RD_BASE):     nxt.rd_base      = data[19:0];
            12'(RW_RD_NV):       nxt.rd_nv        = data[15:0];
            12'(RW_RD_LS):       nxt.rd_ls        = data;
            12'(RW_SEQ_CTRL):    nxt.seq_ctrl     = data;
            12'(RW_SEQ_ATTN_H):  nxt.attn_h       = data[7:0];
            12'(RW_SEQ_ATTN_QT): nxt.attn_qt      = data[15:0];
            12'(RW_SEQ_ATTN_MQ): nxt.attn_mq      = data[3:0];
            12'(RW_SEQ_ATTN_DV): nxt.attn_dv      = data[15:0];
            12'(RW_SEQ_ATTN_KV): nxt.attn_kv      = data[15:0];
            12'(RW_SEQ_ATTN_KB): nxt.attn_kb      = data[15:0];
            12'(RW_SEQ_GEMM_NT): nxt.gemm_nt      = data[15:0];
            12'(RW_SEQ_GEMM_MT): nxt.gemm_mt      = data[15:0];
            12'(RW_SEQ_GEMM_KB): nxt.gemm_kb      = data[15:0];
            12'(RW_SEQ_RING):    nxt.ring_base    = data[19:0];
            12'(RW_SEQ_Q_BASE):  nxt.q_base       = data[19:0];
            12'(RW_SEQ_OUT):     nxt.out_base     = data[19:0];
            12'(RW_SEQ_DRAM_KV): nxt.dram_kv      = data;
            12'(RW_SEQ_TILE_ST): nxt.tile_st      = data;
            12'(RW_SFU_CTRL):    nxt.sfu_ctrl     = data[22:0];
            12'(RW_SFU_INV_N):   nxt.sfu_inv_n_hi = data[31:12];
            12'(RW_SFU_EPS):     nxt.sfu_eps_hi   = data[31:12];
            12'(RW_SFU_B_BASE):  nxt.sfu_b_base   = data[19:0];
            12'(RW_SFU_B_NV):    nxt.sfu_b_nv     = data[15:0];
            12'(RW_SFU_B_LS):    nxt.sfu_b_ls     = data;
            12'(RW_WRAP_CFG):    nxt.wrap_cfg     = data;
            12'(RW_MACRO_SUB):   nxt.macro_sub    = data;
            12'(RW_MACRO_ACT):   nxt.macro_act    = data;
            12'(RW_MACRO_OUT):   nxt.macro_out    = data;
            12'(RW_MACRO_A_STR): nxt.macro_a_str  = data[19:0];
            12'(RW_MACRO_O_STR): nxt.macro_o_str  = data[19:0];
            12'(RW_DRAM_V_BASE): nxt.dram_v_base  = data;
            default: ;
        endcase
        return nxt;
    endfunction

    generate
    if (ENABLE_LEGACY_DIRECT_CSR) begin : gen_legacy_csr
        logic [NUM_RW-1:0][31:0] cfg_rw;
        logic [31:0]             start_mask;
        logic                    cfg_valid;
        logic                    cfg_commit;

        snax_csr_manager #(
            .NUM_RW     (NUM_RW),
            .NUM_RO     (NUM_RO),
            .ADDR_WIDTH (12),
            .START_IDX  (START_IDX)
        ) u_manager (
            .clk                (clk),
            .rst_n              (rst_n),
            .csr_req_valid_i     (csr_req_valid_i),
            .csr_req_ready_o     (csr_req_ready_o),
            .csr_req_write_i     (csr_req_write_i),
            .csr_req_addr_i      (csr_req_addr_i),
            .csr_req_data_i      (csr_req_data_i),
            .csr_rsp_valid_o     (csr_rsp_valid_o),
            .csr_rsp_data_o      (csr_rsp_data_o),
            .csr_rsp_ready_i     (csr_rsp_ready_i),
            .cfg_rw_o            (cfg_rw),
            .start_mask_o        (start_mask),
            .cfg_valid_o         (cfg_valid),
            .cfg_ready_i         (cfg_ready_i),
            .cfg_commit_o        (cfg_commit),
            .ro_i                (ro)
        );

        assign src_sel         = cfg_rw[RW_CTRL][0];
        assign act_ready_force = cfg_rw[RW_CTRL][1];
        assign wt_ready_force  = cfg_rw[RW_CTRL][2];

        assign start_pulse = cfg_commit && start_mask[1];
        assign act_start   = cfg_commit && start_mask[2];
        assign wt_start    = cfg_commit && start_mask[3];
        assign wo_start    = cfg_commit && start_mask[4];
        assign v_start     = cfg_commit && start_mask[5];
        assign rd_start    = cfg_commit && start_mask[6];
        assign sfu_start   = cfg_commit && start_mask[7];
        assign rd_b_start  = cfg_commit && start_mask[8];

        assign act_base         = cfg_rw[RW_ACT_BASE];
        assign act_k_blocks     = cfg_rw[RW_ACT_KB];
        assign act_k_stride     = cfg_rw[RW_ACT_KS];
        assign act_tokens       = cfg_rw[RW_ACT_TOK];
        assign act_token_stride = cfg_rw[RW_ACT_TSTR];

        assign wt_base       = cfg_rw[RW_WT_BASE];
        assign wt_repeat     = cfg_rw[RW_WT_REP];
        assign wt_k_blocks   = cfg_rw[RW_WT_KB];
        assign wt_k_stride   = cfg_rw[RW_WT_KS];
        assign wt_num_tiles  = cfg_rw[RW_WT_NT];
        assign wt_tile_step  = cfg_rw[RW_WT_TS];

        assign wo_mode_fp16   = cfg_rw[RW_WO_MODE][0];
        assign wo_base        = cfg_rw[RW_WO_BASE];
        assign wo_num_vectors = cfg_rw[RW_WO_NV];
        assign wo_line_stride = cfg_rw[RW_WO_LS];

        assign core_mode              = cfg_rw[RW_CORE_MODE][1:0];
        assign core_row_start         = cfg_rw[RW_CORE_ROW][0];
        assign core_key_mask          = cfg_rw[RW_CORE_MASK][15:0];
        assign core_enable_slq        = cfg_rw[RW_CORE_SLQ][0];
        assign core_tile_drain_req    = cfg_rw[RW_CORE_DR][0];
        assign core_tile_drain_idx    = cfg_rw[RW_CORE_DI][3:0];
        assign core_tile_clear_on_drain = cfg_rw[RW_CORE_CLR][0];
        assign core_lane_valid        = cfg_rw[RW_CORE_LV][15:0];
        assign core_lane_acc_idx      = cfg_rw[RW_CORE_LA][3:0];

        assign v_base        = cfg_rw[RW_V_BASE];
        assign v_num_batches = cfg_rw[RW_V_NB];
        assign rd_base       = cfg_rw[RW_RD_BASE];
        assign rd_num_vectors= cfg_rw[RW_RD_NV];
        assign rd_line_stride= cfg_rw[RW_RD_LS];

        assign sfu_en          = cfg_rw[RW_SFU_CTRL][0];
        assign sfu_out_dst     = cfg_rw[RW_SFU_CTRL][1];
        assign sfu_op          = cfg_rw[RW_SFU_CTRL][4:2];
        assign sfu_num_vecs    = cfg_rw[RW_SFU_CTRL][20:5];
        assign sfu_in_a_src    = cfg_rw[RW_SFU_CTRL][21];
        assign sfu_seq_en      = cfg_rw[RW_SFU_CTRL][22];
        assign sfu_inv_n_fp32  = cfg_rw[RW_SFU_INV_N];
        assign sfu_eps_fp32    = cfg_rw[RW_SFU_EPS];
        assign rd_b_base       = cfg_rw[RW_SFU_B_BASE];
        assign rd_b_num_vectors= cfg_rw[RW_SFU_B_NV];
        assign rd_b_line_stride= cfg_rw[RW_SFU_B_LS];

        assign quant_src_sel = cfg_rw[RW_QUANT][1:0];

        assign seq_op_mode         = cfg_rw[RW_SEQ_CTRL][1:0];
        assign seq_is_causal       = cfg_rw[RW_SEQ_CTRL][2];
        assign seq_enable_dma      = cfg_rw[RW_SEQ_CTRL][3];
        assign seq_out_mode        = cfg_rw[RW_SEQ_CTRL][5:4];
        assign seq_dma_wb          = cfg_rw[RW_SEQ_CTRL][6];
        assign seq_gemm_decode     = cfg_rw[RW_SEQ_CTRL][7];
        assign seq_gemm_k_tiles_m1 = cfg_rw[RW_SEQ_CTRL][15:8];
        assign seq_gqa_group_m1    = cfg_rw[RW_SEQ_CTRL][23:16];
        assign seq_act_bank_layout = cfg_rw[RW_SEQ_CTRL][24];
        assign seq_out_bank_layout = cfg_rw[RW_SEQ_CTRL][25];
        assign wo_wrap_en          = cfg_rw[RW_SEQ_CTRL][26];
        assign rd_wrap_en          = cfg_rw[RW_SEQ_CTRL][27];
        assign rd_b_wrap_en        = cfg_rw[RW_SEQ_CTRL][28];
        assign rd_b_wrap_bank_inc  = cfg_rw[RW_SEQ_CTRL][29];
        assign wo_wrap_words_m1    = {5'd0, cfg_rw[RW_WRAP_CFG][10:0]};
        assign rd_wrap_words_m1    = {5'd0, cfg_rw[RW_WRAP_CFG][21:11]};
        assign rd_b_wrap_words_m1  = {6'd0, cfg_rw[RW_WRAP_CFG][31:22]};
        assign seq_attn_h          = cfg_rw[RW_SEQ_ATTN_H][7:0];
        assign seq_attn_q_tiles    = cfg_rw[RW_SEQ_ATTN_QT][15:0];
        assign seq_attn_m_q        = cfg_rw[RW_SEQ_ATTN_MQ][3:0];
        assign seq_attn_dv_tiles   = cfg_rw[RW_SEQ_ATTN_DV][15:0];
        assign seq_attn_kv_tiles   = cfg_rw[RW_SEQ_ATTN_KV][15:0];
        assign seq_attn_k_blocks   = cfg_rw[RW_SEQ_ATTN_KB][15:0];
        assign seq_gemm_n_tiles    = cfg_rw[RW_SEQ_GEMM_NT][15:0];
        assign seq_gemm_m_tiles    = cfg_rw[RW_SEQ_GEMM_MT][15:0];
        assign seq_gemm_k_blocks   = cfg_rw[RW_SEQ_GEMM_KB][15:0];
        assign seq_tcdm_ring_base  = cfg_rw[RW_SEQ_RING];
        assign seq_tcdm_q_base     = cfg_rw[RW_SEQ_Q_BASE];
        assign seq_tcdm_out_base   = cfg_rw[RW_SEQ_OUT];
        assign seq_dram_kv_base    = cfg_rw[RW_SEQ_DRAM_KV];
        assign seq_stride_tile     = cfg_rw[RW_SEQ_TILE_ST];

        wire use_ext_macro = (cfg_rw[RW_MACRO_SUB][15:0] != 16'd0) ||
                             (cfg_rw[RW_MACRO_ACT]       != 32'd0) ||
                             (cfg_rw[RW_MACRO_OUT]       != 32'd0) ||
                             (cfg_rw[RW_MACRO_A_STR]     != 32'd0) ||
                             (cfg_rw[RW_MACRO_O_STR]     != 32'd0);

        assign seq_gemm_macro_dma     = cfg_rw[RW_SEQ_CTRL][30];
        assign seq_use_legacy_gemm    = cfg_rw[RW_SEQ_CTRL][31];
        assign seq_gemm_m_sub_tiles   = (use_ext_macro || (seq_op_mode == 2'b01)) ? cfg_rw[RW_MACRO_SUB][7:0]  : cfg_rw[RW_SEQ_ATTN_H][7:0];
        assign seq_gemm_n_sub_tiles   = (use_ext_macro || (seq_op_mode == 2'b01)) ? cfg_rw[RW_MACRO_SUB][15:8] : cfg_rw[RW_SEQ_ATTN_H][15:8];
        assign seq_dram_act_base      = (use_ext_macro || (seq_op_mode == 2'b01)) ? cfg_rw[RW_MACRO_ACT]       : cfg_rw[RW_SEQ_ATTN_QT];
        assign seq_dram_out_base      = (use_ext_macro || (seq_op_mode == 2'b01)) ? cfg_rw[RW_MACRO_OUT]       : cfg_rw[RW_SEQ_ATTN_DV];
        assign seq_macro_a_buf_stride = (use_ext_macro || (seq_op_mode == 2'b01)) ? cfg_rw[RW_MACRO_A_STR]     : cfg_rw[RW_SEQ_ATTN_KV];
        assign seq_macro_o_buf_stride = (use_ext_macro || (seq_op_mode == 2'b01)) ? cfg_rw[RW_MACRO_O_STR]     : cfg_rw[RW_SEQ_ATTN_KB];
        assign seq_attn_kv_pin_en     = cfg_rw[RW_MACRO_SUB][16];
        assign seq_attn_sep_kv_dram   = cfg_rw[RW_MACRO_SUB][17];
        assign seq_dram_v_base        = cfg_rw[RW_DRAM_V_BASE];
        assign seq_attn_kv_past_tiles = cfg_rw[RW_MACRO_SUB][31:18];
    end else begin : gen_compact_csr
        csr_cfg_t    r_cfg;
        logic [8:0]  r_start_mask;
        logic        cfg_ready_q;
        logic        req_vld_q;
        logic        req_wr_q;
        logic [11:0] req_addr_q;
        logic [31:0] req_data_q;
        logic        rsp_valid_q;
        logic [31:0] rsp_data_q;
        logic        commit_q;
        logic        pending_w, commit_d1_w;

        wire req_fire   = csr_req_valid_i && csr_req_ready_o;
        wire addr_is_ro = (req_addr_q >= 12'd54) && (req_addr_q < 12'd64);

        assign csr_req_ready_o = !req_vld_q && !rsp_valid_q && !pending_w && !commit_d1_w && !commit_q;
        assign csr_rsp_valid_o = rsp_valid_q;
        assign csr_rsp_data_o  = rsp_data_q;

        if (DOUBLE_BUFFER) begin : gen_db
            csr_cfg_t    sh_cfg_q, act_cfg_q;
            logic [8:0]  sh_start_q, act_start_q;
            logic        pending_q, commit_d1_q;

            assign pending_w   = pending_q;
            assign commit_d1_w = commit_d1_q;
            assign r_cfg        = act_cfg_q;
            assign r_start_mask = act_start_q;

            wire wr_fire = req_vld_q && req_wr_q;
            wire rd_fire = req_vld_q && !req_wr_q;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    sh_cfg_q    <= '0;
                    act_cfg_q   <= '0;
                    sh_start_q  <= 9'd0;
                    act_start_q <= 9'd0;
                    cfg_ready_q <= 1'b0;
                    req_vld_q   <= 1'b0;
                    req_wr_q    <= 1'b0;
                    req_addr_q  <= 12'd0;
                    req_data_q  <= 32'd0;
                    rsp_valid_q <= 1'b0;
                    rsp_data_q  <= 32'd0;
                    pending_q   <= 1'b0;
                    commit_d1_q <= 1'b0;
                    commit_q    <= 1'b0;
                end else begin
                    cfg_ready_q <= cfg_ready_i;
                    commit_q    <= commit_d1_q;
                    commit_d1_q <= 1'b0;

                    if (pending_q && cfg_ready_q) begin
                        act_cfg_q   <= sh_cfg_q;
                        act_start_q <= sh_start_q;
                        pending_q   <= 1'b0;
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
                    end else if (wr_fire) begin
                        req_vld_q   <= 1'b0;
                        rsp_valid_q <= 1'b1;
                        rsp_data_q  <= 32'd0;
                        if (req_addr_q == 12'(RW_START)) begin
                            sh_start_q <= req_data_q[8:0];
                            pending_q  <= req_data_q[0];
                        end else begin
                            sh_cfg_q <= write_compact_csr(sh_cfg_q, req_addr_q, req_data_q);
                        end
                    end else if (rd_fire) begin
                        req_vld_q   <= 1'b0;
                        rsp_valid_q <= 1'b1;
                        rsp_data_q  <= addr_is_ro ? ro[req_addr_q - 12'd54]
                                                  : read_compact_csr(sh_cfg_q, sh_start_q, req_addr_q);
                    end
                end
            end
        end else begin : gen_sb
            csr_cfg_t   sb_cfg_q;
            logic [8:0] sb_start_q;

            assign pending_w    = 1'b0;
            assign commit_d1_w  = 1'b0;
            assign r_cfg        = sb_cfg_q;
            assign r_start_mask = sb_start_q;

            // Single-buffer: allow reads anytime, allow writes once cfg_ready_q is high
            wire wr_fire = req_vld_q &&  req_wr_q && cfg_ready_q;
            wire rd_fire = req_vld_q && !req_wr_q;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    sb_cfg_q    <= '0;
                    sb_start_q  <= 9'd0;
                    cfg_ready_q <= 1'b0;
                    req_vld_q   <= 1'b0;
                    req_wr_q    <= 1'b0;
                    req_addr_q  <= 12'd0;
                    req_data_q  <= 32'd0;
                    rsp_valid_q <= 1'b0;
                    rsp_data_q  <= 32'd0;
                    commit_q    <= 1'b0;
                end else begin
                    cfg_ready_q <= cfg_ready_i;
                    commit_q    <= 1'b0;

                    if (rsp_valid_q && csr_rsp_ready_i) begin
                        rsp_valid_q <= 1'b0;
                    end

                    if (req_fire) begin
                        req_vld_q  <= 1'b1;
                        req_wr_q   <= csr_req_write_i;
                        req_addr_q <= csr_req_addr_i;
                        req_data_q <= csr_req_data_i;
                    end else if (wr_fire) begin
                        req_vld_q   <= 1'b0;
                        rsp_valid_q <= 1'b1;
                        rsp_data_q  <= 32'd0;
                        if (req_addr_q == 12'(RW_START)) begin
                            sb_start_q <= req_data_q[8:0];
                            commit_q   <= req_data_q[0];
                        end else begin
                            sb_cfg_q <= write_compact_csr(sb_cfg_q, req_addr_q, req_data_q);
                        end
                    end else if (rd_fire) begin
                        req_vld_q   <= 1'b0;
                        rsp_valid_q <= 1'b1;
                        rsp_data_q  <= addr_is_ro ? ro[req_addr_q - 12'd54]
                                                  : read_compact_csr(sb_cfg_q, sb_start_q, req_addr_q);
                    end
                end
            end
        end

        assign src_sel         = r_cfg.src_sel;
        assign act_ready_force = 1'b0;
        assign wt_ready_force  = 1'b0;

        assign start_pulse = commit_q && r_start_mask[1];
        assign act_start   = 1'b0;
        assign wt_start    = 1'b0;
        assign wo_start    = commit_q && r_start_mask[4];
        assign v_start     = commit_q && r_start_mask[5];
        assign rd_start    = commit_q && r_start_mask[6];
        assign sfu_start   = commit_q && r_start_mask[7];
        assign rd_b_start  = commit_q && r_start_mask[8];

        assign act_base         = 32'd0;
        assign act_k_blocks     = 32'd0;
        assign act_k_stride     = 32'd0;
        assign act_tokens       = 32'd0;
        assign act_token_stride = 32'd0;

        assign wt_base      = 32'd0;
        assign wt_repeat    = 32'd0;
        assign wt_k_blocks  = 32'd0;
        assign wt_k_stride  = 32'd0;
        assign wt_num_tiles = 32'd0;
        assign wt_tile_step = 32'd0;

        assign wo_mode_fp16   = r_cfg.wo_mode_fp16;
        assign wo_base        = {12'd0, r_cfg.wo_base};
        assign wo_num_vectors = {16'd0, r_cfg.wo_nv};
        assign wo_line_stride = r_cfg.wo_ls;

        assign core_mode                = 2'd0;
        assign core_row_start           = 1'b0;
        assign core_key_mask            = 16'd0;
        assign core_enable_slq          = 1'b0;
        assign core_tile_drain_req      = 1'b0;
        assign core_tile_drain_idx      = 4'd0;
        assign core_tile_clear_on_drain = 1'b0;
        assign core_lane_valid          = 16'd0;
        assign core_lane_acc_idx        = 4'd0;

        assign v_base         = {12'd0, r_cfg.v_base};
        assign v_num_batches  = {16'd0, r_cfg.v_nb};
        assign rd_base        = {12'd0, r_cfg.rd_base};
        assign rd_num_vectors = {16'd0, r_cfg.rd_nv};
        assign rd_line_stride = r_cfg.rd_ls;

        assign sfu_en           = r_cfg.sfu_ctrl[0];
        assign sfu_out_dst      = r_cfg.sfu_ctrl[1];
        assign sfu_op           = r_cfg.sfu_ctrl[4:2];
        assign sfu_num_vecs     = r_cfg.sfu_ctrl[20:5];
        assign sfu_in_a_src     = r_cfg.sfu_ctrl[21];
        assign sfu_seq_en       = r_cfg.sfu_ctrl[22];
        assign sfu_inv_n_fp32   = {r_cfg.sfu_inv_n_hi, 12'd0};
        assign sfu_eps_fp32     = {r_cfg.sfu_eps_hi, 12'd0};
        assign rd_b_base        = {12'd0, r_cfg.sfu_b_base};
        assign rd_b_num_vectors = {16'd0, r_cfg.sfu_b_nv};
        assign rd_b_line_stride = r_cfg.sfu_b_ls;

        assign quant_src_sel = 2'd0;

        assign seq_op_mode         = r_cfg.seq_ctrl[1:0];
        assign seq_is_causal       = r_cfg.seq_ctrl[2];
        assign seq_enable_dma      = r_cfg.seq_ctrl[3];
        assign seq_out_mode        = r_cfg.seq_ctrl[5:4];
        assign seq_dma_wb          = r_cfg.seq_ctrl[6];
        assign seq_gemm_decode     = r_cfg.seq_ctrl[7];
        assign seq_gemm_k_tiles_m1 = r_cfg.seq_ctrl[15:8];
        assign seq_gqa_group_m1    = r_cfg.seq_ctrl[23:16];
        assign seq_act_bank_layout = r_cfg.seq_ctrl[24];
        assign seq_out_bank_layout = r_cfg.seq_ctrl[25];
        assign wo_wrap_en          = r_cfg.seq_ctrl[26];
        assign rd_wrap_en          = r_cfg.seq_ctrl[27];
        assign rd_b_wrap_en        = r_cfg.seq_ctrl[28];
        assign rd_b_wrap_bank_inc  = r_cfg.seq_ctrl[29];
        assign wo_wrap_words_m1    = {5'd0, r_cfg.wrap_cfg[10:0]};
        assign rd_wrap_words_m1    = {5'd0, r_cfg.wrap_cfg[21:11]};
        assign rd_b_wrap_words_m1  = {6'd0, r_cfg.wrap_cfg[31:22]};
        assign seq_attn_h          = r_cfg.attn_h;
        assign seq_attn_q_tiles    = r_cfg.attn_qt;
        assign seq_attn_m_q        = r_cfg.attn_mq;
        assign seq_attn_dv_tiles   = r_cfg.attn_dv;
        assign seq_attn_kv_tiles   = r_cfg.attn_kv;
        assign seq_attn_k_blocks   = r_cfg.attn_kb;
        assign seq_gemm_n_tiles    = r_cfg.gemm_nt;
        assign seq_gemm_m_tiles    = r_cfg.gemm_mt;
        assign seq_gemm_k_blocks   = r_cfg.gemm_kb;
        assign seq_tcdm_ring_base  = {12'd0, r_cfg.ring_base};
        assign seq_tcdm_q_base     = {12'd0, r_cfg.q_base};
        assign seq_tcdm_out_base   = {12'd0, r_cfg.out_base};
        assign seq_dram_kv_base    = r_cfg.dram_kv;
        assign seq_stride_tile     = r_cfg.tile_st;

        assign seq_gemm_macro_dma     = r_cfg.seq_ctrl[30];
        assign seq_use_legacy_gemm    = r_cfg.seq_ctrl[31];
        assign seq_gemm_m_sub_tiles   = r_cfg.macro_sub[7:0];
        assign seq_gemm_n_sub_tiles   = r_cfg.macro_sub[15:8];
        assign seq_dram_act_base      = r_cfg.macro_act;
        assign seq_dram_out_base      = r_cfg.macro_out;
        assign seq_macro_a_buf_stride = {12'd0, r_cfg.macro_a_str};
        assign seq_macro_o_buf_stride = {12'd0, r_cfg.macro_o_str};
        assign seq_attn_kv_pin_en     = r_cfg.macro_sub[16];
        assign seq_attn_sep_kv_dram   = r_cfg.macro_sub[17];
        assign seq_dram_v_base        = r_cfg.dram_v_base;
        assign seq_attn_kv_past_tiles = r_cfg.macro_sub[31:18];
    end
    endgenerate

endmodule
