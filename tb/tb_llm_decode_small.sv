// =============================================================================
// tb_llm_decode_small.sv
//
// Small-size complete LLM Decode layer end-to-end on snax_llm_top_clean
// (dimensions matching tb_snax_llm_layer_fused, running in Decode mode):
//   D_MODEL = 128, D_FF = 256, H_Q = 4, H_KV = 4, D_K = D_V = 32
//   Decode: M_Q = 1, Q_TILES = 1 (1 new token at pos N_KV-1 attending to N_KV)
//   HW GEMM tile = 16 rows (row 0 = active decode token, rows 1..15 = 0)
//   512-bit DRAM -> 4-Port Gearbox DMA -> 4x128KB TCDM Ring Buffer
// =============================================================================
`timescale 1ns/1ps

module tb_llm_decode_small;

    `ifndef N_KV
        `define N_KV 64
    `endif
    localparam int N_KV           = `N_KV;
    localparam int M_TOKENS       = 16;
    localparam int D_MODEL        = 128;
    localparam int D_FF           = 256;
    localparam int H_Q            = 4;
    localparam int H_KV           = 4;
    localparam int GQA_M1         = (H_Q / H_KV) - 1; // 0
    localparam int D_K            = 32;
    localparam int D_V            = 32;
    localparam int M_Q            = 1;
    localparam int Q_TILES        = 1;
    localparam int EVAL_TOKENS    = 1;
    localparam int KV_TILES       = N_KV / 16;
    localparam int KV_TILES_PAST  = KV_TILES - 1;
    localparam int DV_TILES       = D_V / 16;         // 2
    localparam int K_PACKETS_QKV  = D_MODEL / 16;     // 8
    localparam int K_PACKETS_DOWN = D_FF / 16;        // 16
    localparam int K_PACKETS_QK   = D_K / 16;         // 2
    localparam int N_TILES_MODEL  = D_MODEL / 16;     // 8
    localparam int N_TILES_KV     = (H_KV * D_K) / 16;// 8
    localparam int N_TILES_FF     = D_FF / 16;        // 16
    localparam int N_CHUNKS       = 2;
    localparam int CHUNK_NT       = N_TILES_FF / N_CHUNKS; // 8

    localparam int V_BASE_BLOCK   = ((H_Q * KV_TILES * K_PACKETS_QK) > 16) ?
                                    (H_Q * KV_TILES * K_PACKETS_QK) : 16;
    localparam int TB_BANK_DEPTH  = 8192;
    localparam int PAST_KV_WORDS  = H_KV * KV_TILES * 2 * 16; // K: K_PACKETS_QK=2, V: DV_TILES=2
    localparam int CLK_PERIOD     = 10;

    localparam int NUM_FP16       = M_TOKENS * D_MODEL; // 2048
    localparam int ROW_WORDS      = D_MODEL / 4;        // 32
    localparam int NUM_VEC_XN     = EVAL_TOKENS * K_PACKETS_QKV; // 8
    localparam int NUM_PRELOAD    = 608;                // X(512)+G1(32)+G2(32)+RoPE(32)
    localparam int DRAM_WORDS     = 4096;               // 256 KB
    localparam int DRAM_WT_BEATS  = 1696;               // 108,544 B / 64

    localparam logic [31:0] INV_D_MODEL_FP32 = 32'h3C000000; // 1/128

    // TCDM bank map (banks 0..15 = Ring Buffer & KV cache)
    localparam int ACT_BANK   = 16;
    localparam int OUT_BANK   = 17;
    localparam int O_ACT_BANK = 20;
    localparam int G_STG_BANK = 21;
    localparam int FF1_BANK   = 22;
    localparam int X_BANK     = 24;
    localparam int XN_BANK    = 25;
    localparam int RES1_BANK  = 26;
    localparam int FIN_BANK   = 27;
    localparam int GAMMA_BANK = 28;
    localparam int ROPE_BANK  = 29;
    localparam int QPKT_BANK  = 30;
    localparam int GAMMA1_WORD = 0;
    localparam int GAMMA2_WORD = 32;

    // DRAM byte offsets
    localparam logic [31:0] DRAM_WQ  = 32'h0000_0000;
    localparam logic [31:0] DRAM_WK  = 32'h0000_3000;
    localparam logic [31:0] DRAM_WV  = 32'h0000_6000;
    localparam logic [31:0] DRAM_WO  = 32'h0000_9000;
    localparam logic [31:0] DRAM_WG  = 32'h0000_C000;
    localparam logic [31:0] DRAM_WU  = 32'h0001_1000;
    localparam logic [31:0] DRAM_WD  = 32'h0001_6000;
    localparam logic [31:0] DRAM_OUT = 32'h0002_0000;

    // CSR map
    localparam logic [11:0] A_CTRL      = 12'd0;
    localparam logic [11:0] A_WO_MODE   = 12'd12;
    localparam logic [11:0] A_WO_BASE   = 12'd13;
    localparam logic [11:0] A_WO_NV     = 12'd14;
    localparam logic [11:0] A_WO_LS     = 12'd15;
    localparam logic [11:0] A_V_BASE    = 12'd25;
    localparam logic [11:0] A_V_NB      = 12'd26;
    localparam logic [11:0] A_RD_BASE   = 12'd27;
    localparam logic [11:0] A_RD_NV     = 12'd28;
    localparam logic [11:0] A_RD_LS     = 12'd29;
    localparam logic [11:0] A_SEQ_CTRL  = 12'd31;
    localparam logic [11:0] A_ATTN_H    = 12'd32;
    localparam logic [11:0] A_ATTN_QT   = 12'd33;
    localparam logic [11:0] A_ATTN_MQ   = 12'd34;
    localparam logic [11:0] A_ATTN_DV   = 12'd35;
    localparam logic [11:0] A_ATTN_KV   = 12'd36;
    localparam logic [11:0] A_ATTN_KB   = 12'd37;
    localparam logic [11:0] A_GEMM_NT   = 12'd38;
    localparam logic [11:0] A_GEMM_MT   = 12'd39;
    localparam logic [11:0] A_GEMM_KB   = 12'd40;
    localparam logic [11:0] A_RING_BASE = 12'd41;
    localparam logic [11:0] A_Q_BASE    = 12'd42;
    localparam logic [11:0] A_OUT_BASE  = 12'd43;
    localparam logic [11:0] A_DRAM_KV   = 12'd44;
    localparam logic [11:0] A_TILE_ST   = 12'd45;
    localparam logic [11:0] A_SFU_CTRL  = 12'd46;
    localparam logic [11:0] A_SFU_INV_N = 12'd47;
    localparam logic [11:0] A_SFU_EPS   = 12'd48;
    localparam logic [11:0] A_SFU_B_BASE= 12'd49;
    localparam logic [11:0] A_SFU_B_NV  = 12'd50;
    localparam logic [11:0] A_SFU_B_LS  = 12'd51;
    localparam logic [11:0] A_START     = 12'd53;
    localparam logic [11:0] A_RO_ID     = 12'd61;
    localparam logic [31:0] CSR_ID_VALUE = 32'h534E_4146;

    logic clk, rst_n;
    initial begin clk = 1'b0; forever #(CLK_PERIOD / 2) clk = ~clk; end

    int cyc_total, cyc_seq, cyc_sfu, cyc_mac, cyc_dma;
    int checks_pass, checks_fail;
    localparam int JOB_TIMEOUT = 2_000_000;

    // DUT interfaces
    logic        csr_req_valid, csr_req_ready, csr_req_write, csr_rsp_valid, csr_rsp_ready;
    logic [11:0] csr_req_addr;
    logic [31:0] csr_req_data, csr_rsp_data;
    assign csr_rsp_ready = 1'b1;
    logic        cpu_mem_req_valid, cpu_mem_req_ready, cpu_mem_req_write, cpu_mem_rsp_valid;
    logic [31:0] cpu_mem_req_addr;
    logic [71:0] cpu_mem_req_wdata, cpu_mem_rsp_rdata;
    logic [8:0]  cpu_mem_req_strb;

    logic [3:0]        dma_tcdm_req_valid, dma_tcdm_req_ready, dma_tcdm_req_write, dma_tcdm_rsp_valid;
    logic [3:0][31:0]  dma_tcdm_req_addr;
    logic [3:0][71:0]  dma_tcdm_req_wdata, dma_tcdm_rsp_rdata;
    logic [3:0][8:0]   dma_tcdm_req_strb;

    logic        dma_desc_valid, dma_desc_ready, dma_desc_done;
    logic [1:0]  dma_desc_dir;
    logic [31:0] dma_desc_dram_addr, dma_desc_tcdm_addr, dma_desc_size;

    logic         dram_req_valid, dram_req_ready, dram_req_write, dram_rsp_valid;
    logic [31:0]  dram_req_addr;
    logic [511:0] dram_req_wdata, dram_rsp_rdata;
    logic [63:0]  dram_req_strb;
    logic [31:0]  stat_rd_beats, stat_wr_beats, stat_busy_cycles;
    logic         busy, done_irq;

    snax_llm_top_clean #(.BANK_DEPTH(TB_BANK_DEPTH)) dut (
        .clk                     (clk),
        .rst_n                   (rst_n),
        .cpu_csr_req_valid_i     (csr_req_valid),
        .cpu_csr_req_ready_o     (csr_req_ready),
        .cpu_csr_req_write_i     (csr_req_write),
        .cpu_csr_req_addr_i      (csr_req_addr),
        .cpu_csr_req_data_i      (csr_req_data),
        .cpu_csr_rsp_valid_o     (csr_rsp_valid),
        .cpu_csr_rsp_ready_i     (csr_rsp_ready),
        .cpu_csr_rsp_data_o      (csr_rsp_data),
        .cpu_mem_req_valid_i     (cpu_mem_req_valid),
        .cpu_mem_req_ready_o     (cpu_mem_req_ready),
        .cpu_mem_req_addr_i      (cpu_mem_req_addr),
        .cpu_mem_req_write_i     (cpu_mem_req_write),
        .cpu_mem_req_wdata_i     (cpu_mem_req_wdata),
        .cpu_mem_req_strb_i      (cpu_mem_req_strb),
        .cpu_mem_rsp_valid_o     (cpu_mem_rsp_valid),
        .cpu_mem_rsp_rdata_o     (cpu_mem_rsp_rdata),
        .dma_mem_req_valid_i     (dma_tcdm_req_valid[0]),
        .dma_mem_req_ready_o     (dma_tcdm_req_ready[0]),
        .dma_mem_req_addr_i      (dma_tcdm_req_addr[0]),
        .dma_mem_req_write_i     (dma_tcdm_req_write[0]),
        .dma_mem_req_wdata_i     (dma_tcdm_req_wdata[0]),
        .dma_mem_req_strb_i      (dma_tcdm_req_strb[0]),
        .dma_mem_rsp_valid_o     (dma_tcdm_rsp_valid[0]),
        .dma_mem_rsp_rdata_o     (dma_tcdm_rsp_rdata[0]),
        .dma_aux_mem_req_valid_i (dma_tcdm_req_valid[3:1]),
        .dma_aux_mem_req_ready_o (dma_tcdm_req_ready[3:1]),
        .dma_aux_mem_req_addr_i  (dma_tcdm_req_addr[3:1]),
        .dma_aux_mem_req_write_i (dma_tcdm_req_write[3:1]),
        .dma_aux_mem_req_wdata_i (dma_tcdm_req_wdata[3:1]),
        .dma_aux_mem_req_strb_i  (dma_tcdm_req_strb[3:1]),
        .dma_aux_mem_rsp_valid_o (dma_tcdm_rsp_valid[3:1]),
        .dma_aux_mem_rsp_rdata_o (dma_tcdm_rsp_rdata[3:1]),
        .dma_desc_valid_o        (dma_desc_valid),
        .dma_desc_ready_i        (dma_desc_ready),
        .dma_desc_dir_o          (dma_desc_dir),
        .dma_desc_dram_addr_o    (dma_desc_dram_addr),
        .dma_desc_tcdm_addr_o    (dma_desc_tcdm_addr),
        .dma_desc_size_o         (dma_desc_size),
        .dma_desc_done_i         (dma_desc_done),
        .subsystem_busy_o        (busy),
        .subsystem_done_irq_o    (done_irq)
    );

    snax_simple_dma #(.TCDM_PORTS(4)) u_dma (
        .clk                  (clk),
        .rst_n                (rst_n),
        .desc_valid_i         (dma_desc_valid),
        .desc_ready_o         (dma_desc_ready),
        .desc_dir_i           (dma_desc_dir),
        .desc_dram_addr_i     (dma_desc_dram_addr),
        .desc_tcdm_addr_i     (dma_desc_tcdm_addr),
        .desc_size_bytes_i    (dma_desc_size),
        .desc_done_o          (dma_desc_done),
        .tcdm_req_valid_o     (dma_tcdm_req_valid),
        .tcdm_req_ready_i     (dma_tcdm_req_ready),
        .tcdm_req_addr_o      (dma_tcdm_req_addr),
        .tcdm_req_write_o     (dma_tcdm_req_write),
        .tcdm_req_wdata_o     (dma_tcdm_req_wdata),
        .tcdm_req_strb_o      (dma_tcdm_req_strb),
        .tcdm_rsp_valid_i     (dma_tcdm_rsp_valid),
        .tcdm_rsp_rdata_i     (dma_tcdm_rsp_rdata),
        .dram_req_valid_o     (dram_req_valid),
        .dram_req_ready_i     (dram_req_ready),
        .dram_req_addr_o      (dram_req_addr),
        .dram_req_write_o     (dram_req_write),
        .dram_req_wdata_o     (dram_req_wdata),
        .dram_req_strb_o      (dram_req_strb),
        .dram_rsp_valid_i     (dram_rsp_valid),
        .dram_rsp_rdata_i     (dram_rsp_rdata),
        .stat_dram_rd_beats_o (stat_rd_beats),
        .stat_dram_wr_beats_o (stat_wr_beats),
        .stat_tcdm_words_o    (),
        .stat_busy_cycles_o   (stat_busy_cycles)
    );

    localparam int DRAM_LATENCY = 4;
    logic [511:0] dram_mem [0:DRAM_WORDS-1];
    logic [511:0] lat_data [0:DRAM_LATENCY-1];
    logic         lat_vld  [0:DRAM_LATENCY-1];
    assign dram_req_ready = 1'b1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < DRAM_LATENCY; i++) begin
                lat_vld[i] <= 1'b0; lat_data[i] <= '0;
            end
            dram_rsp_valid <= 1'b0; dram_rsp_rdata <= '0;
        end else begin
            dram_rsp_valid <= lat_vld[DRAM_LATENCY-1];
            dram_rsp_rdata <= lat_data[DRAM_LATENCY-1];
            for (int i = DRAM_LATENCY-1; i > 0; i--) begin
                lat_vld[i] <= lat_vld[i-1]; lat_data[i] <= lat_data[i-1];
            end
            lat_vld[0] <= 1'b0;
            if (dram_req_valid && dram_req_ready) begin
                if (dram_req_write) dram_mem[dram_req_addr[17:6]] <= dram_req_wdata;
                else begin
                    lat_vld[0]  <= 1'b1;
                    lat_data[0] <= dram_mem[dram_req_addr[17:6]];
                end
            end
        end
    end

    logic [31:0] pre_addr     [0:NUM_PRELOAD-1];
    logic [71:0] pre_data     [0:NUM_PRELOAD-1];
    logic [71:0] k_past_mem   [0:PAST_KV_WORDS-1];
    logic [71:0] v_past_mem   [0:PAST_KV_WORDS-1];
    logic [15:0] g_xnorm      [0:NUM_FP16-1];
    logic [15:0] g_res1       [0:NUM_FP16-1];
    logic [15:0] g_n2         [0:NUM_FP16-1];
    logic [15:0] g_res1_fq    [0:NUM_FP16-1];
    logic [15:0] g_n2_fq      [0:NUM_FP16-1];
    logic [15:0] g_final_fq   [0:NUM_FP16-1];
    logic [31:0] g_final_fp32 [0:NUM_FP16-1];

    function automatic logic [31:0] tcdm_addr(input int word, input int bank);
        tcdm_addr = (32'(word) << 9) | (32'(bank) << 4);
    endfunction

    function automatic real fp16_to_real(input logic [15:0] h);
        logic sign; int e, mf; real v;
        sign = h[15]; e = int'(h[14:10]); mf = int'(h[9:0]);
        if (e == 0 && mf == 0) return 0.0;
        if (e == 31) return sign ? -65504.0 : 65504.0;
        if (e == 0)  v = real'(mf) / 1024.0 * (2.0 ** -14);
        else         v = (1.0 + real'(mf) / 1024.0) * (2.0 ** (e - 15));
        return sign ? -v : v;
    endfunction

    function automatic real fp32_to_real(input logic [31:0] f);
        logic sign; int e, mf; real v;
        sign = f[31]; e = int'(f[30:23]); mf = f[22:0];
        if (e == 0 && mf == 0) return 0.0;
        if (e == 255) return sign ? -1.0e38 : 1.0e38;
        if (e == 0)  v = real'(mf) / 8388608.0 * (2.0 ** -126);
        else         v = (1.0 + real'(mf) / 8388608.0) * (2.0 ** (e - 127));
        return sign ? -v : v;
    endfunction

    task automatic cpu_csr_write(input logic [11:0] a, input logic [31:0] d);
        @(negedge clk);
        csr_req_valid = 1'b1; csr_req_write = 1'b1; csr_req_addr = a; csr_req_data = d;
        @(posedge clk);
        while (!csr_req_ready) @(posedge clk);
        @(negedge clk);
        csr_req_valid = 1'b0; csr_req_write = 1'b0;
    endtask

    task automatic cpu_csr_read(input logic [11:0] a, output logic [31:0] d);
        @(negedge clk);
        csr_req_valid = 1'b1; csr_req_write = 1'b0; csr_req_addr = a;
        @(posedge clk);
        while (!csr_req_ready) @(posedge clk);
        @(negedge clk);
        csr_req_valid = 1'b0;
        while (!csr_rsp_valid) @(posedge clk);
        d = csr_rsp_data;
    endtask

    task automatic cpu_mem_write(input logic [31:0] a, input logic [71:0] d);
        @(negedge clk);
        cpu_mem_req_valid = 1'b1; cpu_mem_req_write = 1'b1;
        cpu_mem_req_addr = a; cpu_mem_req_wdata = d; cpu_mem_req_strb = 9'h1FF;
        @(posedge clk);
        while (!cpu_mem_req_ready) @(posedge clk);
        @(negedge clk);
        cpu_mem_req_valid = 1'b0; cpu_mem_req_write = 1'b0; cpu_mem_req_strb = 9'h000;
    endtask

    task automatic cpu_mem_read(input logic [31:0] a, output logic [71:0] d);
        @(negedge clk);
        cpu_mem_req_valid = 1'b1; cpu_mem_req_write = 1'b0;
        cpu_mem_req_addr = a; cpu_mem_req_strb = 9'h000;
        @(posedge clk);
        while (!cpu_mem_req_ready) @(posedge clk);
        @(negedge clk);
        cpu_mem_req_valid = 1'b0;
        while (!cpu_mem_rsp_valid) @(posedge clk);
        d = cpu_mem_rsp_rdata;
    endtask

    task automatic check(input bit cond, input string name);
        if (cond) checks_pass++;
        else begin checks_fail++; $display("    [FAIL] %s", name); end
    endtask

    task automatic wait_job(input string name);
        bit seen_done, seen_busy;
        int cycle_cnt;
        seen_done = 1'b0; seen_busy = 1'b0; cycle_cnt = 0;
        while (!seen_done && cycle_cnt < JOB_TIMEOUT) begin
            @(posedge clk);
            cycle_cnt = cycle_cnt + 1;
            if (busy) seen_busy = 1'b1;
            if (seen_busy && done_irq && !busy) seen_done = 1'b1;
        end
        check(seen_done, name);
        if (!seen_done)
            $fatal(1, "Timeout on job: %s (seq_state=%0d)", name, dut.u_seq.state_q);
    endtask

    task automatic check_sqnr_rows(input int bank, input logic [15:0] gold [0:NUM_FP16-1],
                                   input string name, input real min_cos);
        real sum_ref2, sum_err2, dot_p, sum_act2, v_a, v_g, sqnr, cos_s;
        logic [71:0] rw;
        sum_ref2 = 0.0; sum_err2 = 0.0; dot_p = 0.0; sum_act2 = 0.0;
        for (int t = 0; t < EVAL_TOKENS; t++)
            for (int w = 0; w < ROW_WORDS; w++) begin
                cpu_mem_read(tcdm_addr(t*ROW_WORDS + w, bank), rw);
                for (int e = 0; e < 4; e++) begin
                    v_a = fp16_to_real(rw[16*e +: 16]);
                    v_g = fp16_to_real(gold[t*D_MODEL + 4*w + e]);
                    sum_ref2 += v_g * v_g; sum_act2 += v_a * v_a;
                    sum_err2 += (v_a - v_g) * (v_a - v_g); dot_p += v_a * v_g;
                end
            end
        sqnr  = (sum_err2 > 0.0) ? (10.0 * $log10(sum_ref2 / sum_err2)) : 999.0;
        cos_s = dot_p / ($sqrt(sum_act2) * $sqrt(sum_ref2));
        $display("    [METRIC] %-18s : SQNR = %0.2f dB, CosineSim = %0.6f", name, sqnr, cos_s);
        check(cos_s >= min_cos, name);
    endtask

    task automatic run_sfu_seq(input logic [2:0] op,
                               input int m_tokens, input int n_tiles, input int k_vecs,
                               input logic [31:0] a_base, input int a_t_16b, input int a_g_16b,
                               input logic [31:0] b_base, input int b_t_16b, input int b_g_16b,
                               input logic [31:0] o_base, input int o_t_16b, input int o_g_16b,
                               input logic [31:0] inv_n, input logic out_fp4 = 1'b0);
        int guard_s;
        cpu_csr_write(A_CTRL,       32'h0000_0001);
        cpu_csr_write(A_GEMM_MT,    32'(m_tokens));
        cpu_csr_write(A_GEMM_NT,    32'(n_tiles));
        cpu_csr_write(A_SFU_CTRL,   32'h0040_0000 | (32'(k_vecs) << 5) | (32'(op) << 2) | (out_fp4 ? 32'h0 : 32'h2) | 32'h1);
        cpu_csr_write(A_SFU_INV_N,  inv_n);
        cpu_csr_write(A_SFU_EPS,    32'h3727C5AC);
        cpu_csr_write(A_RD_BASE,    a_base);
        cpu_csr_write(A_RD_NV,      32'(k_vecs));
        cpu_csr_write(A_RD_LS,      {a_g_16b[15:0], a_t_16b[15:0]});
        cpu_csr_write(A_SFU_B_BASE, b_base);
        cpu_csr_write(A_SFU_B_NV,   32'(k_vecs));
        cpu_csr_write(A_SFU_B_LS,   {b_g_16b[15:0], b_t_16b[15:0]});
        cpu_csr_write(A_WO_MODE,    out_fp4 ? 32'h0 : 32'h1);
        cpu_csr_write(A_WO_BASE,    o_base);
        cpu_csr_write(A_WO_NV,      32'(k_vecs));
        cpu_csr_write(A_WO_LS,      {o_g_16b[15:0], o_t_16b[15:0]});
        cpu_csr_write(A_START,      32'h0000_0003);
        repeat (4) @(posedge clk);
        guard_s = 0;
        while ((dut.sfu_seq_busy || dut.sfu_busy || dut.wo_busy || dut.u_read_fp16.busy_o || dut.rd_b_busy) && guard_s < JOB_TIMEOUT) begin
            @(posedge clk); guard_s++;
        end
        if (guard_s >= JOB_TIMEOUT) $fatal(1, "TIMEOUT waiting for SFU seq");
        cpu_csr_write(A_SFU_CTRL, 32'h0);
    endtask

    task automatic sfu_rms_tokens(input int x_bank, input int gamma_word, input int out_bank);
        run_sfu_seq(3'd2, EVAL_TOKENS, 1, K_PACKETS_QKV,
                    tcdm_addr(0, x_bank), 4 * 32, 0,
                    tcdm_addr(gamma_word, GAMMA_BANK), 0, 0,
                    tcdm_addr(0, out_bank), 4 * 32, 0, INV_D_MODEL_FP32);
    endtask

    task automatic sfu_residual_rows(input int a_bank, input int b_bank, input int out_bank);
        run_sfu_seq(3'd1, EVAL_TOKENS, N_TILES_MODEL, 1,
                    tcdm_addr(0, a_bank),   4 * 32, EVAL_TOKENS * 4 * 32,
                    tcdm_addr(0, b_bank),   4 * 32, 4 * 32,
                    tcdm_addr(0, out_bank), 4 * 32, 4 * 32, 32'h3F800000);
    endtask

    task automatic sfu_silu_chunk(input int chunk);
        run_sfu_seq(3'd4, EVAL_TOKENS, CHUNK_NT, 1,
                    tcdm_addr(0, G_STG_BANK), 4 * 32, EVAL_TOKENS * 4 * 32,
                    tcdm_addr(0, OUT_BANK),   4 * 32, EVAL_TOKENS * 4 * 32,
                    tcdm_addr(chunk * CHUNK_NT, FF1_BANK), K_PACKETS_DOWN * 32, 32,
                    32'h3F800000, 1'b1);
    endtask

    task automatic run_requant(input int rd_bank, input int nvec, input int wo_bank, input string name);
        cpu_csr_write(A_CTRL,     32'h0);
        cpu_csr_write(A_SEQ_CTRL, 32'h3);
        cpu_csr_write(A_RD_BASE,  tcdm_addr(0, rd_bank));
        cpu_csr_write(A_RD_NV,    nvec);
        cpu_csr_write(A_RD_LS,    32'h200);
        cpu_csr_write(A_WO_BASE,  tcdm_addr(0, wo_bank));
        cpu_csr_write(A_START,    32'h0000_0003);
        wait_job(name);
    endtask

    task automatic run_fused_qk_rope_dma(input int n_tiles, input logic [31:0] dram_base,
                                         input logic [31:0] q_base, input int out_bank, input string name);
        int nvec;
        nvec = EVAL_TOKENS * n_tiles;
        cpu_csr_write(A_CTRL,       32'h0);
        cpu_csr_write(A_SEQ_CTRL,   32'h98); // op 00, out_mode=01, enable_dma=1, gemm_decode=1 (bit 7)
        cpu_csr_write(A_GEMM_NT,    n_tiles);
        cpu_csr_write(A_GEMM_MT,    M_TOKENS / 16);
        cpu_csr_write(A_GEMM_KB,    K_PACKETS_QKV);
        cpu_csr_write(A_RING_BASE,  32'h0);
        cpu_csr_write(A_DRAM_KV,    dram_base);
        cpu_csr_write(A_TILE_ST,    K_PACKETS_QKV * 144);
        cpu_csr_write(A_Q_BASE,     q_base);
        cpu_csr_write(A_OUT_BASE,   tcdm_addr(0, out_bank));
        cpu_csr_write(A_SFU_CTRL,   32'h0020_0000 | (32'(nvec) << 5) | (32'd3 << 2) | 32'h1);
        cpu_csr_write(A_SFU_B_BASE, tcdm_addr(0, ROPE_BANK));
        cpu_csr_write(A_SFU_B_NV,   32'(nvec));
        cpu_csr_write(A_SFU_B_LS,   32'h200);
        cpu_csr_write(A_START,      32'h0000_0183);
        wait_job(name);
        while (dut.sfu_busy || dut.rd_b_busy) @(posedge clk);
        cpu_csr_write(A_SFU_CTRL,   32'h0);
    endtask

    task automatic run_gemm_v_dma(input int n_tile, input int v_word, input string name);
        cpu_csr_write(A_CTRL,      32'h0);
        cpu_csr_write(A_SEQ_CTRL,  32'h38);
        cpu_csr_write(A_GEMM_NT,   32'd1);
        cpu_csr_write(A_GEMM_MT,   32'd1);
        cpu_csr_write(A_GEMM_KB,   K_PACKETS_QKV);
        cpu_csr_write(A_RING_BASE, tcdm_addr(1024, 0));
        cpu_csr_write(A_DRAM_KV,   DRAM_WV + 32'(n_tile) * 32'(K_PACKETS_QKV) * 32'd144);
        cpu_csr_write(A_TILE_ST,   K_PACKETS_QKV * 144);
        cpu_csr_write(A_Q_BASE,    tcdm_addr(0, ACT_BANK));
        cpu_csr_write(A_V_BASE,    tcdm_addr(v_word, 0));
        cpu_csr_write(A_V_NB,      32'd1);
        cpu_csr_write(A_START,     32'h0000_0003);
        wait_job(name);
    endtask

    task automatic run_gemm_dma(input int n_tiles, input int k_blks_per_tile, input int k_tiles_m1,
                                input logic [31:0] dram_base, input int tile_stride_bytes,
                                input logic [31:0] q_base, input logic [31:0] out_base, input string name);
        cpu_csr_write(A_CTRL,      32'h0);
        cpu_csr_write(A_SEQ_CTRL,  (32'(k_tiles_m1) << 8) | 32'h88); // enable_dma=1, gemm_decode=1 (bit 7)
        cpu_csr_write(A_GEMM_NT,   n_tiles);
        cpu_csr_write(A_GEMM_MT,   M_TOKENS / 16);
        cpu_csr_write(A_GEMM_KB,   k_blks_per_tile);
        cpu_csr_write(A_RING_BASE, 32'h0);
        cpu_csr_write(A_DRAM_KV,   dram_base);
        cpu_csr_write(A_TILE_ST,   tile_stride_bytes);
        cpu_csr_write(A_Q_BASE,    q_base);
        cpu_csr_write(A_OUT_BASE,  out_base);
        cpu_csr_write(A_START,     32'h0000_0003);
        wait_job(name);
    endtask

    task automatic run_attn_gqa(input string name);
        cpu_csr_write(A_CTRL,      32'h0);
        cpu_csr_write(A_SEQ_CTRL,  (32'(GQA_M1) << 16) | 32'h21); // op 01, non-causal decode, 4D FP4 out
        cpu_csr_write(A_ATTN_H,    H_Q);
        cpu_csr_write(A_ATTN_QT,   Q_TILES);
        cpu_csr_write(A_ATTN_MQ,   M_Q);
        cpu_csr_write(A_ATTN_DV,   DV_TILES);
        cpu_csr_write(A_ATTN_KV,   KV_TILES);
        cpu_csr_write(A_ATTN_KB,   K_PACKETS_QK);
        cpu_csr_write(A_RING_BASE, 32'h0);
        cpu_csr_write(A_Q_BASE,    tcdm_addr(0, QPKT_BANK));
        cpu_csr_write(A_OUT_BASE,  tcdm_addr(0, O_ACT_BANK));
        cpu_csr_write(A_START,     32'h0000_0003);
        wait_job(name);
    endtask

    task automatic run_dma_writeback(input logic [1:0] wb_mode, input logic [31:0] tcdm_base,
                                     input logic [31:0] dram_base, input int size_bytes, input string name);
        cpu_csr_write(A_CTRL,      32'h0);
        cpu_csr_write(A_SEQ_CTRL,  (32'(wb_mode) << 4) | 32'h48); // bit 6 = dma_wb, bit 3 = enable_dma
        cpu_csr_write(A_OUT_BASE,  tcdm_base);
        cpu_csr_write(A_DRAM_KV,   dram_base);
        cpu_csr_write(A_TILE_ST,   size_bytes);
        cpu_csr_write(A_START,     32'h0000_0003);
        wait_job(name);
    endtask

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cyc_total <= 0; cyc_seq <= 0; cyc_sfu <= 0; cyc_mac <= 0; cyc_dma <= 0;
        end else begin
            cyc_total <= cyc_total + 1;
            if (dut.seq_busy)        cyc_seq <= cyc_seq + 1;
            if (dut.sfu_busy)        cyc_sfu <= cyc_sfu + 1;
            if (|dut.lane_valid)     cyc_mac <= cyc_mac + 1;
            if (|dma_tcdm_req_valid) cyc_dma <= cyc_dma + 1;
        end
    end

    initial begin
        logic [31:0] rd32;
        int t0, k_idx, v_idx;
        real sum_ref2, sum_err2, dot_p, sum_rtl2, v_rtl, v_fp32, sqnr, cos_s;
        logic [71:0] rw;

        rst_n = 1'b0;
        csr_req_valid = 1'b0; csr_req_write = 1'b0; csr_req_addr = '0; csr_req_data = '0;
        cpu_mem_req_valid = 1'b0; cpu_mem_req_write = 1'b0;
        cpu_mem_req_addr = '0; cpu_mem_req_wdata = '0; cpu_mem_req_strb = '0;
        checks_pass = 0; checks_fail = 0;

        $display("Loading small LLM Decode DRAM weights (108 KB) & TCDM preload...");
        for (int i = 0; i < DRAM_WORDS; i++) dram_mem[i] = '0;
        $readmemh("sim_llm_decode_small/dram_weights.hex", dram_mem, 0, DRAM_WT_BEATS - 1);
        $readmemh("sim_llm_decode_small/tcdm_preload_addr.hex", pre_addr);
        $readmemh("sim_llm_decode_small/tcdm_preload_data.hex", pre_data);
        $readmemh("sim_llm_decode_small/g_xnorm.hex",      g_xnorm);
        $readmemh("sim_llm_decode_small/g_res1.hex",       g_res1);
        $readmemh("sim_llm_decode_small/g_n2.hex",         g_n2);
        $readmemh("sim_llm_decode_small/g_res1_fq.hex",    g_res1_fq);
        $readmemh("sim_llm_decode_small/g_n2_fq.hex",      g_n2_fq);
        $readmemh("sim_llm_decode_small/g_final_fq.hex",   g_final_fq);
        $readmemh("sim_llm_decode_small/g_final_fp32.hex", g_final_fp32);

        repeat (10) @(posedge clk);
        rst_n = 1'b1;
        repeat (5) @(posedge clk);

        for (int i = 0; i < NUM_PRELOAD; i++) cpu_mem_write(pre_addr[i], pre_data[i]);

        cpu_csr_read(A_RO_ID, rd32);
        check(rd32 == CSR_ID_VALUE, "CSR ID (SNAF)");

        $display("\n========================================================================");
        $display("  SMALL LLM DECODE LAYER END-TO-END (tb_snax_llm_layer_fused scale)");
        $display("  DECODE: M_Q=%0d Q_TILES=%0d N_KV=%0d D_MODEL=%0d D_FF=%0d H=%0d D_K=%0d",
                 M_Q, Q_TILES, N_KV, D_MODEL, D_FF, H_Q, D_K);
        $display("========================================================================\n");

        t0 = cyc_total;
        $display("[1/12] SFU RMSNorm 1...");
        sfu_rms_tokens(X_BANK, GAMMA1_WORD, XN_BANK);
        check_sqnr_rows(XN_BANK, g_xnorm, "RMSNorm 1", 0.999);
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[2/12] HW Requant Xn -> ACT_BANK...");
        run_requant(XN_BANK, NUM_VEC_XN, ACT_BANK, "Requant Xn");
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[3/12] Fused Q (128x128, DMA) + RoPE + LogQ...");
        run_fused_qk_rope_dma(N_TILES_MODEL, DRAM_WQ, tcdm_addr(0, ACT_BANK), OUT_BANK, "Fused Q+RoPE");
        for (int h = 0; h < H_Q; h++)
            for (int kp = 0; kp < K_PACKETS_QK; kp++) begin
                logic [71:0] qp;
                cpu_mem_read(tcdm_addr(h*K_PACKETS_QK + kp, OUT_BANK), qp);
                cpu_mem_write(tcdm_addr(h*K_PACKETS_QK + kp, QPKT_BANK), qp);
            end
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[4/12] Fused K (128x128, DMA) + RoPE + LogQ...");
        run_fused_qk_rope_dma(N_TILES_KV, DRAM_WK, tcdm_addr(0, ACT_BANK), OUT_BANK, "Fused K+RoPE");
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[5/12] Decode GEMM V (M=1, FP16 -> TCDM) + Stage KV-cache (N_KV=%0d)...", N_KV);
        run_gemm_dma(N_TILES_KV, K_PACKETS_QKV, 0, DRAM_WV, K_PACKETS_QKV * 144,
                     tcdm_addr(0, ACT_BANK), tcdm_addr(0, G_STG_BANK), "Decode GEMM V");
        $readmemh("sim_llm_decode_small/k_past.hex", k_past_mem);
        $readmemh("sim_llm_decode_small/v_past.hex", v_past_mem);
        k_idx = 0;
        for (int hkv = 0; hkv < H_KV; hkv++)
            for (int kj = 0; kj < KV_TILES; kj++)
                for (int kp = 0; kp < K_PACKETS_QK; kp++)
                    for (int l = 0; l < 16; l++) begin
                        cpu_mem_write(tcdm_addr(hkv*KV_TILES*K_PACKETS_QK + kj*K_PACKETS_QK + kp, l), k_past_mem[k_idx]);
                        k_idx++;
                    end
        v_idx = 0;
        for (int hkv = 0; hkv < H_KV; hkv++)
            for (int dvi = 0; dvi < DV_TILES; dvi++)
                for (int kj = 0; kj < KV_TILES; kj++)
                    for (int l = 0; l < 16; l++) begin
                        cpu_mem_write(tcdm_addr(V_BASE_BLOCK + hkv*DV_TILES*KV_TILES + dvi*KV_TILES + kj, l), v_past_mem[v_idx]);
                        v_idx++;
                    end
        for (int hkv = 0; hkv < H_KV; hkv++)
            for (int kp = 0; kp < K_PACKETS_QK; kp++) begin
                logic [71:0] kp_pkt;
                cpu_mem_read(tcdm_addr(hkv*K_PACKETS_QK + kp, OUT_BANK), kp_pkt);
                cpu_mem_write(tcdm_addr(hkv*KV_TILES*K_PACKETS_QK + KV_TILES_PAST*K_PACKETS_QK + kp, 15), kp_pkt);
            end
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[6/12] HW Decode FlashAttention (M_Q=1, N_KV=%0d, H=%0d, D=%0d)...", N_KV, H_Q, D_K);
        run_attn_gqa("Decode FlashAttention");
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[7/12] GEMM O (128x128, DMA)...");
        run_gemm_dma(N_TILES_MODEL, K_PACKETS_QKV, 0, DRAM_WO, K_PACKETS_QKV * 144,
                     tcdm_addr(0, O_ACT_BANK), tcdm_addr(0, OUT_BANK), "GEMM O");
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[8/12] SFU Residual 1 + RMSNorm 2...");
        sfu_residual_rows(OUT_BANK, X_BANK, RES1_BANK);
        check_sqnr_rows(RES1_BANK, g_res1_fq, "Residual 1 vs FQ", 0.95);
        check_sqnr_rows(RES1_BANK, g_res1,    "Residual 1 vs FP32", 0.94);
        sfu_rms_tokens(RES1_BANK, GAMMA2_WORD, XN_BANK);
        check_sqnr_rows(XN_BANK, g_n2_fq, "RMSNorm 2 vs FQ", 0.95);
        check_sqnr_rows(XN_BANK, g_n2,    "RMSNorm 2 vs FP32", 0.94);
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[9/12] HW Requant Norm2 -> ACT_BANK...");
        run_requant(XN_BANK, NUM_VEC_XN, ACT_BANK, "Requant Norm2");
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[10/12] FFN Gate & Up (2 x 256x128, DMA) + SiLU*Up (2 chunks)...");
        for (int c = 0; c < N_CHUNKS; c++) begin
            run_gemm_dma(CHUNK_NT, K_PACKETS_QKV, 0,
                         DRAM_WG + 32'(c) * 32'(CHUNK_NT) * 32'(K_PACKETS_QKV) * 32'd144,
                         K_PACKETS_QKV * 144,
                         tcdm_addr(0, ACT_BANK), tcdm_addr(0, G_STG_BANK), "GEMM Gate");
            run_gemm_dma(CHUNK_NT, K_PACKETS_QKV, 0,
                         DRAM_WU + 32'(c) * 32'(CHUNK_NT) * 32'(K_PACKETS_QKV) * 32'd144,
                         K_PACKETS_QKV * 144,
                         tcdm_addr(0, ACT_BANK), tcdm_addr(0, OUT_BANK), "GEMM Up");
            sfu_silu_chunk(c);
        end
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[11/12] GEMM Down (128x256, DMA)...");
        run_gemm_dma(N_TILES_MODEL, K_PACKETS_DOWN, 0,
                     DRAM_WD, K_PACKETS_DOWN * 144,
                     tcdm_addr(0, FF1_BANK), tcdm_addr(0, OUT_BANK), "GEMM Down");
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[12/12] SFU Residual 2 -> Final Output Check...");
        sfu_residual_rows(OUT_BANK, RES1_BANK, FIN_BANK);
        check_sqnr_rows(FIN_BANK, g_final_fq, "Final vs PyFQ", 0.90);

        sum_ref2 = 0.0; sum_err2 = 0.0; dot_p = 0.0; sum_rtl2 = 0.0;
        for (int t = 0; t < EVAL_TOKENS; t++)
            for (int w = 0; w < ROW_WORDS; w++) begin
                cpu_mem_read(tcdm_addr(t*ROW_WORDS + w, FIN_BANK), rw);
                for (int e = 0; e < 4; e++) begin
                    v_rtl  = fp16_to_real(rw[16*e +: 16]);
                    v_fp32 = fp32_to_real(g_final_fp32[t*D_MODEL + 4*w + e]);
                    sum_ref2 += v_fp32 * v_fp32; sum_rtl2 += v_rtl * v_rtl;
                    sum_err2 += (v_rtl - v_fp32) * (v_rtl - v_fp32);
                    dot_p    += v_rtl * v_fp32;
                end
            end
        sqnr  = (sum_err2 > 0.0) ? (10.0 * $log10(sum_ref2 / sum_err2)) : 999.0;
        cos_s = dot_p / ($sqrt(sum_rtl2) * $sqrt(sum_ref2));
        check(cos_s >= 0.88, "Final Layer Output vs FP32");

        run_dma_writeback(2'b11, tcdm_addr(0, FIN_BANK), DRAM_OUT, 8 * 9, "DMA WB Final Out");

        $display("\n========================================================================");
        $display("  SMALL LLM DECODE LAYER RESULT (N_KV=%0d)", N_KV);
        $display("========================================================================");
        $display("  RTL vs FP32       : SQNR = %0.2f dB, CosineSim = %0.6f", sqnr, cos_s);
        $display("  Total cycles      : %0d (seq=%0d, sfu=%0d, mac=%0d)",
                 cyc_total, cyc_seq, cyc_sfu, cyc_mac);
        $display("  DRAM 512b rd beats: %0d (%0d bytes)", stat_rd_beats, stat_rd_beats * 64);
        $display("  DRAM 512b wr beats: %0d (%0d bytes)", stat_wr_beats, stat_wr_beats * 64);
        $display("  Checks            : %0d passed, %0d failed", checks_pass, checks_fail);
        $display("========================================================================\n");
        if (checks_fail == 0) begin
            $display(">>> SMALL LLM DECODE LAYER PASSED (0 errors) <<<");
            $finish;
        end else begin
            $fatal(1, "SMALL LLM DECODE LAYER FAILED (%0d check fails)", checks_fail);
        end
    end
    initial begin
        if (!$test$plusargs("NO_WAVE")) begin
`ifdef FST_TRACE
            $dumpfile("tb_snax_llm_decode_small.fst");
`else
            $dumpfile("tb_snax_llm_decode_small.vcd");
`endif
            if ($test$plusargs("WAVE_DEEP"))
                $dumpvars(0, tb_llm_decode_small);
            else
                $dumpvars(0, dut);
        end
    end




endmodule
