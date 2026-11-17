// =============================================================================
// tb_snax_llm_layer_mid.sv
//
// MID-SIZE complete LLaMA-style prefill layer on snax_llm_top_clean.
//
//   tokens = 32, d_model = 128, d_ff = 256, heads = 4,
//   d_k = d_v = 32 (H * head_dim = 128),
//   M_Q = 8 -> Q_TILES = 4, KV_TILES = 2, DV_TILES = 2,
//   K_PACKETS_QKV = 8, K_PACKETS_QK = 2, K_PACKETS_DOWN = 16.
//
// The flow is the same as the minimal layer test, but every multi-tile /
// multi-head / multi-KV path is exercised:
//   X -> CPU RMSNorm1 -> HW requant -> GEMM Q/K/V
//   -> CPU RoPE -> HW Q/K requant -> packet reorder into attention layout
//   -> HW causal FlashAttention (H=4, KV_TILES=2, DV_TILES=2)
//   -> packet reorder -> GEMM O -> CPU residual + RMSNorm2
//   -> HW requant -> GEMM gate/up in 2 N-chunks
//   -> CPU SiLU*up -> HW requant -> GEMM down -> CPU residual -> final FP16.
//
// V is still the tested fused V path, but issued as one M=1/N=1/V_NB=1 job
// per (head, dv-slice, kv-tile).  This deliberately avoids the known
// multi-batch limitation of the current fused-V sequencer while still
// exercising the transpose + shared log quantizer on every V tile.
// =============================================================================
`timescale 1ns/1ps

module tb_snax_llm_layer_mid;

    // ---------------------------------------------------------------- shape --
    localparam int M_TOKENS   = 32;
    localparam int D_MODEL    = 128;
    localparam int D_FF       = 256;
    localparam int N_HEADS    = 4;
    localparam int D_K        = 32;
    localparam int D_V        = 32;
    localparam int M_Q        = 8;
    localparam int N_Q        = M_TOKENS;
    localparam int N_KV       = M_TOKENS;
    // Tiling convention:
    //   K_TILE   = full reduction chunk held in TCDM (e.g. 128)
    //   K_PACKETS = K_TILE / BLOCK_SIZE(16): number of 16-K packets the
    //               weight streamer walks INSIDE one start.
    // One wt_streamer start covers one 16 x K_TILE weight tile.
    localparam int K_TILE_QKV = D_MODEL;        // 128
    localparam int K_TILE_DOWN = D_FF;          // 256
    localparam int K_TILE_QK  = D_K;            // 32
    localparam int K_TILE_PV  = 16;             // 16 keys in one V packet
    localparam int K_PACKETS_QKV = K_TILE_QKV / 16;   // 8
    localparam int K_PACKETS_DOWN = K_TILE_DOWN / 16; // 16
    localparam int K_PACKETS_QK = K_TILE_QK / 16;     // 2
    localparam int K_PACKETS_PV = K_TILE_PV / 16;     // 1
    localparam int Q_TILES    = N_Q / M_Q;      // 4
    localparam int KV_TILES   = N_KV / 16;      // 2
    localparam int DV_TILES   = D_V / 16;       // 2
    localparam int N_TILES_MODEL = D_MODEL / 16;   // 8
    localparam int N_TILES_O     = (N_HEADS * D_V) / 16; // 8
    localparam int N_TILES_GU    = (2 * D_FF) / 16;      // 32
    localparam int CLK_PERIOD = 10;

    localparam int NUM_FP16   = M_TOKENS * D_MODEL;      // 4096
    localparam int NUM_H      = M_TOKENS * D_FF;         // 8192
    localparam int ROW_WORDS  = D_MODEL / 4;             // 32
    localparam int NUM_VEC_XN = M_TOKENS * K_PACKETS_QKV;       // 256
    localparam int NUM_QVEC   = N_TILES_MODEL * M_TOKENS;// 256
    localparam int NUM_ATTN   = N_HEADS * Q_TILES * DV_TILES * M_Q; // 256
    localparam int NUM_VVEC   = N_HEADS * DV_TILES * KV_TILES * 16; // 256
    localparam int NUM_FFNVEC = M_TOKENS * K_PACKETS_DOWN;        // 512
    localparam int NUM_PRELOAD = 12000;

    // ------------------------------------------------------------ TCDM map --
    localparam int ACT_BANK  = 16;
    localparam int OUT_BANK  = 17;
    localparam int STG_BANK  = 18;
    localparam int PKT_BANK  = 19;
    localparam int O_ACT_BANK = 20;
    localparam int G_STG_BANK = 21;
    localparam int FF1_BANK  = 22;
    localparam int FF2_BANK  = 23;
    localparam int X_BANK    = 24;
    localparam int XN_BANK   = 25;
    localparam int RES1_BANK = 26;
    localparam int FIN_BANK  = 27;
    localparam int GAMMA_BANK = 28;
    localparam int ROPE_BANK = 29;
    localparam int QPKT_BANK = 30;
    localparam int KPKT_BANK = 31;

    localparam int GAMMA1_WORD = 0;
    localparam int GAMMA2_WORD = 32;
    localparam int COS_WORD    = 0;
    localparam int SIN_WORD    = 128;

    localparam int V_BASE_BLOCK = 16;
    localparam int O_ACT_WORD   = 0;

    localparam int W_W_Q  = 0;
    localparam int W_W_K  = 64;
    localparam int W_W_V  = 128;
    localparam int W_W_O  = 192;
    localparam int W_W_GU = 256;
    localparam int W_W_D  = 512;

    localparam logic [31:0] Q_BASE_ADDR = 32'h0000_0100;   // bank 16, word 0
    localparam logic [31:0] OUT_BASE    = 32'h0000_0110;   // bank 17, word 0
    localparam logic [31:0] RING_BASE   = 32'h0000_0000;   // bank 0,  word 0

    // ------------------------------------------------------------ CSR map ----
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
    localparam logic [11:0] A_START     = 12'd53;
    localparam logic [11:0] A_RO_ID     = 12'd61;
    localparam logic [11:0] A_SFU_CTRL  = 12'd46;
    localparam logic [11:0] A_SFU_INV_N = 12'd47;
    localparam logic [11:0] A_SFU_EPS   = 12'd48;
    localparam logic [11:0] A_SFU_B_BASE= 12'd49;
    localparam logic [11:0] A_SFU_B_NV  = 12'd50;
    localparam logic [11:0] A_SFU_B_LS  = 12'd51;
    localparam logic [31:0] CSR_ID_VALUE = 32'h534E_4146;

    localparam int JOB_TIMEOUT     = 400000;
    localparam int HANDSHAKE_GUARD = 500000;

    // --------------------------------------------------------------- DUT -----
    logic clk, rst_n;

    logic        csr_req_valid, csr_req_ready, csr_req_write;
    logic [11:0] csr_req_addr;
    logic [31:0] csr_req_data;
    logic        csr_rsp_valid, csr_rsp_ready;
    logic [31:0] csr_rsp_data;

    logic        cpu_mem_req_valid, cpu_mem_req_ready;
    logic [31:0] cpu_mem_req_addr;
    logic        cpu_mem_req_write;
    logic [71:0] cpu_mem_req_wdata;
    logic [8:0]  cpu_mem_req_strb;
    logic        cpu_mem_rsp_valid;
    logic [71:0] cpu_mem_rsp_rdata;

    logic        dma_mem_req_valid, dma_mem_req_ready;
    logic [31:0] dma_mem_req_addr;
    logic        dma_mem_req_write;
    logic [71:0] dma_mem_req_wdata;
    logic [8:0]  dma_mem_req_strb;
    logic        dma_mem_rsp_valid;
    logic [71:0] dma_mem_rsp_rdata;

    logic        dma_desc_valid, dma_desc_ready;
    logic [1:0]  dma_desc_dir;
    logic [31:0] dma_desc_dram_addr, dma_desc_tcdm_addr, dma_desc_size;
    logic        dma_desc_done;

    logic busy, done_irq;

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    initial begin
        rst_n = 1'b0;
        #(CLK_PERIOD * 5);
        rst_n = 1'b1;
    end

    initial begin
        #(20_000_000);   // 20 ms = 2M cycles
        $display("\n!!! GLOBAL TIMEOUT: simulation did not finish !!!\n");
        $finish;
    end

    snax_llm_top_clean dut (
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
        .dma_mem_req_valid_i     (dma_mem_req_valid),
        .dma_mem_req_ready_o     (dma_mem_req_ready),
        .dma_mem_req_addr_i      (dma_mem_req_addr),
        .dma_mem_req_write_i     (dma_mem_req_write),
        .dma_mem_req_wdata_i     (dma_mem_req_wdata),
        .dma_mem_req_strb_i      (dma_mem_req_strb),
        .dma_mem_rsp_valid_o     (dma_mem_rsp_valid),
        .dma_mem_rsp_rdata_o     (dma_mem_rsp_rdata),
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

    initial begin
        if (!$test$plusargs("NO_WAVE")) begin
            $dumpfile("tb_snax_llm_layer_mid.vcd");
            $dumpvars(0, dut);
        end
    end

    // ------------------------------------------------------------ golden -----
    logic [31:0] pre_addr [0:NUM_PRELOAD-1];
    logic [71:0] pre_data [0:NUM_PRELOAD-1];

    logic [15:0] g_xnorm [0:NUM_FP16-1];
    logic [15:0] g_q     [0:NUM_FP16-1];
    logic [15:0] g_k     [0:NUM_FP16-1];
    logic [15:0] g_res1  [0:NUM_FP16-1];
    logic [15:0] g_n2    [0:NUM_FP16-1];
    logic [15:0] g_o     [0:NUM_FP16-1];
    logic [15:0] g_down  [0:NUM_FP16-1];
    logic [15:0] g_final [0:NUM_FP16-1];
    logic [15:0] g_h     [0:NUM_H-1];
    logic [15:0] g_attn_fp16 [0:NUM_FP16-1];
    localparam int NUM_LQ = N_HEADS * Q_TILES * M_Q;   // 128
    logic [31:0] g_l_q [0:NUM_LQ-1];

    logic [71:0] g_xnpkt  [0:NUM_QVEC-1];
    logic [71:0] g_n2pkt  [0:NUM_QVEC-1];
    logic [71:0] g_qpkt   [0:NUM_QVEC-1];
    logic [71:0] g_kpkt   [0:NUM_QVEC-1];
    logic [71:0] g_vpkt   [0:NUM_VVEC-1];
    logic [71:0] g_attn_pkt [0:NUM_ATTN-1];
    logic [71:0] g_ffnpkt [0:NUM_FFNVEC-1];

    int checks_pass, checks_fail;
    int job_cycles_total;
    int stage_match, stage_mismatch;
    int guard;
    int cyc_total, cyc_core_compute, cyc_core_fp16, cyc_cpu, cyc_dma, cyc_seq;
    int cyc_drain_req, cyc_wo_busy, cyc_gemm_run, cyc_gemm_launch, cyc_wait;
    int cyc_v_busy, cyc_nrm_busy;
    int s_compute, s_drain, s_wo, s_run, s_launch, s_wait, s_v, s_nrm;
    int i, j, w, e, gn, kb, lane, key, dvi, token, qi_v, mi_v, h, c, nt;
    int cycle_cnt;
    int wt_start_cnt, wt_req_cnt;
    wire wt_req_fire = (&dut.wt_m_req_valid) && (&dut.wt_m_req_ready);
    logic [71:0] rw;
    logic [15:0] a16, e16;
    logic [31:0] rd32;

    localparam real CPU_TOL = 0.05;
    real st_max;
    int  st_fail;

    // --------------------------------------------------------------- helpers -
    function automatic logic [31:0] tcdm_addr(input int word, input int bank);
        tcdm_addr = (32'(word) << 9) | (32'(bank) << 4);
    endfunction

    function automatic logic [71:0] pack4(input logic [15:0] a, b, c, d);
        pack4 = {d, c, b, a};
    endfunction

    function automatic real fp16_to_real(input logic [15:0] hv);
        logic s; logic [4:0] ex; logic [9:0] mf; real v;
        s = hv[15]; ex = hv[14:10]; mf = hv[9:0];
        if (ex == 5'd0 && mf == 10'd0) return 0.0;
        if (ex == 5'b11111) return s ? -999999.0 : 999999.0;
        if (ex == 5'd0) v = real'(mf) / 16777216.0;
        else            v = (1.0 + real'(mf) / 1024.0) * $itor(32'd1 << ex) / 32768.0;
        return s ? -v : v;
    endfunction

    function automatic logic [15:0] real_to_fp16(input real v);
        logic s; real a; int expn; logic [4:0] ex; logic [9:0] mant;
        if (v == 0.0) return 16'h0000;
        s = (v < 0.0); a = s ? -v : v;
        expn = 0;
        while (a >= 2.0) begin a = a / 2.0; expn = expn + 1; end
        while (a <  1.0) begin a = a * 2.0; expn = expn - 1; end
        if (expn + 15 <= 0)  return {s, 5'd0, 10'd0};
        if (expn + 15 >= 31) return {s, 5'b11111, 10'd0};
        ex = expn + 15;
        mant = $rtoi((a - 1.0) * 1024.0 + 0.5);
        if (mant >= 1024) begin mant = 10'd0; ex = ex + 5'd1; end
        return {s, ex, mant};
    endfunction

    task automatic check(input bit cond, input string name);
        if (cond) checks_pass = checks_pass + 1;
        else begin
            checks_fail = checks_fail + 1;
            $display("    [FAIL] %s", name);
        end
    endtask

    task automatic tol_reset();
        begin st_max = 0.0; st_fail = 0; end
    endtask

    task automatic tol_add(input real got, input real exp);
        real d, lim;
        begin
            d = got - exp;
            if (d < 0.0) d = -d;
            if (d > st_max) st_max = d;
            lim = CPU_TOL * (1.0 + ((exp < 0.0) ? -exp : exp));
            if (d > lim) st_fail = st_fail + 1;
        end
    endtask

    task automatic tol_report(input string name, input int nvals);
        begin
            // CPU model stages are informational: the DUT is driven by the
            // golden reload unless +CPU_REAL is used, and the hardware packet
            // checks below are the authoritative bit-exact criteria.
            $display("    [CPU-TOL] %s max_abs=%0.5f fail=%0d/%0d",
                     name, st_max, st_fail, nvals);
        end
    endtask

    task automatic fatal(input string msg);
        $display("\n!!! FATAL: %s !!!\n", msg);
        $finish;
    endtask

    // =========================================================================
    // CPU / DMA models
    // =========================================================================
    task automatic cpu_csr_write(input logic [11:0] a, input logic [31:0] d);
        begin
            guard = 0;
            @(negedge clk);
            csr_req_valid = 1'b1; csr_req_write = 1'b1;
            csr_req_addr = a;     csr_req_data = d;
            #1;
            while (!csr_req_ready && guard < HANDSHAKE_GUARD) begin
                @(negedge clk); #1; guard = guard + 1;
            end
            if (!csr_req_ready) fatal("CSR write ready timeout");
            @(posedge clk);
            @(negedge clk);
            csr_req_valid = 1'b0;
        end
    endtask

    task automatic cpu_csr_read(input logic [11:0] a, output logic [31:0] d);
        begin
            guard = 0;
            @(negedge clk);
            csr_req_valid = 1'b1; csr_req_write = 1'b0;
            csr_req_addr = a;     csr_req_data = 32'd0;
            #1;
            while (!csr_req_ready && guard < HANDSHAKE_GUARD) begin
                @(negedge clk); #1; guard = guard + 1;
            end
            if (!csr_req_ready) fatal("CSR read ready timeout");
            @(posedge clk);
            @(negedge clk);
            csr_req_valid = 1'b0;
            d = csr_rsp_data;
        end
    endtask

    task automatic cpu_mem_write(input logic [31:0] a, input logic [71:0] d);
        begin
            guard = 0;
            @(negedge clk);
            cpu_mem_req_valid = 1'b1; cpu_mem_req_write = 1'b1;
            cpu_mem_req_addr = a; cpu_mem_req_wdata = d; cpu_mem_req_strb = 9'h1FF;
            #1;
            while (!cpu_mem_req_ready && guard < HANDSHAKE_GUARD) begin
                @(negedge clk); #1; guard = guard + 1;
            end
            if (!cpu_mem_req_ready) fatal("CPU TCDM write ready timeout");
            @(posedge clk);
            @(negedge clk);
            cpu_mem_req_valid = 1'b0; cpu_mem_req_write = 1'b0; cpu_mem_req_strb = 9'h000;
        end
    endtask

    task automatic cpu_mem_read(input logic [31:0] a, output logic [71:0] d);
        begin
            guard = 0;
            @(negedge clk);
            cpu_mem_req_valid = 1'b1; cpu_mem_req_write = 1'b0;
            cpu_mem_req_addr = a; cpu_mem_req_wdata = '0; cpu_mem_req_strb = 9'h000;
            #1;
            while (!cpu_mem_req_ready && guard < HANDSHAKE_GUARD) begin
                @(negedge clk); #1; guard = guard + 1;
            end
            if (!cpu_mem_req_ready) fatal("CPU TCDM read ready timeout");
            @(posedge clk);
            @(negedge clk);
            cpu_mem_req_valid = 1'b0;
            d = cpu_mem_rsp_rdata;
        end
    endtask

    task automatic dma_write(input logic [31:0] a, input logic [71:0] d);
        begin
            guard = 0;
            @(negedge clk);
            dma_mem_req_valid = 1'b1; dma_mem_req_write = 1'b1;
            dma_mem_req_addr = a; dma_mem_req_wdata = d; dma_mem_req_strb = 9'h1FF;
            #1;
            while (!dma_mem_req_ready && guard < HANDSHAKE_GUARD) begin
                @(negedge clk); #1; guard = guard + 1;
            end
            if (!dma_mem_req_ready) fatal("DMA write ready timeout");
            @(posedge clk);
            @(negedge clk);
            dma_mem_req_valid = 1'b0; dma_mem_req_write = 1'b0; dma_mem_req_strb = 9'h000;
        end
    endtask

    // =========================================================================
    // CPU stage helpers
    // =========================================================================
    task automatic cpu_copy_vec(input int src_word, input int src_bank,
                                input int dst_word, input int dst_bank);
        logic [71:0] wtmp;
        begin
            for (int k = 0; k < 4; k++) begin
                cpu_mem_read(tcdm_addr(src_word + k, src_bank), wtmp);
                cpu_mem_write(tcdm_addr(dst_word + k, dst_bank), wtmp);
            end
        end
    endtask

    // RMSNorm over D_MODEL channels held in ROW_WORDS words.
    task automatic cpu_norm_row(input int x_word, input int x_bank,
                                input int g_word, input int g_bank,
                                input int o_word, input int o_bank);
        logic [71:0] xw [0:ROW_WORDS-1];
        logic [71:0] gw [0:ROW_WORDS-1];
        logic [71:0] ow [0:ROW_WORDS-1];
        logic [15:0] xv [0:D_MODEL-1];
        logic [15:0] gv [0:D_MODEL-1];
        logic [15:0] yv [0:D_MODEL-1];
        real acc, inv;
        begin
            for (int k = 0; k < ROW_WORDS; k++) begin
                cpu_mem_read(tcdm_addr(x_word + k, x_bank), xw[k]);
                cpu_mem_read(tcdm_addr(g_word + k, g_bank), gw[k]);
            end
            acc = 0.0;
            for (int d = 0; d < D_MODEL; d++) begin
                xv[d] = xw[d/4][16*(d%4) +: 16];
                gv[d] = gw[d/4][16*(d%4) +: 16];
                acc = acc + fp16_to_real(xv[d]) * fp16_to_real(xv[d]);
            end
            inv = 1.0 / $sqrt(acc / D_MODEL);
            for (int d = 0; d < D_MODEL; d++)
                yv[d] = real_to_fp16((fp16_to_real(xv[d]) * inv) * fp16_to_real(gv[d]));
            for (int k = 0; k < ROW_WORDS; k++)
                ow[k] = pack4(yv[4*k], yv[4*k+1], yv[4*k+2], yv[4*k+3]);
            for (int k = 0; k < ROW_WORDS; k++)
                cpu_mem_write(tcdm_addr(o_word + k, o_bank), ow[k]);
        end
    endtask

    // RoPE per head: D_K=32, pairs (j, j+D_K/2), two N-tile vectors per head.
    task automatic cpu_rope(input int base_word, input int bank);
        logic [3:0][71:0] cw, sw;
        logic [3:0][71:0] vwa, vwb;
        logic [31:0][15:0] headv, resv;
        real cval, sval, x0, x1;
        int half, vecs_per_head;
        begin
            half = D_K / 2;              // 16
            vecs_per_head = D_K / 16;    // 2
            for (int t = 0; t < M_TOKENS; t++) begin
                for (int k = 0; k < 4; k++) begin
                    cpu_mem_read(tcdm_addr(COS_WORD + t*4 + k, ROPE_BANK), cw[k]);
                    cpu_mem_read(tcdm_addr(SIN_WORD + t*4 + k, ROPE_BANK), sw[k]);
                end
                for (int hh = 0; hh < N_HEADS; hh++) begin
                    for (int k = 0; k < 4; k++) begin
                        cpu_mem_read(tcdm_addr(base_word +
                            ((hh*vecs_per_head + 0)*M_TOKENS + t)*4 + k, bank), vwa[k]);
                        cpu_mem_read(tcdm_addr(base_word +
                            ((hh*vecs_per_head + 1)*M_TOKENS + t)*4 + k, bank), vwb[k]);
                    end
                    for (int k = 0; k < 4; k++) begin
                        for (int el = 0; el < 4; el++) begin
                            headv[4*k + el]      = vwa[k][16*el +: 16];
                            headv[16 + 4*k + el] = vwb[k][16*el +: 16];
                        end
                    end
                    for (int jj = 0; jj < half; jj++) begin
                        cval = fp16_to_real(cw[jj/4][16*(jj%4) +: 16]);
                        sval = fp16_to_real(sw[jj/4][16*(jj%4) +: 16]);
                        x0 = fp16_to_real(headv[jj]);
                        x1 = fp16_to_real(headv[jj+half]);
                        resv[jj]      = real_to_fp16(x0 * cval - x1 * sval);
                        resv[jj+half] = real_to_fp16(x0 * sval + x1 * cval);
                    end
                    for (int k = 0; k < 4; k++) begin
                        vwa[k] = pack4(resv[4*k], resv[4*k+1], resv[4*k+2], resv[4*k+3]);
                        vwb[k] = pack4(resv[16+4*k], resv[16+4*k+1],
                                       resv[16+4*k+2], resv[16+4*k+3]);
                    end
                    for (int k = 0; k < 4; k++) begin
                        cpu_mem_write(tcdm_addr(base_word +
                            ((hh*vecs_per_head + 0)*M_TOKENS + t)*4 + k, bank), vwa[k]);
                        cpu_mem_write(tcdm_addr(base_word +
                            ((hh*vecs_per_head + 1)*M_TOKENS + t)*4 + k, bank), vwb[k]);
                    end
                end
            end
        end
    endtask

    // Q requant packets (linear GEMM order) -> attention Q layout in bank16.
    task automatic cpu_reorder_q();
        logic [71:0] pkt;
        int hh, kbb, qqi, mm, ww, idx;
        begin
            for (int g = 0; g < N_TILES_MODEL; g++) begin
                for (int t = 0; t < M_TOKENS; t++) begin
                    cpu_mem_read(tcdm_addr(g*M_TOKENS + t, QPKT_BANK), pkt);
                    hh = g / 2;
                    kbb = g % 2;
                    qqi = t / M_Q;
                    mm = t % M_Q;
                    ww = hh*Q_TILES*M_Q*K_PACKETS_QK + qqi*M_Q*K_PACKETS_QK + mm*K_PACKETS_QK + kbb;
                    cpu_mem_write(tcdm_addr(ww, ACT_BANK), pkt);
                end
            end
        end
    endtask

    // K requant packets (linear GEMM order) -> attention K layout in banks 0..15.
    task automatic cpu_reorder_k();
        logic [71:0] pkt;
        int hh, kbb, kjj, lnn, ww, idx;
        begin
            for (int g = 0; g < N_TILES_MODEL; g++) begin
                for (int t = 0; t < M_TOKENS; t++) begin
                    cpu_mem_read(tcdm_addr(g*M_TOKENS + t, KPKT_BANK), pkt);
                    hh = g / 2;
                    kbb = g % 2;
                    kjj = t / 16;
                    lnn = t % 16;
                    ww = hh*KV_TILES*K_PACKETS_QK + kjj*K_PACKETS_QK + kbb;
                    cpu_mem_write(tcdm_addr(ww, lnn), pkt);
                end
            end
        end
    endtask

    // Attention output (h,qi,dvi,m) -> token-major O activation in bank20.
    task automatic cpu_reorder_attn_o();
        logic [71:0] pkt;
        int idx, t, kbb;
        begin
            for (int hh = 0; hh < N_HEADS; hh++) begin
                for (int qqi = 0; qqi < Q_TILES; qqi++) begin
                    for (int dvi = 0; dvi < DV_TILES; dvi++) begin
                        for (int mm = 0; mm < M_Q; mm++) begin
                            idx = ((hh*Q_TILES + qqi)*DV_TILES + dvi)*M_Q + mm;
                            t = qqi*M_Q + mm;
                            kbb = hh*DV_TILES + dvi;
                            cpu_mem_read(tcdm_addr(idx, PKT_BANK), pkt);
                            cpu_mem_write(tcdm_addr(t*N_TILES_O + kbb, O_ACT_BANK), pkt);
                        end
                    end
                end
            end
        end
    endtask

    task automatic cpu_copy_gate();
        begin
            for (int g = 0; g < 8; g++)
                for (int t = 0; t < M_TOKENS; t++)
                    cpu_copy_vec((g*M_TOKENS + t)*4, OUT_BANK,
                                 (g*M_TOKENS + t)*4, G_STG_BANK);
        end
    endtask

    // SiLU(gate)*up for one 128-channel chunk; write FF1/FF2 FP16 vectors.
    task automatic cpu_silu_chunk(input int chunk);
        logic [3:0][71:0] gw, uw, hw;
        logic [15:0] gv [0:15];
        logic [15:0] uv [0:15];
        logic [15:0] hv [0:15];
        real g, u;
        int kb_final, vec, wr_word, bank;
        begin
            for (int t = 0; t < M_TOKENS; t++) begin
                for (int n = 0; n < 8; n++) begin
                    for (int k = 0; k < 4; k++) begin
                        cpu_mem_read(tcdm_addr((n*M_TOKENS + t)*4 + k, G_STG_BANK), gw[k]);
                        cpu_mem_read(tcdm_addr((n*M_TOKENS + t)*4 + k, OUT_BANK), uw[k]);
                    end
                    for (int k = 0; k < 4; k++) begin
                        for (int el = 0; el < 4; el++) begin
                            gv[4*k + el] = gw[k][16*el +: 16];
                            uv[4*k + el] = uw[k][16*el +: 16];
                        end
                    end
                    for (int l = 0; l < 16; l++) begin
                        g = fp16_to_real(gv[l]);
                        u = fp16_to_real(uv[l]);
                        hv[l] = real_to_fp16((g / (1.0 + $exp(-g))) * u);
                    end
                    for (int k = 0; k < 4; k++)
                        hw[k] = pack4(hv[4*k], hv[4*k+1], hv[4*k+2], hv[4*k+3]);
                    kb_final = chunk*8 + n;
                    if (t < 16) begin
                        bank = FF1_BANK;
                        vec  = t*K_PACKETS_DOWN + kb_final;
                    end else begin
                        bank = FF2_BANK;
                        vec  = (t-16)*K_PACKETS_DOWN + kb_final;
                    end
                    wr_word = vec*4;
                    for (int k = 0; k < 4; k++)
                        cpu_mem_write(tcdm_addr(wr_word + k, bank), hw[k]);
                end
            end
        end
    endtask

    // -------------------------------------------------------------------------
    // SFU integration helpers (TCDM read A/B + write-out FP16)
    // -------------------------------------------------------------------------
    task automatic write_vec(input int word, input int bank, input real vals [0:15]);
        logic [71:0] w;
        begin
            for (int k = 0; k < 4; k++) begin
                w = pack4(real_to_fp16(vals[4*k]), real_to_fp16(vals[4*k+1]),
                          real_to_fp16(vals[4*k+2]), real_to_fp16(vals[4*k+3]));
                cpu_mem_write(tcdm_addr(word + k, bank), w);
            end
        end
    endtask

    task automatic read_vec(input int word, input int bank, output real vals [0:15]);
        logic [71:0] w;
        begin
            for (int k = 0; k < 4; k++) begin
                cpu_mem_read(tcdm_addr(word + k, bank), w);
                for (int e = 0; e < 4; e++)
                    vals[4*k + e] = fp16_to_real(w[16*e +: 16]);
            end
        end
    endtask

    task automatic wait_sfu_idle(input string tag);
        int guard;
        begin
            repeat (8) @(posedge clk);
            guard = 0;
            while (dut.sfu_busy || dut.wo_busy || dut.u_read_fp16.busy_o ||
                   dut.u_read_fp16_b.busy_o) begin
                @(posedge clk); guard++;
                if (guard > 20000) fatal({tag, " timeout"});
            end
        end
    endtask

    task automatic run_sfu(input logic [2:0] op, input bit rms, input int nvec,
                           input int a_word, input int a_bank,
                           input int b_word, input int b_bank,
                           input int o_word, input int o_bank,
                           input logic [31:0] inv_n);
        logic [31:0] start_nonrms, start_rms1, start_rms2;
        begin
            start_nonrms = 32'h0000_01D1; // commit, wo, rdA, sfu, rdB
            start_rms1   = 32'h0000_00C1; // commit, rdA, sfu
            start_rms2   = 32'h0000_0151; // commit, wo, rdA, rdB
            cpu_csr_write(A_CTRL,       32'h0000_0001);
            cpu_csr_write(A_SFU_CTRL,   (32'(nvec) << 5) | (32'(op) << 2) | 32'h2 | 32'h1);
            cpu_csr_write(A_SFU_INV_N,  inv_n);
            cpu_csr_write(A_SFU_EPS,    32'h3727C5AC);
            cpu_csr_write(A_RD_BASE,    tcdm_addr(a_word, a_bank));
            cpu_csr_write(A_RD_NV,      nvec);
            cpu_csr_write(A_RD_LS,      32'h0000_0200);
            cpu_csr_write(A_SFU_B_BASE, tcdm_addr(b_word, b_bank));
            cpu_csr_write(A_SFU_B_NV,   nvec);
            cpu_csr_write(A_SFU_B_LS,   32'h0000_0200);
            cpu_csr_write(A_WO_MODE,    32'd1);
            cpu_csr_write(A_WO_BASE,    tcdm_addr(o_word, o_bank));
            cpu_csr_write(A_WO_NV,      nvec);
            cpu_csr_write(A_WO_LS,      32'h0000_0200);

            if (rms) begin
                cpu_csr_write(A_START, start_rms1);
                repeat (4) @(posedge clk);
                while (dut.u_read_fp16.busy_o) @(posedge clk);
                cpu_csr_write(A_START, start_rms2);
            end else begin
                cpu_csr_write(A_START, start_nonrms);
            end
            wait_sfu_idle("SFU job");
            cpu_csr_write(A_SFU_CTRL, 32'h0);
        end
    endtask

    task automatic sfu_rms_tokens(input int x_bank, input int gamma_word,
                                  input int out_bank);
        begin
            for (int t = 0; t < M_TOKENS; t++)
                run_sfu(3'd2, 1'b1, K_PACKETS_QKV,
                        t*ROW_WORDS, x_bank, gamma_word, GAMMA_BANK,
                        t*ROW_WORDS, out_bank, 32'h3C000000); // 1/128
        end
    endtask

    task automatic sfu_residual_rows(input int o_word, input int o_bank,
                                     input int x_word, input int x_bank,
                                     input int r_word, input int r_bank);
        int src_word, xw_word, rw_word;
        begin
            for (int t = 0; t < M_TOKENS; t++) begin
                for (int g = 0; g < N_TILES_MODEL; g++) begin
                    src_word = (g*M_TOKENS + t)*4;
                    xw_word  = x_word + t*ROW_WORDS + g*4;
                    rw_word  = r_word + t*ROW_WORDS + g*4;
                    run_sfu(3'd1, 1'b0, 1, src_word, o_bank, xw_word, x_bank,
                            rw_word, r_bank, 32'h3F800000);
                end
            end
        end
    endtask

    task automatic sfu_residual_final(input int d_word, input int d_bank,
                                      input int r_word, input int r_bank,
                                      input int f_word, input int f_bank);
        int src_word, rw_word, fw_word;
        begin
            for (int t = 0; t < M_TOKENS; t++) begin
                for (int g = 0; g < N_TILES_MODEL; g++) begin
                    src_word = (g*M_TOKENS + t)*4;
                    rw_word  = r_word + t*ROW_WORDS + g*4;
                    fw_word  = f_word + t*ROW_WORDS + g*4;
                    run_sfu(3'd1, 1'b0, 1, src_word, d_bank, rw_word, r_bank,
                            fw_word, f_bank, 32'h3F800000);
                end
            end
        end
    endtask

    task automatic sfu_silu_chunk(input int chunk);
        int kb_final, wr_word, bank, vec;
        begin
            for (int t = 0; t < M_TOKENS; t++) begin
                for (int n = 0; n < 8; n++) begin
                    kb_final = chunk*8 + n;
                    if (t < 16) begin bank = FF1_BANK; vec = t*K_PACKETS_DOWN + kb_final; end
                    else begin bank = FF2_BANK; vec = (t-16)*K_PACKETS_DOWN + kb_final; end
                    wr_word = vec*4;
                    run_sfu(3'd4, 1'b0, 1,
                            (n*M_TOKENS + t)*4, G_STG_BANK,
                            (n*M_TOKENS + t)*4, OUT_BANK,
                            wr_word, bank, 32'h3F800000);
                end
            end
        end
    endtask

    // RoPE in the mid-layer layout: each head has two 16-lane vectors, while
    // the SFU rotates 8 adjacent pairs. Repack [j,j+16] into the SFU pair order.
    task automatic sfu_rope(input int base_word, input int bank);
        logic [3:0][71:0] cw, sw, vwa, vwb;
        logic [31:0][15:0] headv, headv2;
        real av[0:15], bv[0:15], ov[0:15];
        int half, vecs_per_head;
        begin
            half = D_K / 2;
            vecs_per_head = D_K / 16;
            for (int t = 0; t < M_TOKENS; t++) begin
                for (int k = 0; k < 4; k++) begin
                    cpu_mem_read(tcdm_addr(COS_WORD + t*4 + k, ROPE_BANK), cw[k]);
                    cpu_mem_read(tcdm_addr(SIN_WORD + t*4 + k, ROPE_BANK), sw[k]);
                end
                for (int hh = 0; hh < N_HEADS; hh++) begin
                    for (int k = 0; k < 4; k++) begin
                        cpu_mem_read(tcdm_addr(base_word +
                            ((hh*vecs_per_head + 0)*M_TOKENS + t)*4 + k, bank), vwa[k]);
                        cpu_mem_read(tcdm_addr(base_word +
                            ((hh*vecs_per_head + 1)*M_TOKENS + t)*4 + k, bank), vwb[k]);
                    end
                    for (int k = 0; k < 4; k++) begin
                        for (int el = 0; el < 4; el++) begin
                            headv[4*k + el]      = vwa[k][16*el +: 16];
                            headv[16 + 4*k + el] = vwb[k][16*el +: 16];
                        end
                    end
                    for (int v = 0; v < 2; v++) begin
                        for (int kk = 0; kk < 8; kk++) begin
                            av[2*kk]   = fp16_to_real(headv[v*8 + kk]);
                            av[2*kk+1] = fp16_to_real(headv[16 + v*8 + kk]);
                            bv[2*kk]   = fp16_to_real(cw[(v*8 + kk)/4][16*((v*8 + kk)%4) +: 16]);
                            bv[2*kk+1] = fp16_to_real(sw[(v*8 + kk)/4][16*((v*8 + kk)%4) +: 16]);
                        end
                        write_vec(0, 22, av);
                        write_vec(0, 23, bv);
                        run_sfu(3'd3, 1'b0, 1, 0, 22, 0, 23, 0, 19, 32'h3F800000);
                        read_vec(0, 19, ov);
                        for (int kk = 0; kk < 8; kk++) begin
                            headv2[v*8 + kk]      = real_to_fp16(ov[2*kk]);
                            headv2[16 + v*8 + kk] = real_to_fp16(ov[2*kk+1]);
                        end
                    end
                    for (int k = 0; k < 4; k++) begin
                        vwa[k] = pack4(headv2[4*k], headv2[4*k+1], headv2[4*k+2], headv2[4*k+3]);
                        vwb[k] = pack4(headv2[16+4*k], headv2[16+4*k+1],
                                       headv2[16+4*k+2], headv2[16+4*k+3]);
                    end
                    for (int k = 0; k < 4; k++) begin
                        cpu_mem_write(tcdm_addr(base_word +
                            ((hh*vecs_per_head + 0)*M_TOKENS + t)*4 + k, bank), vwa[k]);
                        cpu_mem_write(tcdm_addr(base_word +
                            ((hh*vecs_per_head + 1)*M_TOKENS + t)*4 + k, bank), vwb[k]);
                    end
                end
            end
        end
    endtask

    // res1 = O + X for the whole token row.
    task automatic cpu_residual_rows(input int o_word, input int o_bank,
                                     input int x_word, input int x_bank,
                                     input int r_word, input int r_bank);
        logic [3:0][71:0] ow, xw, rwv;
        logic [3:0][15:0] lv;
        real a, b;
        int src_word, xw_word, rw_word;
        begin
            for (int t = 0; t < M_TOKENS; t++) begin
                for (int g = 0; g < N_TILES_MODEL; g++) begin
                    src_word = (g*M_TOKENS + t)*4;
                    xw_word  = x_word + t*ROW_WORDS + g*4;
                    rw_word  = r_word + t*ROW_WORDS + g*4;
                    for (int k = 0; k < 4; k++) begin
                        cpu_mem_read(tcdm_addr(o_word + src_word + k, o_bank), ow[k]);
                        cpu_mem_read(tcdm_addr(xw_word + k, x_bank), xw[k]);
                    end
                    for (int k = 0; k < 4; k++) begin
                        for (int el = 0; el < 4; el++) begin
                            a = fp16_to_real(ow[k][16*el +: 16]);
                            b = fp16_to_real(xw[k][16*el +: 16]);
                            lv[el] = real_to_fp16(a + b);
                        end
                        rwv[k] = pack4(lv[0], lv[1], lv[2], lv[3]);
                        cpu_mem_write(tcdm_addr(rw_word + k, r_bank), rwv[k]);
                    end
                end
            end
        end
    endtask

    task automatic cpu_residual_final(input int d_word, input int d_bank,
                                      input int r_word, input int r_bank,
                                      input int f_word, input int f_bank);
        logic [3:0][71:0] dw, rwv, fw;
        logic [3:0][15:0] lv;
        real a, b;
        int src_word, rw_word, fw_word;
        begin
            for (int t = 0; t < M_TOKENS; t++) begin
                for (int g = 0; g < N_TILES_MODEL; g++) begin
                    src_word = (g*M_TOKENS + t)*4;
                    rw_word  = r_word + t*ROW_WORDS + g*4;
                    fw_word  = f_word + t*ROW_WORDS + g*4;
                    for (int k = 0; k < 4; k++) begin
                        cpu_mem_read(tcdm_addr(d_word + src_word + k, d_bank), dw[k]);
                        cpu_mem_read(tcdm_addr(rw_word + k, r_bank), rwv[k]);
                    end
                    for (int k = 0; k < 4; k++) begin
                        for (int el = 0; el < 4; el++) begin
                            a = fp16_to_real(dw[k][16*el +: 16]);
                            b = fp16_to_real(rwv[k][16*el +: 16]);
                            lv[el] = real_to_fp16(a + b);
                        end
                        fw[k] = pack4(lv[0], lv[1], lv[2], lv[3]);
                        cpu_mem_write(tcdm_addr(fw_word + k, f_bank), fw[k]);
                    end
                end
            end
        end
    endtask

    // =========================================================================
    // Check / reload helpers
    // =========================================================================
    task automatic check_rows(input int bank,
                              input logic [15:0] gold [0:NUM_FP16-1],
                              input string name);
        logic [71:0] tmp;
        begin
            stage_match = 0; stage_mismatch = 0;
            tol_reset();
            for (int t = 0; t < M_TOKENS; t++)
                for (int ww = 0; ww < ROW_WORDS; ww++) begin
                    cpu_mem_read(tcdm_addr(t*ROW_WORDS + ww, bank), tmp);
                    for (int ee = 0; ee < 4; ee++) begin
                        a16 = tmp[16*ee +: 16];
                        e16 = gold[t*D_MODEL + 4*ww + ee];
                        tol_add(fp16_to_real(a16), fp16_to_real(e16));
                        if (a16 === e16) stage_match++;
                        else stage_match = stage_match; // keep count
                    end
                end
            $display("    %s: %0d/%0d exact-ish", name, stage_match, NUM_FP16);
            tol_report(name, NUM_FP16);
        end
    endtask

    task automatic reload_rows(input int bank,
                               input logic [15:0] gold [0:NUM_FP16-1]);
        begin
            for (int t = 0; t < M_TOKENS; t++)
                for (int ww = 0; ww < ROW_WORDS; ww++)
                    cpu_mem_write(tcdm_addr(t*ROW_WORDS + ww, bank),
                                  pack4(gold[t*D_MODEL + 4*ww + 0],
                                        gold[t*D_MODEL + 4*ww + 1],
                                        gold[t*D_MODEL + 4*ww + 2],
                                        gold[t*D_MODEL + 4*ww + 3]));
        end
    endtask

    task automatic check_gemm_out(input int bank, input int n_tiles,
                                  input logic [15:0] gold [0:NUM_FP16-1],
                                  input string name);
        logic [71:0] tmp;
        int idx;
        begin
            stage_match = 0; stage_mismatch = 0;
            tol_reset();
            for (int g = 0; g < n_tiles; g++)
                for (int t = 0; t < M_TOKENS; t++)
                    for (int ww = 0; ww < 4; ww++) begin
                        idx = g*M_TOKENS + t;
                        cpu_mem_read(tcdm_addr(idx*4 + ww, bank), tmp);
                        for (int ee = 0; ee < 4; ee++) begin
                            a16 = tmp[16*ee +: 16];
                            e16 = gold[idx*16 + 4*ww + ee];
                            if (a16 === e16) stage_match++;
                        end
                    end
            $display("    %s exact: %0d/%0d", name, stage_match, n_tiles*M_TOKENS*16);
            check(stage_match == n_tiles*M_TOKENS*16, name);
        end
    endtask

    task automatic check_gemm_out_tol(input int bank, input int n_tiles,
                                      input logic [15:0] gold [0:NUM_FP16-1],
                                      input string name);
        logic [71:0] tmp;
        int idx;
        begin
            stage_match = 0;
            tol_reset();
            for (int g = 0; g < n_tiles; g++)
                for (int t = 0; t < M_TOKENS; t++)
                    for (int ww = 0; ww < 4; ww++) begin
                        idx = g*M_TOKENS + t;
                        cpu_mem_read(tcdm_addr(idx*4 + ww, bank), tmp);
                        for (int ee = 0; ee < 4; ee++) begin
                            a16 = tmp[16*ee +: 16];
                            e16 = gold[idx*16 + 4*ww + ee];
                            tol_add(fp16_to_real(a16), fp16_to_real(e16));
                            if (a16 === e16) stage_match++;
                        end
                    end
            $display("    %s exact: %0d/%0d", name, stage_match, n_tiles*M_TOKENS*16);
            tol_report(name, n_tiles*M_TOKENS*16);
        end
    endtask

    task automatic reload_gemm_out(input int bank, input int n_tiles,
                                   input logic [15:0] gold [0:NUM_FP16-1]);
        int idx;
        begin
            for (int g = 0; g < n_tiles; g++)
                for (int t = 0; t < M_TOKENS; t++) begin
                    idx = g*M_TOKENS + t;
                    for (int ww = 0; ww < 4; ww++)
                        cpu_mem_write(tcdm_addr(idx*4 + ww, bank),
                                      pack4(gold[idx*16 + 4*ww + 0],
                                            gold[idx*16 + 4*ww + 1],
                                            gold[idx*16 + 4*ww + 2],
                                            gold[idx*16 + 4*ww + 3]));
                end
        end
    endtask

    task automatic check_h(input logic [15:0] gold [0:NUM_H-1]);
        logic [71:0] tmp;
        int bank, vv, ww;
        begin
            stage_match = 0; stage_mismatch = 0;
            tol_reset();
            for (int t = 0; t < M_TOKENS; t++)
                for (int kbb = 0; kbb < K_PACKETS_DOWN; kbb++)
                    for (int ln = 0; ln < 16; ln++) begin
                        bank = (t < 16) ? FF1_BANK : FF2_BANK;
                        vv = (t % 16)*K_PACKETS_DOWN + kbb;
                        ww = vv*4 + ln/4;
                        cpu_mem_read(tcdm_addr(ww, bank), tmp);
                        a16 = tmp[16*(ln%4) +: 16];
                        e16 = gold[t*D_FF + kbb*16 + ln];
                        tol_add(fp16_to_real(a16), fp16_to_real(e16));
                        if (a16 === e16) stage_match++;
                    end
            $display("    SiLU*up: %0d/%0d exact", stage_match, NUM_H);
            tol_report("CPU tolerance: SiLU*up", NUM_H);
        end
    endtask

    task automatic reload_h(input logic [15:0] gold [0:NUM_H-1]);
        int bank, vv;
        begin
            for (int t = 0; t < M_TOKENS; t++)
                for (int kbb = 0; kbb < K_PACKETS_DOWN; kbb++) begin
                    bank = (t < 16) ? FF1_BANK : FF2_BANK;
                    vv = (t % 16)*K_PACKETS_DOWN + kbb;
                    for (int ww = 0; ww < 4; ww++)
                        cpu_mem_write(tcdm_addr(vv*4 + ww, bank),
                                      pack4(gold[t*D_FF + kbb*16 + 4*ww + 0],
                                            gold[t*D_FF + kbb*16 + 4*ww + 1],
                                            gold[t*D_FF + kbb*16 + 4*ww + 2],
                                            gold[t*D_FF + kbb*16 + 4*ww + 3]));
                end
        end
    endtask

    task automatic check_packets(input int bank, input int base_word,
                                 input int count,
                                 input logic [71:0] gold [0:NUM_QVEC-1],
                                 input string name);
        int bad;
        begin
            bad = 0;
            for (int ii = 0; ii < count; ii++) begin
                cpu_mem_read(tcdm_addr(base_word + ii, bank), rw);
                if (rw !== gold[ii]) begin
                    if (bad < 6 || name == "attention")
                        $display("    pkt mismatch %s i=%0d act=%018X exp=%018X",
                                 name, ii, rw, gold[ii]);
                    bad++;
                end
            end
            $display("    %s packets: %0d/%0d match", name, count-bad, count);
            check(bad == 0, name);
        end
    endtask

    task automatic check_packets512(input int bank, input int base_word,
                                    input logic [71:0] gold [0:NUM_FFNVEC-1],
                                    input string name);
        int bad;
        begin
            bad = 0;
            for (int ii = 0; ii < NUM_FFNVEC; ii++) begin
                cpu_mem_read(tcdm_addr(base_word + ii, bank), rw);
                if (rw !== gold[ii]) begin
                    if (bad < 6) $display("    pkt mismatch %s i=%0d act=%018X exp=%018X",
                                          name, ii, rw, gold[ii]);
                    bad++;
                end
            end
            $display("    %s packets: %0d/%0d match", name, NUM_FFNVEC-bad, NUM_FFNVEC);
            check(bad == 0, name);
        end
    endtask

    task automatic check_v_packets();
        int idx, word;
        int bad;
        begin
            bad = 0;
            for (int hh = 0; hh < N_HEADS; hh++)
                for (int dv = 0; dv < DV_TILES; dv++)
                    for (int kj = 0; kj < KV_TILES; kj++)
                        for (int ln = 0; ln < 16; ln++) begin
                            idx = ((hh*DV_TILES + dv)*KV_TILES + kj)*16 + ln;
                            word = V_BASE_BLOCK + hh*DV_TILES*KV_TILES +
                                   dv*KV_TILES + kj;
                            cpu_mem_read(tcdm_addr(word, ln), rw);
                            if (rw !== g_vpkt[idx]) begin
                                if (bad < 6)
                                    $display("    V pkt mismatch h=%0d dvi=%0d kj=%0d lane=%0d act=%018X exp=%018X",
                                             hh, dv, kj, ln, rw, g_vpkt[idx]);
                                bad++;
                            end
                        end
            $display("    V packets: %0d/%0d match", NUM_VVEC-bad, NUM_VVEC);
            check(bad == 0, "V packets bit-exact (fused transpose)");
        end
    endtask

    // =========================================================================
    // Job launchers
    // =========================================================================
    bit seen_done, seen_busy;
    bit attn_active;
    int attn_out_cnt;

    always @(posedge clk) begin
        if ($test$plusargs("L_DBG") &&
            attn_active && dut.core_fp16_valid && dut.core_fp16_ready) begin
            if (attn_out_cnt == 23)
                $display("    [LDBG] i=23 rtl_l=%0d gen_l=%0d q=%0d",
                         $signed(dut.u_core.u_l_accumulator.l_q[dut.u_core.drain_query_idx]),
                         $signed(g_l_q[15]), dut.u_core.drain_query_idx);
            if (attn_out_cnt == 65)
                $display("    [LDBG] i=65 rtl_l=%0d gen_l=%0d q=%0d",
                         $signed(dut.u_core.u_l_accumulator.l_q[dut.u_core.drain_query_idx]),
                         $signed(g_l_q[33]), dut.u_core.drain_query_idx);
            attn_out_cnt = attn_out_cnt + 1;
        end
        if ($test$plusargs("P_DBG")) begin
            if (dut.u_core.sm_in_valid &&
                dut.u_seq.cnt_h_q == 0 && dut.u_seq.cnt_qi_q == 1 &&
                dut.u_seq.cnt_dvi_q == 0 && dut.u_core.sm_query_idx == 7) begin
                $write("    [SDBG] h0 qi1 dvi0 m7 kj=%0d scores=", dut.u_seq.cnt_kj_q);
                for (int l = 0; l < 16; l++)
                    $write("%0d ", $signed(dut.u_core.score_for_softmax[l]));
                $write("\n");
            end
            if (dut.u_core.sm_out_valid &&
                dut.u_seq.cnt_h_q == 0 && dut.u_seq.cnt_qi_q == 1 &&
                dut.u_seq.cnt_dvi_q == 0 && dut.u_core.sm_query_idx_d3 == 7) begin
                $display("    [PDBG] h0 qi1 dvi0 m7 kj=%0d pblk=%0d csum=%0d lc=%0d rescv=%0b rq=%0d rr=%0d codes=%016X",
                         dut.u_seq.cnt_kj_q,
                         $signed(dut.u_core.sm_p_blk_out),
                         dut.u_core.u_softmax_requantizer.csum_out,
                         $signed(dut.u_core.sm_l_contrib_out),
                         dut.u_core.sm_resc_valid,
                         $signed(dut.u_core.sm_resc_q),
                         dut.u_core.sm_resc_r,
                         dut.u_core.sm_code_out);
            end
        end
    end

    // One weight-streamer start must cover exactly one 16 x K_TILE tile.
    // Each K_TILE is delivered as K_PACKETS 16-element TCDM requests.
    always @(posedge clk) begin
        if (rst_n) begin
            if (dut.wt_start)  wt_start_cnt = wt_start_cnt + 1;
            if (wt_req_fire)   wt_req_cnt   = wt_req_cnt + 1;
        end
    end

    task automatic wait_job(input string name);
        begin
            seen_done = 1'b0; seen_busy = 1'b0; cycle_cnt = 0;
            while (!seen_done && cycle_cnt < JOB_TIMEOUT) begin
                @(posedge clk);
                cycle_cnt = cycle_cnt + 1;
                if (busy) seen_busy = 1'b1;
                if (seen_busy && done_irq && !busy) seen_done = 1'b1;
            end
            job_cycles_total = job_cycles_total + cycle_cnt;
            check(seen_done, name);
            $display("    [job] %-24s cycles=%0d", name, cycle_cnt);
            if (!seen_done)
                $display("    [dbg] seq_state=%0d wo_state=%0d wo_vec=%0d fifo=%0d rd_busy=%0b act_busy=%0b wt_busy=%0b v_busy=%0b busy=%0b",
                         dut.u_seq.state_q, dut.u_write_out.state_q,
                         dut.u_write_out.vec_cnt_q, dut.u_write_out.in_fifo_level,
                         dut.rd_busy, dut.act_busy, dut.wt_busy, dut.v_busy, busy);
        end
    endtask

    task automatic snap_mix();
        begin
            s_compute = cyc_core_compute; s_drain = cyc_drain_req;
            s_wo = cyc_wo_busy; s_run = cyc_gemm_run; s_launch = cyc_gemm_launch;
            s_wait = cyc_wait; s_v = cyc_v_busy; s_nrm = cyc_nrm_busy;
        end
    endtask

    task automatic print_mix();
        begin
            $display("    [mix] compute=%0d drain_req=%0d wo_busy=%0d launch=%0d run=%0d wait=%0d v_busy=%0d nrm=%0d",
                     cyc_core_compute - s_compute, cyc_drain_req - s_drain,
                     cyc_wo_busy - s_wo, cyc_gemm_launch - s_launch,
                     cyc_gemm_run - s_run, cyc_wait - s_wait,
                     cyc_v_busy - s_v, cyc_nrm_busy - s_nrm);
        end
    endtask

    task automatic run_gemm(input int n_tiles, input int m_tiles,
                            input int k_blks, input int ring_word,
                            input logic [31:0] q_base, input string name);
        int wt_s0, wt_q0;
        begin
            cpu_csr_write(A_CTRL,      32'h0);
            cpu_csr_write(A_SEQ_CTRL,  32'h0);          // op 00 GEMM, DMA off
            cpu_csr_write(A_GEMM_NT,   n_tiles);
            cpu_csr_write(A_GEMM_MT,   m_tiles);
            // A_GEMM_KB is K_PACKETS (K_TILE/16), not a K-tile count.
            cpu_csr_write(A_GEMM_KB,   k_blks);
            cpu_csr_write(A_RING_BASE, tcdm_addr(ring_word, 0));
            cpu_csr_write(A_Q_BASE,    q_base);
            cpu_csr_write(A_OUT_BASE,  OUT_BASE);
            snap_mix();
            wt_s0 = wt_start_cnt;
            wt_q0 = wt_req_cnt;
            cpu_csr_write(A_START,     32'h0000_0003);
            wait_job(name);
            check((wt_start_cnt - wt_s0) == (m_tiles * n_tiles),
                  "wt starts == M_TILES*N_TILES");
            check((wt_req_cnt - wt_q0) == (m_tiles * n_tiles * k_blks),
                  "wt K-packets == starts*K_PACKETS");
            $display("    [wt] %-24s starts=%0d reqs=%0d (tile=16x%0d, K_PACKETS=%0d)",
                     name, wt_start_cnt-wt_s0, wt_req_cnt-wt_q0,
                     k_blks*16, k_blks);
            print_mix();
        end
    endtask

    task automatic run_gemm_v(input int n_tile, input int act_word,
                              input int v_word, input string name);
        int wt_s0, wt_q0;
        begin
            cpu_csr_write(A_CTRL,      32'h0);
            cpu_csr_write(A_SEQ_CTRL,  32'h30);         // op 00, fused V
            cpu_csr_write(A_GEMM_NT,   32'd1);
            cpu_csr_write(A_GEMM_MT,   32'd1);
            cpu_csr_write(A_GEMM_KB,   K_PACKETS_QKV);
            cpu_csr_write(A_RING_BASE, tcdm_addr(W_W_V + n_tile*K_PACKETS_QKV, 0));
            cpu_csr_write(A_Q_BASE,    tcdm_addr(act_word, ACT_BANK));
            cpu_csr_write(A_V_BASE,    tcdm_addr(v_word, 0));
            cpu_csr_write(A_V_NB,      32'd1);
            snap_mix();
            wt_s0 = wt_start_cnt;
            wt_q0 = wt_req_cnt;
            cpu_csr_write(A_START,     32'h0000_0003);
            wait_job(name);
            check((wt_start_cnt - wt_s0) == 1,
                  "V wt starts == 1");
            check((wt_req_cnt - wt_q0) == K_PACKETS_QKV,
                  "V wt K-packets == K_PACKETS_QKV");
            $display("    [wt] %-24s starts=%0d reqs=%0d (tile=16x128, K_PACKETS=8)",
                     name, wt_start_cnt-wt_s0, wt_req_cnt-wt_q0);
            print_mix();
        end
    endtask

    task automatic run_attn(input string name);
        begin
            cpu_csr_write(A_CTRL,      32'h0);
            cpu_csr_write(A_SEQ_CTRL,  32'h15);         // op 01, causal, FP4 out
            cpu_csr_write(A_ATTN_H,    N_HEADS);
            cpu_csr_write(A_ATTN_QT,   Q_TILES);
            cpu_csr_write(A_ATTN_MQ,   M_Q);
            cpu_csr_write(A_ATTN_DV,   DV_TILES);
            cpu_csr_write(A_ATTN_KV,   KV_TILES);
            cpu_csr_write(A_ATTN_KB,   K_PACKETS_QK);
            cpu_csr_write(A_RING_BASE, RING_BASE);
            cpu_csr_write(A_Q_BASE,    Q_BASE_ADDR);
            // Attention output must NOT alias the Q packet bank: the sequencer
            // re-reads Q for every dvi slice, so the dvi=0 output would clobber
            // Q[0..M_Q/2-1] before the dvi=1 QK pass.  Use the free packet bank.
            cpu_csr_write(A_OUT_BASE,  tcdm_addr(0, PKT_BANK));
            cpu_csr_write(A_DRAM_KV,   32'h0);
            cpu_csr_write(A_TILE_ST,   32'h200);
            snap_mix();
            attn_active = 1'b1;
            attn_out_cnt = 0;
            cpu_csr_write(A_START,     32'h0000_0003);
            wait_job(name);
            attn_active = 1'b0;
            print_mix();
        end
    endtask

    // Debug-only attention run: FP16 output (out_mode=0) to compare the
    // attention normalizer/l directly against the Python golden.
    task automatic run_attn_fp16_debug(input string name);
        begin
            cpu_csr_write(A_CTRL,      32'h0);
            cpu_csr_write(A_SEQ_CTRL,  32'h05);         // op 01, causal, FP16 out
            cpu_csr_write(A_ATTN_H,    N_HEADS);
            cpu_csr_write(A_ATTN_QT,   Q_TILES);
            cpu_csr_write(A_ATTN_MQ,   M_Q);
            cpu_csr_write(A_ATTN_DV,   DV_TILES);
            cpu_csr_write(A_ATTN_KV,   KV_TILES);
            cpu_csr_write(A_ATTN_KB,   K_PACKETS_QK);
            cpu_csr_write(A_RING_BASE, RING_BASE);
            cpu_csr_write(A_Q_BASE,    Q_BASE_ADDR);
            cpu_csr_write(A_OUT_BASE,  tcdm_addr(0, PKT_BANK));
            cpu_csr_write(A_DRAM_KV,   32'h0);
            cpu_csr_write(A_TILE_ST,   32'h200);
            snap_mix();
            cpu_csr_write(A_START,     32'h0000_0003);
            wait_job(name);
            print_mix();
        end
    endtask

    task automatic run_requant(input int rd_word, input int rd_bank,
                               input int nvec, input int wo_word,
                               input int wo_bank, input string name);
        begin
            cpu_csr_write(A_CTRL,     32'h0);
            cpu_csr_write(A_SEQ_CTRL, 32'h3);           // op 11 readback-requant
            cpu_csr_write(A_RD_BASE,  tcdm_addr(rd_word, rd_bank));
            cpu_csr_write(A_RD_NV,    nvec);
            cpu_csr_write(A_RD_LS,    32'h200);
            cpu_csr_write(A_WO_BASE,  tcdm_addr(wo_word, wo_bank));
            snap_mix();
            cpu_csr_write(A_START,    32'h0000_0003);
            wait_job(name);
            print_mix();
        end
    endtask

    // Simple cycle counters.
    always @(posedge clk) begin
        if (rst_n) begin
            cyc_total        = cyc_total + 1;
            if (|dut.lane_valid)       cyc_core_compute = cyc_core_compute + 1;
            if (dut.core_fp16_valid)   cyc_core_fp16    = cyc_core_fp16 + 1;
            if (cpu_mem_req_valid || csr_req_valid) cyc_cpu = cyc_cpu + 1;
            if (dma_mem_req_valid)     cyc_dma          = cyc_dma + 1;
            if (dut.seq_busy)          cyc_seq          = cyc_seq + 1;
            if (dut.tile_drain_req)    cyc_drain_req    = cyc_drain_req + 1;
            if (dut.wo_busy)           cyc_wo_busy      = cyc_wo_busy + 1;
            if (dut.u_seq.state_q == 5'd11) cyc_gemm_launch = cyc_gemm_launch + 1;
            if (dut.u_seq.state_q == 5'd12) cyc_gemm_run    = cyc_gemm_run + 1;
            if (dut.u_seq.state_q == 5'd19) cyc_wait        = cyc_wait + 1;
            if (dut.v_busy)            cyc_v_busy       = cyc_v_busy + 1;
            if (dut.u_seq.nrm_busy)    cyc_nrm_busy     = cyc_nrm_busy + 1;
        end
    end

    // =========================================================================
    // Main sequence
    // =========================================================================
    initial begin
        csr_req_valid = 1'b0; csr_req_write = 1'b0; csr_req_addr = '0; csr_req_data = '0;
        csr_rsp_ready = 1'b1;
        cpu_mem_req_valid = 1'b0; cpu_mem_req_write = 1'b0; cpu_mem_req_addr = '0;
        cpu_mem_req_wdata = '0; cpu_mem_req_strb = '0;
        dma_mem_req_valid = 1'b0; dma_mem_req_write = 1'b0; dma_mem_req_addr = '0;
        dma_mem_req_wdata = '0; dma_mem_req_strb = '0;
        dma_desc_ready = 1'b1; dma_desc_done = 1'b0;

        checks_pass = 0; checks_fail = 0; job_cycles_total = 0;
        cyc_total = 0; cyc_core_compute = 0; cyc_core_fp16 = 0;
        cyc_cpu = 0; cyc_dma = 0; cyc_seq = 0;
        cyc_drain_req = 0; cyc_wo_busy = 0; cyc_gemm_run = 0;
        cyc_gemm_launch = 0; cyc_wait = 0; cyc_v_busy = 0; cyc_nrm_busy = 0;
        wt_start_cnt = 0; wt_req_cnt = 0;

        $readmemh("sim_snax_llm_layer_mid/hex/tcdm_preload_addr.hex", pre_addr);
        $readmemh("sim_snax_llm_layer_mid/hex/tcdm_preload_data.hex", pre_data);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_xnorm_fp16.hex", g_xnorm);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_q_fp16.hex", g_q);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_k_fp16.hex", g_k);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_res1_fp16.hex", g_res1);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_norm2_fp16.hex", g_n2);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_o_fp16.hex", g_o);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_down_fp16.hex", g_down);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_h_fp16.hex", g_h);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_attn_fp16.hex", g_attn_fp16);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_final_fp16.hex", g_final);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_q_packets.hex", g_qpkt);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_k_packets.hex", g_kpkt);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_v_packets.hex", g_vpkt);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_attn_packets.hex", g_attn_pkt);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_xn_packets.hex", g_xnpkt);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_norm2_packets.hex", g_n2pkt);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_ffn_packets.hex", g_ffnpkt);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_l_q16.hex", g_l_q);

        wait (rst_n == 1'b1);
        repeat (5) @(posedge clk);

        $display("\n========================================================================");
        $display("  SNAX FP4 LLM - MID-SIZE FULL PREFILL LAYER");
        $display("  tokens=%0d d_model=%0d d_ff=%0d heads=%0d d_k=%0d d_v=%0d n_kv=%0d",
                 M_TOKENS, D_MODEL, D_FF, N_HEADS, D_K, D_V, N_KV);
        $display("  Q_TILES=%0d KV_TILES=%0d DV_TILES=%0d K_PACKETS_QKV=%0d K_PACKETS_QK=%0d",
                 Q_TILES, KV_TILES, DV_TILES, K_PACKETS_QKV, K_PACKETS_QK);
        $display("  CPU model tolerance=%0.4f; DUT runs SFU stages; final values approximate.", CPU_TOL);
        $display("========================================================================\n");

        $display("[1] DMA preloading TCDM (%0d words) ...", NUM_PRELOAD);
        for (i = 0; i < NUM_PRELOAD; i++)
            if (pre_addr[i] !== 32'bx)
                dma_write(pre_addr[i], pre_data[i]);

        cpu_csr_read(A_RO_ID, rd32);
        check(rd32 == CSR_ID_VALUE, "CSR ID (SNAF)");

        // ------------------------------------------------------- RMSNorm (1) --
        $display("[2] SFU RMSNorm 1 ...");
        sfu_rms_tokens(X_BANK, GAMMA1_WORD, XN_BANK);
        check_rows(XN_BANK, g_xnorm, "SFU RMS1");

        // ------------------------------------------- activation quant (Xn) ---
        $display("[3] HW requant Xn -> activation packets ...");
        run_requant(0, XN_BANK, NUM_VEC_XN, 0, ACT_BANK, "requant Xn");

        // --------------------------------------------------- GEMM Q, K, V ---
        $display("[4] HW GEMM Q (M=2,N=8,K=8) ...");
        run_gemm(N_TILES_MODEL, M_TOKENS/16, K_PACKETS_QKV, W_W_Q,
                 Q_BASE_ADDR, "GEMM Q");
        $display("[5] SFU RoPE Q ...");
        sfu_rope(0, OUT_BANK);
        check_gemm_out_tol(OUT_BANK, N_TILES_MODEL, g_q, "SFU RoPE Q");
        $display("[6] HW requant Q -> packet staging ...");
        run_requant(0, OUT_BANK, NUM_QVEC, 0, QPKT_BANK, "requant Q");

        $display("[7] HW GEMM K (M=2,N=8,K=8) ...");
        run_gemm(N_TILES_MODEL, M_TOKENS/16, K_PACKETS_QKV, W_W_K,
                 Q_BASE_ADDR, "GEMM K");
        $display("[8] SFU RoPE K ...");
        sfu_rope(0, OUT_BANK);
        check_gemm_out_tol(OUT_BANK, N_TILES_MODEL, g_k, "SFU RoPE K");
        $display("[9] HW requant K -> packet staging ...");
        run_requant(0, OUT_BANK, NUM_QVEC, 0, KPKT_BANK, "requant K");

        // --------------------------------------------------- V fused jobs ----
        $display("[10] Fused V + transpose, one (h,dvi,kj) job at a time ...");
        for (h = 0; h < N_HEADS; h++)
            for (dvi = 0; dvi < DV_TILES; dvi++)
                for (int kj = 0; kj < KV_TILES; kj++) begin
                    run_gemm_v(h*DV_TILES + dvi, kj*16*K_PACKETS_QKV,
                               V_BASE_BLOCK + h*DV_TILES*KV_TILES +
                               dvi*KV_TILES + kj, "GEMM V + transpose");
                end
        // -------------------------------------- Q/K -> attention layout -----
        $display("[11] Reorder Q/K packets into attention layout ...");
        cpu_reorder_k();
        cpu_reorder_q();

        // ------------------------------------------------------ attention ----
        $display("[12] HW causal FlashAttention H=4, KV=2, DV=2 ...");
        run_attn("FlashAttention");
        cpu_reorder_attn_o();

        if ($test$plusargs("ATTN_FP16")) begin
            int dbg_bad;
            int dbg_idx, dbg_q, dbg_ch;
            run_attn_fp16_debug("Attention FP16 debug");
            dbg_bad = 0;
            for (h = 0; h < N_HEADS; h++)
                for (int qqi = 0; qqi < Q_TILES; qqi++)
                    for (dvi = 0; dvi < DV_TILES; dvi++)
                        for (int mm = 0; mm < M_Q; mm++) begin
                            dbg_idx = ((h*Q_TILES + qqi)*DV_TILES + dvi)*M_Q + mm;
                            dbg_q = qqi*M_Q + mm;
                            for (int ln = 0; ln < 16; ln++) begin
                                cpu_mem_read(tcdm_addr(dbg_idx*4 + ln/4, PKT_BANK), rw);
                                a16 = rw[16*(ln%4) +: 16];
                                dbg_ch = h*D_V + dvi*16 + ln;
                                e16 = g_attn_fp16[dbg_q*D_MODEL + dbg_ch];
                                if (a16 !== e16) begin
                                    if (dbg_bad < 20)
                                        $display("    [ATTN-FP16] h=%0d qi=%0d dvi=%0d m=%0d ch=%0d act=0x%04X exp=0x%04X",
                                                 h, qqi, dvi, mm, dbg_ch, a16, e16);
                                    dbg_bad++;
                                end
                            end
                        end
            $display("    Attention FP16 debug: %0d/%0d exact", NUM_FP16-dbg_bad, NUM_FP16);
        end

        // ------------------------------------------------------ GEMM O ------
        $display("[13] GEMM O from attention packets ...");
        run_gemm(N_TILES_MODEL, M_TOKENS/16, N_TILES_O,
                 W_W_O, tcdm_addr(O_ACT_WORD, O_ACT_BANK), "GEMM O");

        // ------------------------------------------ residual + RMSNorm 2 ----
        $display("[14] SFU residual 1 and RMSNorm 2 ...");
        sfu_residual_rows(0, OUT_BANK, 0, X_BANK, 0, RES1_BANK);
        check_rows(RES1_BANK, g_res1, "SFU residual 1");
        sfu_rms_tokens(RES1_BANK, GAMMA2_WORD, XN_BANK);
        check_rows(XN_BANK, g_n2, "SFU RMS2");

        $display("[15] Requant norm2 -> FFN activation ...");
        run_requant(0, XN_BANK, NUM_VEC_XN, 0, ACT_BANK, "requant norm2");

        // ------------------------------------------------- gate/up chunks ---
        $display("[16] GEMM gate/up in 2 N-chunks ...");
        for (c = 0; c < 2; c++) begin
            run_gemm(8, M_TOKENS/16, K_PACKETS_QKV,
                     W_W_GU + c*8*K_PACKETS_QKV, Q_BASE_ADDR, "GEMM gate chunk");
            cpu_copy_gate();
            run_gemm(8, M_TOKENS/16, K_PACKETS_QKV,
                     W_W_GU + (16 + c*8)*K_PACKETS_QKV, Q_BASE_ADDR, "GEMM up chunk");
            sfu_silu_chunk(c);
        end

        $display("[17] Requant FFN activation (2 x 256 vectors) ...");
        run_requant(0, FF1_BANK, NUM_VEC_XN, 0, ACT_BANK, "requant ffn lo");
        run_requant(0, FF2_BANK, NUM_VEC_XN, 256, ACT_BANK, "requant ffn hi");

        // ------------------------------------------------------ GEMM down ---
        $display("[18] GEMM down (M=2,N=8,K=16) ...");
        run_gemm(N_TILES_MODEL, M_TOKENS/16, K_PACKETS_DOWN,
                 W_W_D, Q_BASE_ADDR, "GEMM down");

        // -------------------------------------------------- residual 2 ------
        $display("[19] SFU residual 2 and final check ...");
        sfu_residual_final(0, OUT_BANK, 0, RES1_BANK, 0, FIN_BANK);

        stage_match = 0; stage_mismatch = 0;
        tol_reset();
        for (token = 0; token < M_TOKENS; token++)
            for (w = 0; w < ROW_WORDS; w++) begin
                cpu_mem_read(tcdm_addr(token*ROW_WORDS + w, FIN_BANK), rw);
                for (e = 0; e < 4; e++) begin
                    a16 = rw[16*e +: 16];
                    e16 = g_final[token*D_MODEL + 4*w + e];
                    tol_add(fp16_to_real(a16), fp16_to_real(e16));
                    if (a16 === e16) stage_match++;
                end
            end
        $display("    Final layer output: %0d/%0d exact (SFU approximate path)",
                 stage_match, NUM_FP16);
        tol_report("CPU tolerance: final residual", NUM_FP16);

        $display("\n========================================================================");
        $display("  MID-SIZE FULL LAYER RESULT");
        $display("========================================================================");
        $display("  Final FP16 values : %0d / %0d", stage_match, NUM_FP16);
        $display("  Total job cycles  : %0d", job_cycles_total);
        $display("  Cycle mix         : total=%0d seq_busy=%0d core_compute=%0d core_fp16=%0d cpu=%0d dma=%0d",
                 cyc_total, cyc_seq, cyc_core_compute, cyc_core_fp16, cyc_cpu, cyc_dma);
        $display("  Checks            : %0d passed, %0d failed", checks_pass, checks_fail);
        $display("========================================================================\n");

        if (checks_fail == 0)
            $display(">>> SFU MID LAYER COMPLETED: end-to-end SFU path exercised; final values are approximate <<<\n");
        else
            $display(">>> SFU MID LAYER CHECK FAIL (%0d check fails) <<<\n", checks_fail);

        $finish;
    end

endmodule
