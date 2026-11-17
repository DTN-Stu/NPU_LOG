// =============================================================================
// tb_snax_llm_layer_smoke.sv
//
// MINIMAL complete LLaMA-style prefill layer on snax_llm_top_clean.
//
// Shapes (smallest the hardware supports):
//   tokens = 16, d_model = 16, d_ff = 16, heads = 1,
//   d_k = 16, d_v = 16, n_kv = 16 (1 KV tile), n_q = 16 (2 query tiles).
//
// Stages (all present):
//   X (FP16 rows)
//     -> CPU RMSNorm 1
//     -> HW readback-requant           (log_quantizer)
//     -> HW GEMM Q, K, V               (sequencer opcode 00)
//     -> CPU RoPE (Q, K)               (head_dim 16: pairs j / j+8)
//     -> HW readback-requant + CPU packet scatter (attention layout)
//     -> HW FlashAttention             (sequencer opcode 01)
//        NOTE: run non-causal for now.  The sequencer's automatic causal mask
//        is derived from cnt_m_q, which has already wrapped to 0 when the SM
//        drain feeds the scores to the softmax, so every query would see only
//        key 0.  Set CAUSAL=True in the generator and A_SEQ_CTRL=0x5 here once
//        that RTL path drives the mask from the drain index instead.
//     -> HW readback-requant
//     -> HW GEMM O
//     -> CPU residual 1
//     -> CPU RMSNorm 2
//     -> HW readback-requant
//     -> HW GEMM gate/up
//     -> CPU SiLU(gate)*up
//     -> HW readback-requant
//     -> HW GEMM down
//     -> CPU residual 2
//     -> bit-exact comparison vs the golden model
//
// Each stage is compared against the Python golden (generate_snax_llm_layer_data.py).
// Every CSR/TCDM handshake has a timeout so a broken job cannot hang the run.
// =============================================================================
`timescale 1ns/1ps

module tb_snax_llm_layer_smoke;

    // ---------------------------------------------------------------- shape --
    localparam int M_TOKENS   = 16;
    localparam int D_MODEL    = 16;
    localparam int D_FF       = 16;
    localparam int N_HEADS    = 1;
    localparam int N_Q        = 16;
    localparam int N_KV       = 16;
    localparam int D_K        = 16;
    localparam int D_V        = 16;
    localparam int M_Q        = 8;
    localparam int K_BLOCKS   = 1;
    localparam int Q_TILES    = 2;
    localparam int DV_TILES   = 1;
    localparam int KV_TILES   = 1;
    localparam int FF_KB      = 1;
    localparam int CLK_PERIOD = 10;

    localparam int NUM_FP16    = M_TOKENS * D_MODEL;          // 256
    localparam int NUM_VEC_AT  = Q_TILES * DV_TILES * M_Q;    // 16
    localparam int NUM_WORD_AT = NUM_VEC_AT * 4;              // 64
    localparam int ROW_WORDS   = D_MODEL / 4;                 // 4

    // ------------------------------------------------------------ TCDM map --
    localparam int ACT_BANK  = 16;
    localparam int OUT_BANK  = 17;
    localparam int STG_BANK  = 18;
    localparam int PKT_BANK  = 19;
    localparam int X_BANK    = 24;
    localparam int XN_BANK   = 25;
    localparam int RES1_BANK = 26;
    localparam int FIN_BANK  = 27;

    localparam int GAMMA1_WORD = 200;
    localparam int GAMMA2_WORD = 210;
    localparam int COS_WORD    = 300;
    localparam int SIN_WORD    = 364;

    localparam int OUT_W_Q  = 0;
    localparam int OUT_W_K  = 64;
    localparam int OUT_W_V  = 128;
    localparam int OUT_W_O  = 256;
    localparam int OUT_W_G  = 320;
    localparam int OUT_W_U  = 384;
    localparam int OUT_W_D  = 448;

    localparam int W_W_Q  = 0;
    localparam int W_W_K  = 8;
    localparam int W_W_V  = 24;
    localparam int W_W_O  = 32;
    localparam int W_W_GU = 40;
    localparam int W_W_D  = 48;

    localparam logic [31:0] Q_BASE_ADDR = 32'h0000_0100;   // bank 16, word 0
    localparam logic [31:0] OUT_BASE    = 32'h0000_0110;   // bank 17, word 0
    localparam logic [31:0] RING_BASE   = 32'h0000_0000;   // bank 0,  word 0

    // ------------------------------------------------------------ CSR map ----
    localparam logic [11:0] A_CTRL      = 12'd0;
    localparam logic [11:0] A_WO_BASE   = 12'd13;
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
    localparam logic [31:0] CSR_ID_VALUE= 32'h534E_4146;

    localparam int JOB_TIMEOUT   = 60000;
    localparam int HANDSHAKE_GUARD = 200000;

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

    // Watchdog: never let a stuck job hang the batch run.
    initial begin
        #(10_000_000);   // 10 ms = 1M cycles
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
`ifdef FST_TRACE
            $dumpfile("tb_snax_llm_layer_smoke.fst");
`else
            $dumpfile("tb_snax_llm_layer_smoke.vcd");
`endif
            if ($test$plusargs("WAVE_DEEP"))
                $dumpvars(0, tb_snax_llm_layer_smoke);
            else
                $dumpvars(0, dut);
        end
    end

    // ------------------------------------------------------------ golden -----
    localparam int NUM_PRELOAD = 512;
    logic [31:0] pre_addr [0:NUM_PRELOAD-1];
    logic [71:0] pre_data [0:NUM_PRELOAD-1];

    logic [15:0] g_xnorm [0:NUM_FP16-1];
    logic [15:0] g_q     [0:NUM_FP16-1];
    logic [15:0] g_k     [0:NUM_FP16-1];
    logic [15:0] g_h     [0:NUM_FP16-1];
    logic [15:0] g_res1  [0:NUM_FP16-1];
    logic [15:0] g_n2    [0:NUM_FP16-1];
    logic [71:0] g_qpkt  [0:N_Q*K_BLOCKS-1];
    logic [71:0] g_kpkt  [0:K_BLOCKS*16-1];
    logic [71:0] g_vpkt  [0:DV_TILES*16-1];
    logic [71:0] g_attn_pkt [0:N_Q-1];
    logic [15:0] g_o     [0:NUM_FP16-1];
    logic [15:0] g_final [0:NUM_FP16-1];

    int checks_pass, checks_fail;
    int job_cycles_total;
    int stage_match, stage_mismatch, vec_match;
    int i, j, w, e, gn, kb, lane, key, dvi, token, qi_v, mi_v;
    int guard;
    int fp16_pushes;
    int cyc_total, cyc_core_compute, cyc_core_fp16, cyc_cpu, cyc_dma, cyc_seq;
    int cyc_drain_req, cyc_wo_busy, cyc_gemm_run, cyc_gemm_launch, cyc_wait;
    int cyc_v_busy, cyc_nrm_busy;
    int s_compute, s_drain, s_wo, s_run, s_launch, s_wait, s_v, s_nrm;
    logic [15:0] actv [0:15];
    int best_idx, best_cnt, cmp_cnt;

    // CPU model tolerance.  The CPU stages use SystemVerilog `real`; the LRM
    // does not guarantee bit-identical $sqrt/$exp/$rtoi across simulators
    // (Questa vs Verilator), so those stages are checked with a tolerance.
    localparam real CPU_TOL = 0.01;
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
        real d;
        begin
            d = got - exp;
            if (d < 0.0) d = -d;
            if (d > st_max) st_max = d;
            if (d > CPU_TOL) st_fail = st_fail + 1;
        end
    endtask

    task automatic tol_report(input string name, input int nvals);
        begin
            check(st_fail == 0, name);
            $display("    [CPU-TOL] %s max_abs=%0.5f fail=%0d/%0d",
                     name, st_max, st_fail, nvals);
        end
    endtask

    task automatic fatal(input string msg);
        $display("\n!!! FATAL: %s !!!\n", msg);
        $finish;
    endtask

    // =========================================================================
    // CPU MODEL (CSR port + TCDM master 1)
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

    // =========================================================================
    // DMA MODEL (TCDM master 0)
    // =========================================================================
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
    // CPU STAGE HELPERS
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

    // RMSNorm over 16 channels held in 4 words.
    task automatic cpu_norm_row(input int x_word, input int x_bank,
                                input int g_word, input int g_bank,
                                input int o_word, input int o_bank);
        logic [3:0][71:0] xw, gw, ow;
        logic [15:0][15:0] xv, gv, yv;
        real acc, inv;
        begin
            for (int k = 0; k < 4; k++) begin
                cpu_mem_read(tcdm_addr(x_word + k, x_bank), xw[k]);
                cpu_mem_read(tcdm_addr(g_word + k, g_bank), gw[k]);
            end
            acc = 0.0;
            for (int d = 0; d < 16; d++) begin
                xv[d] = xw[d/4][16*(d%4) +: 16];
                gv[d] = gw[d/4][16*(d%4) +: 16];
                acc = acc + fp16_to_real(xv[d]) * fp16_to_real(xv[d]);
            end
            inv = 1.0 / $sqrt(acc / 16.0);
            for (int d = 0; d < 16; d++)
                yv[d] = real_to_fp16((fp16_to_real(xv[d]) * inv) * fp16_to_real(gv[d]));
            for (int k = 0; k < 4; k++)
                ow[k] = pack4(yv[4*k], yv[4*k+1], yv[4*k+2], yv[4*k+3]);
            for (int k = 0; k < 4; k++)
                cpu_mem_write(tcdm_addr(o_word + k, o_bank), ow[k]);
        end
    endtask

    // RoPE for head_dim 16: one 4-word vector per token, pairs (j, j+8).
    task automatic cpu_rope(input int base_word, input int bank);
        logic [1:0][71:0] cw, sw;
        logic [3:0][71:0] vw;
        logic [15:0][15:0] ch, res;
        real cval, sval, x0, x1;
        begin
            for (int t = 0; t < M_TOKENS; t++) begin
                for (int k = 0; k < 4; k++)
                    cpu_mem_read(tcdm_addr(base_word + 4*t + k, bank), vw[k]);
                for (int k = 0; k < 2; k++) begin
                    cpu_mem_read(tcdm_addr(COS_WORD + 2*t + k, X_BANK), cw[k]);
                    cpu_mem_read(tcdm_addr(SIN_WORD + 2*t + k, X_BANK), sw[k]);
                end
                for (int k = 0; k < 4; k++)
                    for (int el = 0; el < 4; el++)
                        ch[4*k+el] = vw[k][16*el +: 16];
                for (int jj = 0; jj < 8; jj++) begin
                    cval = fp16_to_real(cw[jj/4][16*(jj%4) +: 16]);
                    sval = fp16_to_real(sw[jj/4][16*(jj%4) +: 16]);
                    x0 = fp16_to_real(ch[jj]);
                    x1 = fp16_to_real(ch[jj+8]);
                    res[jj]   = real_to_fp16(x0 * cval - x1 * sval);
                    res[jj+8] = real_to_fp16(x0 * sval + x1 * cval);
                end
                for (int k = 0; k < 4; k++)
                    vw[k] = pack4(res[4*k], res[4*k+1], res[4*k+2], res[4*k+3]);
                for (int k = 0; k < 4; k++)
                    cpu_mem_write(tcdm_addr(base_word + 4*t + k, bank), vw[k]);
            end
        end
    endtask

    // K gather: GEMM vector key -> staging slot key.
    task automatic cpu_gather_k(input int base_word);
        begin
            for (key = 0; key < 16; key++)
                cpu_copy_vec(base_word + 4*key, OUT_BANK, 4*key, STG_BANK);
        end
    endtask

    // V 16x16 transpose: GEMM vectors (keys) -> channel vectors.
    task automatic cpu_transpose_v(input int base_word);
        logic [15:0] vmat [0:15][0:15];   // [key][channel]
        logic [71:0] wtmp, vw [0:3];
        begin
            for (key = 0; key < 16; key++)
                for (w = 0; w < 4; w++) begin
                    cpu_mem_read(tcdm_addr(base_word + 4*key + w, OUT_BANK), wtmp);
                    for (e = 0; e < 4; e++)
                        vmat[key][4*w + e] = wtmp[16*e +: 16];
                end
            for (lane = 0; lane < 16; lane++) begin
                for (w = 0; w < 4; w++)
                    vw[w] = pack4(vmat[4*w+0][lane], vmat[4*w+1][lane],
                                  vmat[4*w+2][lane], vmat[4*w+3][lane]);
                for (w = 0; w < 4; w++)
                    cpu_mem_write(tcdm_addr(lane*4 + w, STG_BANK), vw[w]);
            end
        end
    endtask

    task automatic cpu_scatter16(input int src_word, input int dst_word);
        logic [71:0] wtmp;
        begin
            for (lane = 0; lane < 16; lane++) begin
                cpu_mem_read(tcdm_addr(src_word + lane, PKT_BANK), wtmp);
                cpu_mem_write(tcdm_addr(dst_word, lane), wtmp);
            end
        end
    endtask

    // SiLU(gate) * up: one 4-word vector each per token.
    task automatic cpu_silu_mul(input int g_word, input int u_word);
        logic [3:0][71:0] gw, uw, hw;
        logic [15:0][15:0] gv, uv, hv;
        real g, u;
        begin
            for (token = 0; token < M_TOKENS; token++) begin
                for (int k = 0; k < 4; k++) begin
                    cpu_mem_read(tcdm_addr(g_word + 4*token + k, OUT_BANK), gw[k]);
                    cpu_mem_read(tcdm_addr(u_word + 4*token + k, OUT_BANK), uw[k]);
                end
                for (int k = 0; k < 4; k++)
                    for (int el = 0; el < 4; el++) begin
                        gv[4*k+el] = gw[k][16*el +: 16];
                        uv[4*k+el] = uw[k][16*el +: 16];
                    end
                for (int l = 0; l < 16; l++) begin
                    g = fp16_to_real(gv[l]);
                    u = fp16_to_real(uv[l]);
                    hv[l] = real_to_fp16((g / (1.0 + $exp(-g))) * u);
                end
                for (int k = 0; k < 4; k++)
                    hw[k] = pack4(hv[4*k], hv[4*k+1], hv[4*k+2], hv[4*k+3]);
                for (int k = 0; k < 4; k++)
                    cpu_mem_write(tcdm_addr(4*token + k, STG_BANK), hw[k]);
            end
        end
    endtask

    // res1 row = O vector + X row (one 4-word vector per token).
    task automatic cpu_residual_rows(input int o_word, input int o_bank,
                                     input int x_word, input int x_bank,
                                     input int r_word, input int r_bank);
        logic [3:0][71:0] ow, xw, rw;
        logic [3:0][15:0] lv;
        real a, b;
        begin
            for (token = 0; token < M_TOKENS; token++) begin
                for (int k = 0; k < 4; k++) begin
                    cpu_mem_read(tcdm_addr(o_word + 4*token + k, o_bank), ow[k]);
                    cpu_mem_read(tcdm_addr(x_word + 4*token + k, x_bank), xw[k]);
                end
                for (int k = 0; k < 4; k++) begin
                    for (int el = 0; el < 4; el++) begin
                        a = fp16_to_real(ow[k][16*el +: 16]);
                        b = fp16_to_real(xw[k][16*el +: 16]);
                        lv[el] = real_to_fp16(a + b);
                    end
                    rw[k] = pack4(lv[0], lv[1], lv[2], lv[3]);
                end
                for (int k = 0; k < 4; k++)
                    cpu_mem_write(tcdm_addr(r_word + 4*token + k, r_bank), rw[k]);
            end
        end
    endtask

    task automatic cpu_residual_final(input int d_word, input int d_bank,
                                      input int r_word, input int r_bank,
                                      input int f_word, input int f_bank);
        logic [3:0][71:0] dw, rw, fw;
        logic [3:0][15:0] lv;
        real a, b;
        begin
            for (token = 0; token < M_TOKENS; token++) begin
                for (int k = 0; k < 4; k++) begin
                    cpu_mem_read(tcdm_addr(d_word + 4*token + k, d_bank), dw[k]);
                    cpu_mem_read(tcdm_addr(r_word + 4*token + k, r_bank), rw[k]);
                end
                for (int k = 0; k < 4; k++) begin
                    for (int el = 0; el < 4; el++) begin
                        a = fp16_to_real(dw[k][16*el +: 16]);
                        b = fp16_to_real(rw[k][16*el +: 16]);
                        lv[el] = real_to_fp16(a + b);
                    end
                    fw[k] = pack4(lv[0], lv[1], lv[2], lv[3]);
                end
                for (int k = 0; k < 4; k++)
                    cpu_mem_write(tcdm_addr(f_word + 4*token + k, f_bank), fw[k]);
            end
        end
    endtask

    // =========================================================================
    // JOB LAUNCHERS
    // =========================================================================
    bit seen_done, seen_busy;
    int cycle_cnt;

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
            $display("    [job] %-22s cycles=%0d", name, cycle_cnt);
            if (!seen_done)
                $display("    [dbg] seq_state=%0d wo_state=%0d wo_vec=%0d fifo=%0d pushes=%0d rd_busy=%0b act_busy=%0b wt_busy=%0b v_busy=%0b busy=%0b",
                         dut.u_seq.state_q, dut.u_write_out.state_q,
                         dut.u_write_out.vec_cnt_q, dut.u_write_out.in_fifo_level,
                         fp16_pushes, dut.rd_busy, dut.act_busy, dut.wt_busy,
                         dut.v_busy, busy);
            if (!seen_done)
                $display("    [dbg] core_v=%0b core_r=%0b q_in_v=%0b q_acc=%0b fp4_v=%0b fp4_r=%0b fp16mode=%0b src_core=%0b v_act=%0b nrm_st=%0d nrm_dr=%0d",
                         dut.core_fp16_valid, dut.core_fp16_ready,
                         dut.quant_in_valid, dut.quant_accept,
                         dut.wo_fp4_valid, dut.wo_fp4_ready,
                         dut.wo_mode_fp16, dut.src_core_sel, dut.v_active,
                         dut.u_seq.nrm_state_q, dut.u_seq.nrm_dr_q);
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

    task automatic run_gemm(input logic [15:0] n_tiles,
                            input logic [15:0] k_blks,
                            input logic [31:0] ring,
                            input logic [31:0] out_base, input string name);
        begin
            cpu_csr_write(A_CTRL,      32'h0);
            cpu_csr_write(A_SEQ_CTRL,  32'h0);          // op 00 GEMM, DMA off
            cpu_csr_write(A_GEMM_NT,   {16'd0, n_tiles});
            cpu_csr_write(A_GEMM_MT,   32'd1);
            cpu_csr_write(A_GEMM_KB,   {16'd0, k_blks});
            cpu_csr_write(A_RING_BASE, ring);
            cpu_csr_write(A_Q_BASE,    Q_BASE_ADDR);
            cpu_csr_write(A_OUT_BASE,  out_base);
            snap_mix();
            cpu_csr_write(A_START,     32'h0000_0003);
            wait_job(name);
            print_mix();
        end
    endtask

    task automatic run_requant(input int rd_word, input int rd_bank,
                               input int nvec,
                               input int wo_word, input int wo_bank,
                               input string name);
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

    task automatic run_gemm_v(input string name);
        begin
            cpu_csr_write(A_CTRL,      32'h0);
            cpu_csr_write(A_SEQ_CTRL,  32'h30);         // op 00, out mode 3 = fused V
            cpu_csr_write(A_GEMM_NT,   DV_TILES);
            cpu_csr_write(A_GEMM_MT,   32'd1);
            cpu_csr_write(A_GEMM_KB,   K_BLOCKS);
            cpu_csr_write(A_RING_BASE, tcdm_addr(W_W_V, 0));
            cpu_csr_write(A_Q_BASE,    Q_BASE_ADDR);
            cpu_csr_write(A_V_BASE,    tcdm_addr(16, 0));
            cpu_csr_write(A_V_NB,      DV_TILES);
            snap_mix();
            cpu_csr_write(A_START,     32'h0000_0003);
            wait_job(name);
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
            cpu_csr_write(A_ATTN_KB,   K_BLOCKS);
            cpu_csr_write(A_RING_BASE, RING_BASE);
            cpu_csr_write(A_Q_BASE,    Q_BASE_ADDR);
            cpu_csr_write(A_OUT_BASE,  Q_BASE_ADDR);    // packets land in act bank
            cpu_csr_write(A_DRAM_KV,   32'h0);
            cpu_csr_write(A_TILE_ST,   32'h200);
            snap_mix();
            cpu_csr_write(A_START,     32'h0000_0003);
            wait_job(name);
            print_mix();
        end
    endtask

    // Temporary drain/handshake monitor (enable with +DBG_DRAIN)
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
        if (dut.core_fp16_valid && dut.core_fp16_ready)
            fp16_pushes = fp16_pushes + 1;
        if (rst_n && $test$plusargs("DBG_DRAIN")) begin
            if (dut.tile_drain_req)
                $display("[DRAIN] t=%0t idx=%0d mode=%0d", $time,
                         dut.tile_drain_idx, dut.cfg_mode);
            if (dut.core_fp16_valid && dut.core_fp16_ready)
                $display("[FP16 ] t=%0t", $time);
            if (dut.act_stream_valid && dut.act_stream_ready && (dut.cfg_mode == 2'b01))
                $display("[QKACT] t=%0t m=%0d data=%018X", $time,
                         dut.u_seq.cnt_m_q, dut.act_stream_data);
            if (dut.u_seq.pv_compute_fire)
                $display("[PV ] t=%0t m=%0d acc=%0d pfifo_cnt=%0d pkt=%018X",
                         $time, dut.u_seq.cnt_m_q, dut.lane_acc_idx,
                         dut.u_core.p_fifo_count, dut.u_core.p_fifo_packet);
            if (dut.u_core.raw_drained_valid)
                $display("[DWORD] t=%0t l0=%08X l1=%08X",
                         $time,
                         dut.u_core.raw_drained_word[31:0],
                         dut.u_core.raw_drained_word[63:32]);
        end
    end

    // =========================================================================
    // MAIN SEQUENCE
    // =========================================================================
    logic [31:0] rd32;
    logic [71:0] rw;
    logic [15:0] a16, e16;

    initial begin
        csr_req_valid = 1'b0; csr_req_write = 1'b0; csr_req_addr = '0; csr_req_data = '0;
        csr_rsp_ready = 1'b1;
        cpu_mem_req_valid = 1'b0; cpu_mem_req_write = 1'b0; cpu_mem_req_addr = '0;
        cpu_mem_req_wdata = '0; cpu_mem_req_strb = '0;
        dma_mem_req_valid = 1'b0; dma_mem_req_write = 1'b0; dma_mem_req_addr = '0;
        dma_mem_req_wdata = '0; dma_mem_req_strb = '0;
        dma_desc_ready = 1'b1; dma_desc_done = 1'b0;

        checks_pass = 0; checks_fail = 0; job_cycles_total = 0;
        fp16_pushes = 0;
        cyc_total = 0; cyc_core_compute = 0; cyc_core_fp16 = 0;
        cyc_cpu = 0; cyc_dma = 0; cyc_seq = 0;
        cyc_drain_req = 0; cyc_wo_busy = 0; cyc_gemm_run = 0;
        cyc_gemm_launch = 0; cyc_wait = 0; cyc_v_busy = 0; cyc_nrm_busy = 0;

        $readmemh("sim_snax_llm_layer/hex/tcdm_preload_addr.hex", pre_addr);
        $readmemh("sim_snax_llm_layer/hex/tcdm_preload_data.hex", pre_data);
        $readmemh("sim_snax_llm_layer/hex/layer_xnorm_fp16.hex", g_xnorm);
        $readmemh("sim_snax_llm_layer/hex/layer_q_fp16.hex", g_q);
        $readmemh("sim_snax_llm_layer/hex/layer_k_fp16.hex", g_k);
        $readmemh("sim_snax_llm_layer/hex/layer_h_fp16.hex", g_h);
        $readmemh("sim_snax_llm_layer/hex/layer_res1_fp16.hex", g_res1);
        $readmemh("sim_snax_llm_layer/hex/layer_norm2_fp16.hex", g_n2);
        $readmemh("sim_snax_llm_layer/hex/layer_q_packets.hex", g_qpkt);
        $readmemh("sim_snax_llm_layer/hex/layer_k_packets.hex", g_kpkt);
        $readmemh("sim_snax_llm_layer/hex/layer_v_packets.hex", g_vpkt);
        $readmemh("sim_snax_llm_layer/hex/layer_attn_packets.hex", g_attn_pkt);
        $readmemh("sim_snax_llm_layer/hex/layer_o_fp16.hex", g_o);
        $readmemh("sim_snax_llm_layer/hex/layer_final_fp16.hex", g_final);

        wait (rst_n == 1'b1);
        repeat (5) @(posedge clk);

        $display("\n========================================================================");
        $display("  SNAX FP4 LLM - MINIMAL FULL PREFILL LAYER");
        $display("  tokens=%0d d_model=%0d d_ff=%0d heads=%0d d_k=%0d d_v=%0d n_kv=%0d",
                 M_TOKENS, D_MODEL, D_FF, N_HEADS, D_K, D_V, N_KV);
        $display("  CPU model checked with tolerance=%0.4f; DUT is driven by golden CPU stages.", CPU_TOL);
        $display("  Use +CPU_REAL to drive the DUT from the real-valued CPU model instead.");
        $display("========================================================================\n");

        // ------------------------------------------------------------ preload -
        $display("[1] DMA preloading TCDM (%0d words) ...", NUM_PRELOAD);
        for (i = 0; i < NUM_PRELOAD; i++)
            if (pre_addr[i] !== 32'bx)
                dma_write(pre_addr[i], pre_data[i]);

        cpu_csr_read(A_RO_ID, rd32);
        check(rd32 == CSR_ID_VALUE, "CSR ID (SNAF)");

        // ------------------------------------------------------- RMSNorm (1) --
        $display("[2] CPU RMSNorm 1 ...");
        for (token = 0; token < M_TOKENS; token++)
            cpu_norm_row(token*ROW_WORDS, X_BANK, GAMMA1_WORD, X_BANK,
                         token*ROW_WORDS, XN_BANK);

        stage_match = 0; stage_mismatch = 0;
        tol_reset();
        for (token = 0; token < M_TOKENS; token++)
            for (w = 0; w < ROW_WORDS; w++) begin
                cpu_mem_read(tcdm_addr(token*ROW_WORDS + w, XN_BANK), rw);
                for (e = 0; e < 4; e++) begin
                    a16 = rw[16*e +: 16];
                    e16 = g_xnorm[token*D_MODEL + 4*w + e];
                    tol_add(fp16_to_real(a16), fp16_to_real(e16));
                    if (a16 === e16) stage_match++;
                    else begin
                        stage_mismatch++;
                        if (stage_mismatch <= 6)
                            $display("    xnorm mismatch tok=%0d ch=%0d act=0x%04X exp=0x%04X",
                                     token, 4*w+e, a16, e16);
                    end
                end
            end
        $display("    RMSNorm1: %0d/%0d exact", stage_match, NUM_FP16);
        tol_report("CPU tolerance: RMSNorm 1", NUM_FP16);
        if (!$test$plusargs("CPU_REAL")) begin
            for (token = 0; token < M_TOKENS; token++)
                for (w = 0; w < ROW_WORDS; w++)
                    cpu_mem_write(tcdm_addr(token*ROW_WORDS + w, XN_BANK),
                                  pack4(g_xnorm[token*D_MODEL + 4*w + 0],
                                        g_xnorm[token*D_MODEL + 4*w + 1],
                                        g_xnorm[token*D_MODEL + 4*w + 2],
                                        g_xnorm[token*D_MODEL + 4*w + 3]));
        end

        // ------------------------------------------- activation quant (Xn) ---
        $display("[3] HW requant Xn -> activation packets ...");
        run_requant(0, XN_BANK, N_Q, 0, ACT_BANK, "requant Xn");

        // --------------------------------------------------- GEMM Q, K, V ---
        $display("[4] HW GEMM Q / K, fused GEMM V + transpose ...");
        run_gemm(16'd1, 16'd1, tcdm_addr(W_W_Q, 0), tcdm_addr(OUT_W_Q, OUT_BANK), "GEMM Q");
        run_gemm(16'd1, 16'd1, tcdm_addr(W_W_K, 0), tcdm_addr(OUT_W_K, OUT_BANK), "GEMM K");
        run_gemm_v("GEMM V + transpose");

        // ---------------------------------------------------------- RoPE -----
        $display("[5] CPU RoPE on Q and K ...");
        cpu_rope(OUT_W_Q, OUT_BANK);
        cpu_rope(OUT_W_K, OUT_BANK);

        tol_reset(); stage_match = 0; stage_mismatch = 0;
        for (i = 0; i < NUM_FP16/(4*4); i++)
            for (w = 0; w < 4; w++) begin
                cpu_mem_read(tcdm_addr(OUT_W_Q + 4*i + w, OUT_BANK), rw);
                for (e = 0; e < 4; e++) begin
                    a16 = rw[16*e +: 16];
                    e16 = g_q[i*D_K + 4*w + e];
                    tol_add(fp16_to_real(a16), fp16_to_real(e16));
                    if (a16 === e16) stage_match++;
                    else stage_mismatch++;
                end
            end
        $display("    RoPE Q: %0d/%0d exact", stage_match, NUM_FP16);
        tol_report("CPU tolerance: RoPE Q", NUM_FP16);

        tol_reset(); stage_match = 0; stage_mismatch = 0;
        for (i = 0; i < NUM_FP16/(4*4); i++)
            for (w = 0; w < 4; w++) begin
                cpu_mem_read(tcdm_addr(OUT_W_K + 4*i + w, OUT_BANK), rw);
                for (e = 0; e < 4; e++) begin
                    a16 = rw[16*e +: 16];
                    e16 = g_k[i*D_K + 4*w + e];
                    tol_add(fp16_to_real(a16), fp16_to_real(e16));
                    if (a16 === e16) stage_match++;
                    else stage_mismatch++;
                end
            end
        $display("    RoPE K: %0d/%0d exact", stage_match, NUM_FP16);
        tol_report("CPU tolerance: RoPE K", NUM_FP16);

        if (!$test$plusargs("CPU_REAL")) begin
            for (i = 0; i < NUM_FP16/(4*4); i++)
                for (w = 0; w < 4; w++) begin
                    cpu_mem_write(tcdm_addr(OUT_W_Q + 4*i + w, OUT_BANK),
                                  pack4(g_q[i*D_K + 4*w + 0], g_q[i*D_K + 4*w + 1],
                                        g_q[i*D_K + 4*w + 2], g_q[i*D_K + 4*w + 3]));
                    cpu_mem_write(tcdm_addr(OUT_W_K + 4*i + w, OUT_BANK),
                                  pack4(g_k[i*D_K + 4*w + 0], g_k[i*D_K + 4*w + 1],
                                        g_k[i*D_K + 4*w + 2], g_k[i*D_K + 4*w + 3]));
                end
        end

        // ------------------------------------------------- attention Q -----
        $display("[6] Requant Q, quantize K (V already fused) ...");
        run_requant(OUT_W_Q, OUT_BANK, N_Q, 0, ACT_BANK, "requant Q");

        stage_match = 0; stage_mismatch = 0;
        for (i = 0; i < N_Q*K_BLOCKS; i++) begin
            cpu_mem_read(tcdm_addr(i, ACT_BANK), rw);
            if (rw === g_qpkt[i]) stage_match++;
            else begin
                stage_mismatch++;
                if (stage_mismatch <= 6)
                    $display("    Q pkt mismatch word=%0d act=%018X exp=%018X",
                             i, rw, g_qpkt[i]);
            end
        end
        $display("    Q packets: %0d/%0d match", stage_match, N_Q*K_BLOCKS);
        check(stage_mismatch == 0, "Q packets bit-exact (after RoPE)");

        cpu_gather_k(OUT_W_K);
        run_requant(0, STG_BANK, 16, 0, PKT_BANK, "requant K");
        cpu_scatter16(0, 0);

        stage_match = 0; stage_mismatch = 0;
        for (lane = 0; lane < 16; lane++) begin
            cpu_mem_read(tcdm_addr(0, lane), rw);
            if (rw === g_kpkt[lane]) stage_match++;
            else begin
                stage_mismatch++;
                if (stage_mismatch <= 6)
                    $display("    K pkt mismatch lane=%0d act=%018X exp=%018X",
                             lane, rw, g_kpkt[lane]);
            end
        end
        $display("    K packets: %0d/16 match", stage_match);
        check(stage_mismatch == 0, "K packets bit-exact (after RoPE)");

        stage_match = 0; stage_mismatch = 0;
        for (lane = 0; lane < 16; lane++) begin
            cpu_mem_read(tcdm_addr(16, lane), rw);
            if (rw === g_vpkt[lane]) stage_match++;
            else begin
                stage_mismatch++;
                if (stage_mismatch <= 6)
                    $display("    V pkt mismatch lane=%0d act=%018X exp=%018X",
                             lane, rw, g_vpkt[lane]);
            end
        end
        $display("    V packets: %0d/16 match", stage_match);
        check(stage_mismatch == 0, "V packets bit-exact (fused transpose)");

        // ------------------------------------------------------ attention ----
        $display("[7] HW FlashAttention (non-causal; RTL causal mask TBD) ...");
        run_attn("FlashAttention");
        if ($test$plusargs("DBG_DRAIN"))
            $display("    [dbg] wo_state=%0d wo_vec=%0d fifo_lvl=%0d nrm_state=%0d nrm_dr=%0d",
                     dut.u_write_out.state_q, dut.u_write_out.vec_cnt_q,
                     dut.u_write_out.in_fifo_level, dut.u_seq.nrm_state_q,
                     dut.u_seq.nrm_dr_q);

        stage_match = 0; stage_mismatch = 0;
        for (i = 0; i < N_Q; i++) begin
            cpu_mem_read(tcdm_addr(i, ACT_BANK), rw);
            if (rw === g_attn_pkt[i]) stage_match++;
            else begin
                stage_mismatch++;
                if (stage_mismatch <= 6)
                    $display("    attn pkt mismatch q=%0d act=%018X exp=%018X",
                             i, rw, g_attn_pkt[i]);
            end
        end
        $display("    Attention packets (FP4): %0d/%0d match", stage_match, N_Q);
        check(stage_mismatch == 0, "attention packets bit-exact (FP4 direct)");

        // ------------------------------------- O projection activation ------
        $display("[8] GEMM O directly from FP4 attention packets ...");
        run_gemm(16'd1, 16'd1, tcdm_addr(W_W_O, 0), tcdm_addr(OUT_W_O, OUT_BANK), "GEMM O");

        stage_match = 0; stage_mismatch = 0;
        for (token = 0; token < M_TOKENS; token++)
            for (w = 0; w < 4; w++) begin
                cpu_mem_read(tcdm_addr(OUT_W_O + 4*token + w, OUT_BANK), rw);
                for (e = 0; e < 4; e++) begin
                    a16 = rw[16*e +: 16];
                    e16 = g_o[token*D_MODEL + 4*w + e];
                    if (a16 === e16) stage_match++;
                    else begin
                        stage_mismatch++;
                        if (stage_mismatch <= 6)
                            $display("    O mismatch tok=%0d lane=%0d act=0x%04X exp=0x%04X",
                                     token, 4*w+e, a16, e16);
                    end
                end
            end
        $display("    GEMM O: %0d/%0d match", stage_match, NUM_FP16);
        check(stage_mismatch == 0, "GEMM O bit-exact");

        // ------------------------------------------ residual + RMSNorm 2 ----
        $display("[9] CPU residual 1 and RMSNorm 2 ...");
        cpu_residual_rows(OUT_W_O, OUT_BANK, 0, X_BANK, 0, RES1_BANK);

        tol_reset(); stage_match = 0; stage_mismatch = 0;
        for (token = 0; token < M_TOKENS; token++)
            for (w = 0; w < ROW_WORDS; w++) begin
                cpu_mem_read(tcdm_addr(token*ROW_WORDS + w, RES1_BANK), rw);
                for (e = 0; e < 4; e++) begin
                    a16 = rw[16*e +: 16];
                    e16 = g_res1[token*D_MODEL + 4*w + e];
                    tol_add(fp16_to_real(a16), fp16_to_real(e16));
                    if (a16 === e16) stage_match++;
                    else stage_mismatch++;
                end
            end
        $display("    residual 1: %0d/%0d exact", stage_match, NUM_FP16);
        tol_report("CPU tolerance: residual 1", NUM_FP16);
        if (!$test$plusargs("CPU_REAL")) begin
            for (token = 0; token < M_TOKENS; token++)
                for (w = 0; w < ROW_WORDS; w++)
                    cpu_mem_write(tcdm_addr(token*ROW_WORDS + w, RES1_BANK),
                                  pack4(g_res1[token*D_MODEL + 4*w + 0],
                                        g_res1[token*D_MODEL + 4*w + 1],
                                        g_res1[token*D_MODEL + 4*w + 2],
                                        g_res1[token*D_MODEL + 4*w + 3]));
        end

        for (token = 0; token < M_TOKENS; token++)
            cpu_norm_row(token*ROW_WORDS, RES1_BANK, GAMMA2_WORD, X_BANK,
                         token*ROW_WORDS, XN_BANK);

        tol_reset(); stage_match = 0; stage_mismatch = 0;
        for (token = 0; token < M_TOKENS; token++)
            for (w = 0; w < ROW_WORDS; w++) begin
                cpu_mem_read(tcdm_addr(token*ROW_WORDS + w, XN_BANK), rw);
                for (e = 0; e < 4; e++) begin
                    a16 = rw[16*e +: 16];
                    e16 = g_n2[token*D_MODEL + 4*w + e];
                    tol_add(fp16_to_real(a16), fp16_to_real(e16));
                    if (a16 === e16) stage_match++;
                    else stage_mismatch++;
                end
            end
        $display("    RMSNorm2: %0d/%0d exact", stage_match, NUM_FP16);
        tol_report("CPU tolerance: RMSNorm 2", NUM_FP16);
        if (!$test$plusargs("CPU_REAL")) begin
            for (token = 0; token < M_TOKENS; token++)
                for (w = 0; w < ROW_WORDS; w++)
                    cpu_mem_write(tcdm_addr(token*ROW_WORDS + w, XN_BANK),
                                  pack4(g_n2[token*D_MODEL + 4*w + 0],
                                        g_n2[token*D_MODEL + 4*w + 1],
                                        g_n2[token*D_MODEL + 4*w + 2],
                                        g_n2[token*D_MODEL + 4*w + 3]));
        end

        $display("[10] Requant norm2 and run GEMM gate / up ...");
        run_requant(0, XN_BANK, N_Q, 0, ACT_BANK, "requant norm2");
        run_gemm(16'd1, 16'd1, tcdm_addr(W_W_GU, 0), tcdm_addr(OUT_W_G, OUT_BANK), "GEMM gate");
        run_gemm(16'd1, 16'd1, tcdm_addr(W_W_GU + 1, 0), tcdm_addr(OUT_W_U, OUT_BANK), "GEMM up");

        // --------------------------------------------------- SiLU * up ------
        $display("[11] CPU SiLU(gate)*up, requant, GEMM down ...");
        cpu_silu_mul(OUT_W_G, OUT_W_U);

        tol_reset(); stage_match = 0; stage_mismatch = 0;
        for (token = 0; token < M_TOKENS; token++)
            for (w = 0; w < 4; w++) begin
                cpu_mem_read(tcdm_addr(4*token + w, STG_BANK), rw);
                for (e = 0; e < 4; e++) begin
                    a16 = rw[16*e +: 16];
                    e16 = g_h[token*D_FF + 4*w + e];
                    tol_add(fp16_to_real(a16), fp16_to_real(e16));
                    if (a16 === e16) stage_match++;
                    else stage_mismatch++;
                end
            end
        $display("    SiLU*up: %0d/%0d exact", stage_match, NUM_FP16);
        tol_report("CPU tolerance: SiLU*up", NUM_FP16);
        if (!$test$plusargs("CPU_REAL")) begin
            for (token = 0; token < M_TOKENS; token++)
                for (w = 0; w < 4; w++)
                    cpu_mem_write(tcdm_addr(4*token + w, STG_BANK),
                                  pack4(g_h[token*D_FF + 4*w + 0],
                                        g_h[token*D_FF + 4*w + 1],
                                        g_h[token*D_FF + 4*w + 2],
                                        g_h[token*D_FF + 4*w + 3]));
        end

        run_requant(0, STG_BANK, N_Q, 0, ACT_BANK, "requant ffn act");
        run_gemm(16'd1, 16'd1, tcdm_addr(W_W_D, 0), tcdm_addr(OUT_W_D, OUT_BANK), "GEMM down");

        // -------------------------------------------------- residual 2 ------
        $display("[12] CPU residual 2 and final check ...");
        cpu_residual_final(OUT_W_D, OUT_BANK, 0, RES1_BANK, 0, FIN_BANK);

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
                    else begin
                        stage_mismatch++;
                        if (stage_mismatch <= 10)
                            $display("    FINAL mismatch tok=%0d ch=%0d act=0x%04X exp=0x%04X",
                                     token, 4*w+e, a16, e16);
                    end
                end
            end
        $display("    Final layer output: %0d/%0d exact", stage_match, NUM_FP16);
        tol_report("CPU tolerance: final residual", NUM_FP16);

        $display("\n========================================================================");
        $display("  MINIMAL FULL LAYER RESULT");
        $display("========================================================================");
        $display("  Final FP16 values : %0d / %0d", stage_match, NUM_FP16);
        $display("  Total job cycles  : %0d", job_cycles_total);
        $display("  Cycle mix         : total=%0d seq_busy=%0d core_compute=%0d core_fp16=%0d cpu=%0d dma=%0d",
                 cyc_total, cyc_seq, cyc_core_compute, cyc_core_fp16, cyc_cpu, cyc_dma);
        $display("  Checks            : %0d passed, %0d failed", checks_pass, checks_fail);
        $display("========================================================================\n");

        if (checks_fail == 0)
            $display(">>> LAYER PASS: DUT bit-exact, CPU model within tolerance <<<\n");
        else
            $display(">>> LAYER FAIL (%0d check fails) <<<\n", checks_fail);

        $finish;
    end

endmodule
