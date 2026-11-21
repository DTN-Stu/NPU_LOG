// =============================================================================
// tb_llama3_1b_layer.sv
//
// Full-scale NousResearch/Llama-3.2-1B end-to-end prefill layer on
// snax_llm_top_clean + 8-port snax_simple_dma + 512-bit External DRAM:
//   M_TOKENS = 16, D_MODEL = 2048, D_FF = 8192,
//   H_Q = 32, H_KV = 8 (GQA 4:1, gqa_group_m1 = 3), D_K = D_V = 64,
//   All 7 weight matrices (~34.2 MB FP4+Log8) streamed from DRAM via 4-slot x 128 KB
//   ring buffer; W_down (K=8192) uses 2x4096 Output-Stationary K-Tiling.
// =============================================================================
`timescale 1ns/1ps

module tb_llama3_1b_layer;

    // --------------------------------------------------------- Llama-3.2-1B --
    `ifndef M_TOKENS
        `define M_TOKENS 16
    `endif
    `ifndef N_KV
        `define N_KV `M_TOKENS
    `endif
    `ifndef DATA_DIR
        `ifdef DATA_DIR_PREFILL
            `define DATA_DIR "sim_llama3_1b_prefill"
        `elsif DECODE
            `define DATA_DIR "sim_llama3_1b_decode"
        `else
            `define DATA_DIR "sim_llama3_1b"
        `endif
    `endif
    localparam int N_KV           = `N_KV;
    localparam int M_TOKENS       = `M_TOKENS;
    localparam int M_BATCH        = 16;
    localparam int M_CHUNKS       = M_TOKENS / M_BATCH;
    localparam int D_MODEL        = 2048;
    localparam int D_FF           = 8192;
    localparam int H_Q            = 32;
    localparam int H_KV           = 8;
    localparam int GQA_M1         = (H_Q / H_KV) - 1; // 3
    localparam int D_K            = 64;
    localparam int D_V            = 64;
    `ifdef DECODE
        localparam bit IS_DECODE = 1'b1;
        localparam int M_Q       = 1;
        localparam int Q_TILES   = 1;
        localparam int EVAL_TOKENS = 1;
    `else
        localparam bit IS_DECODE = 1'b0;
        localparam int M_Q       = 8;
        localparam int Q_TILES   = M_TOKENS / M_Q;
        localparam int EVAL_TOKENS = M_TOKENS;
    `endif
    localparam int KV_TILES       = N_KV / 16;        // 1 (or N_KV/16)
    localparam int KV_TILES_PAST  = KV_TILES - 1;
    localparam int TB_BANK_DEPTH  = 1024; // True 288 KB TCDM (32 banks x 1024 words x 9B) for BOTH Prefill & Decode
    localparam int KV_DRAM_BEATS  = H_KV * 4 * KV_TILES * 12; // 768B (12 beats) per (hkv, dvi, kj) tile
    localparam int PAST_KV_WORDS  = (!IS_DECODE && N_KV > 16) ? (H_KV * KV_TILES_PAST * 4 * 16) : 1;
    localparam int DV_TILES       = D_V / 16;         // 4
    localparam int K_PACKETS_QKV  = D_MODEL / 16;     // 128
    localparam int K_PACKETS_QK   = D_K / 16;         // 4
    localparam int K_PACKETS_DOWN = D_FF / 16;        // 512
    localparam int K_TILE_DOWN    = 256;              // 256 packets = 4096 per K-tile
    localparam int N_TILES_MODEL  = D_MODEL / 16;     // 128
    localparam int N_TILES_KV     = (H_KV * D_K) / 16;// 32
    localparam int N_TILES_FF     = D_FF / 16;        // 512
    localparam int CHUNK_NT       = 16;               // 32 chunks x 16 = 512 N-tiles
    localparam int N_CHUNKS       = N_TILES_FF / CHUNK_NT; // 32

    localparam int NUM_FP16       = M_TOKENS * D_MODEL; // 32,768
    localparam int ROW_WORDS      = D_MODEL / 4;        // 512
    localparam int NUM_VEC_XN     = EVAL_TOKENS * K_PACKETS_QKV; // 2,048 (prefill) or 128 (decode)
    localparam int NUM_PRELOAD    = IS_DECODE ? 2048 : (2 * M_TOKENS * 512 + 1024);

    localparam int CLK_PERIOD     = 10;
    localparam int DRAM_LATENCY   = 6;
    localparam int JOB_TIMEOUT    = 5_000_000;

    // ------------------------------------------------------------- TCDM map --
    // Banks 0..15: 4-slot x 128 KB Ring Buffer (words 0..1023)
    //              and non-DMA Attention K[0..7] (words 0..31), V[0..7] (words 128..159)
    localparam int ACT_BANK       = 16;
    localparam int OUT_BANK       = 17;
    localparam int O_ACT_BANK     = 20;
    localparam int G_STG_BANK     = 21;
    localparam int FF1_BANK       = 22;
    localparam int X_BANK         = 24;
    localparam int XN_BANK        = 25;
    localparam int RES1_BANK      = 26;
    localparam int FIN_BANK       = 27;
    localparam int GAMMA_BANK     = 28;
    localparam int ROPE_BANK      = 29;
    localparam int QPKT_BANK      = 30;

    localparam int GAMMA1_WORD    = 0;
    localparam int GAMMA2_WORD    = 512;
    localparam int V_BASE_BLOCK   = ((H_Q * KV_TILES * K_PACKETS_QK) > 16) ? (H_Q * KV_TILES * K_PACKETS_QK) : 16;

    // --------------------------------------------------------- DRAM offsets --
    localparam logic [31:0] DRAM_WQ = 32'h0000_0000;
    localparam logic [31:0] DRAM_WK = 32'h0024_0000;
    localparam logic [31:0] DRAM_WV = 32'h002D_0000;
    localparam logic [31:0] DRAM_WO = 32'h0036_0000;
    localparam logic [31:0] DRAM_WG = 32'h005A_0000;
    localparam logic [31:0] DRAM_WU = 32'h00EA_0000;
    localparam logic [31:0] DRAM_WD  = 32'h017A_0000;
    localparam logic [31:0] DRAM_KC  = 32'h0210_0000; // K-cache writeback (4,608 B)
    localparam logic [31:0] DRAM_VC  = 32'h0210_2000; // V-cache writeback (4,608 B)
    localparam logic [31:0] DRAM_OUT  = 32'h0211_0000; // Final Output writeback
    localparam logic [31:0] DRAM_KV   = 32'h0220_0000; // Ring-Buffer KV-cache in DRAM (768 B/tile)
    localparam logic [31:0] DRAM_X    = 32'h0230_0000; // Swap X
    localparam logic [31:0] DRAM_RES1 = 32'h0238_0000; // Swap RES1
    localparam logic [31:0] DRAM_QPKT = 32'h0240_0000; // Swap QPKT
    localparam logic [31:0] DRAM_DOWN = 32'h0248_0000; // Swap Down

    // ------------------------------------------------------------- CSR map --
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
    localparam logic [11:0] A_WRAP_CFG  = 12'd52;
    localparam logic [11:0] A_START     = 12'd53;
    localparam logic [11:0] A_RO_ID     = 12'd61;
    localparam logic [31:0] CSR_ID_VALUE = 32'h534E_4146;

    // --------------------------------------------------------------- DUT -----
    logic clk, rst_n;
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    initial begin
        #(250_000_000); // 25M cycles global safety timeout
        $display("\n!!! GLOBAL TIMEOUT !!!\n");
        $finish;
    end

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

    logic [7:0]        dma_tcdm_req_valid, dma_tcdm_req_ready, dma_tcdm_req_write, dma_tcdm_rsp_valid;
    logic [7:0][31:0]  dma_tcdm_req_addr;
    logic [7:0][71:0]  dma_tcdm_req_wdata, dma_tcdm_rsp_rdata;
    logic [7:0][8:0]   dma_tcdm_req_strb;

    logic        dma_desc_valid, dma_desc_ready, dma_desc_done;
    logic [1:0]  dma_desc_dir;
    logic [31:0] dma_desc_dram_addr, dma_desc_tcdm_addr, dma_desc_size;

    logic         dram_req_valid, dram_req_ready, dram_req_write, dram_rsp_valid;
    logic [31:0]  dram_req_addr;
    logic [511:0] dram_req_wdata, dram_rsp_rdata;
    logic [63:0]  dram_req_strb;

    logic [31:0]  stat_rd_beats, stat_wr_beats, stat_busy_cycles;
    logic         busy, done_irq;

    snax_llm_top_clean #(
        .BANK_DEPTH (TB_BANK_DEPTH)
    ) dut (
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
        .dma_aux_mem_req_valid_i (dma_tcdm_req_valid[7:1]),
        .dma_aux_mem_req_ready_o (dma_tcdm_req_ready[7:1]),
        .dma_aux_mem_req_addr_i  (dma_tcdm_req_addr[7:1]),
        .dma_aux_mem_req_write_i (dma_tcdm_req_write[7:1]),
        .dma_aux_mem_req_wdata_i (dma_tcdm_req_wdata[7:1]),
        .dma_aux_mem_req_strb_i  (dma_tcdm_req_strb[7:1]),
        .dma_aux_mem_rsp_valid_o (dma_tcdm_rsp_valid[7:1]),
        .dma_aux_mem_rsp_rdata_o (dma_tcdm_rsp_rdata[7:1]),
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

    snax_simple_dma #(.TCDM_PORTS(8)) u_dma (
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

    // 512-bit External DRAM (64 MB = 1,048,576 x 64B)
    localparam int DRAM_WORDS = 1048576;
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
                if (dram_req_write) begin
                    dram_mem[dram_req_addr[25:6]] <= dram_req_wdata;
                end else begin
                    lat_vld[0]  <= 1'b1;
                    lat_data[0] <= dram_mem[dram_req_addr[25:6]];
                end
            end
        end
    end

    // ----------------------------------------------------- Preload & Golden --
    logic [31:0] pre_addr     [0:NUM_PRELOAD-1];
    logic [71:0] pre_data     [0:NUM_PRELOAD-1];
    logic [71:0] rope_mbatch  [0:M_CHUNKS*256-1];
    logic [71:0] o_act_buf    [0:M_TOKENS-1][0:127];
    logic [71:0] k_past_mem   [0:PAST_KV_WORDS-1];
    logic [71:0] v_past_mem   [0:PAST_KV_WORDS-1];
    logic [15:0] g_xnorm      [0:NUM_FP16-1];
    logic [15:0] g_res1       [0:NUM_FP16-1];
    logic [15:0] g_n2         [0:NUM_FP16-1];
    logic [15:0] g_res1_fq    [0:NUM_FP16-1];
    logic [15:0] g_n2_fq      [0:NUM_FP16-1];
    logic [15:0] g_final_fq   [0:NUM_FP16-1];
    logic [15:0] g_final_fp16 [0:NUM_FP16-1];
    logic [31:0] g_final_fp32 [0:NUM_FP16-1];

    // -------------------------------------------------------------- Helpers --
    function automatic logic [31:0] tcdm_addr(input int word, input int bank);
        tcdm_addr = (32'(word) << 9) | (32'(bank) << 4);
    endfunction

    function automatic real fp16_to_real(input logic [15:0] h);
        logic sign;
        int e, mf;
        real v;
        sign = h[15];
        e    = int'(h[14:10]);
        mf   = int'(h[9:0]);
        if (e == 0 && mf == 0) return 0.0;
        if (e == 31) return sign ? -65504.0 : 65504.0;
        if (e == 0)  v = real'(mf) / 16777216.0;
        else         v = (1.0 + real'(mf) / 1024.0) * (2.0 ** (e - 15));
        return sign ? -v : v;
    endfunction

    function automatic real fp32_to_real(input logic [31:0] f);
        logic sign;
        int e;
        logic [22:0] mf;
        real v;
        sign = f[31];
        e    = int'(f[30:23]);
        mf   = f[22:0];
        if (e == 0 && mf == 0) return 0.0;
        if (e == 255) return sign ? -1.0e38 : 1.0e38;
        if (e == 0)  v = real'(mf) / 8388608.0 * (2.0 ** -126);
        else         v = (1.0 + real'(mf) / 8388608.0) * (2.0 ** (e - 127));
        return sign ? -v : v;
    endfunction

    task automatic cpu_csr_write(input logic [11:0] a, input logic [31:0] d);
        @(negedge clk);
        csr_req_valid = 1'b1; csr_req_write = 1'b1;
        csr_req_addr  = a;    csr_req_data  = d;
        @(posedge clk);
        while (!csr_req_ready) @(posedge clk);
        @(negedge clk);
        csr_req_valid = 1'b0; csr_req_write = 1'b0;
    endtask

    task automatic cpu_csr_read(input logic [11:0] a, output logic [31:0] d);
        @(negedge clk);
        csr_req_valid = 1'b1; csr_req_write = 1'b0;
        csr_req_addr  = a;
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
        cpu_mem_req_addr  = a;    cpu_mem_req_wdata = d; cpu_mem_req_strb = 9'h1FF;
        @(posedge clk);
        while (!cpu_mem_req_ready) @(posedge clk);
        @(negedge clk);
        cpu_mem_req_valid = 1'b0; cpu_mem_req_write = 1'b0; cpu_mem_req_strb = 9'h000;
    endtask

    task automatic cpu_mem_read(input logic [31:0] a, output logic [71:0] d);
        @(negedge clk);
        cpu_mem_req_valid = 1'b1; cpu_mem_req_write = 1'b0;
        cpu_mem_req_addr  = a;    cpu_mem_req_strb  = 9'h000;
        @(posedge clk);
        while (!cpu_mem_req_ready) @(posedge clk);
        @(negedge clk);
        cpu_mem_req_valid = 1'b0;
        while (!cpu_mem_rsp_valid) @(posedge clk);
        d = cpu_mem_rsp_rdata;
    endtask

    // -------------------------------------------------------- Cycle Counters --
    longint cyc_total, cyc_seq, cyc_sfu, cyc_core_compute, cyc_dma, job_cycles_total;
    int checks_pass, checks_fail;

    always @(posedge clk) begin
        if (rst_n) begin
            cyc_total <= cyc_total + 1;
            if (dut.seq_busy)          cyc_seq <= cyc_seq + 1;
            if (dut.sfu_busy)          cyc_sfu <= cyc_sfu + 1;
            if (|dut.lane_valid)       cyc_core_compute <= cyc_core_compute + 1;
            if (|dma_tcdm_req_valid)   cyc_dma <= cyc_dma + 1;
        end
    end

    task automatic check(input bit cond, input string msg);
        if (cond) checks_pass++;
        else begin
            checks_fail++;
            $display("    [FAIL] %s", msg);
        end
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
        job_cycles_total = job_cycles_total + cycle_cnt;
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
                    sum_ref2 += v_g * v_g;
                    sum_act2 += v_a * v_a;
                    sum_err2 += (v_a - v_g) * (v_a - v_g);
                    dot_p    += v_a * v_g;
                end
            end
        sqnr  = (sum_err2 > 0.0) ? (10.0 * $log10(sum_ref2 / sum_err2)) : 999.0;
        cos_s = dot_p / ($sqrt(sum_act2) * $sqrt(sum_ref2));
        $display("    [METRIC] %-18s : SQNR = %0.2f dB, CosineSim = %0.6f", name, sqnr, cos_s);
        check(cos_s >= min_cos, name);
    endtask

    task automatic check_sqnr_rows_16bank(input int word_off, input logic [15:0] gold [0:NUM_FP16-1],
                                          input string name, input real min_cos, input int tok_off = 0);
        real sum_ref2, sum_err2, dot_p, sum_act2, v_a, v_g, sqnr, cos_s;
        int n_tok;
        logic [71:0] rw;
        n_tok = IS_DECODE ? 1 : M_BATCH;
        sum_ref2 = 0.0; sum_err2 = 0.0; dot_p = 0.0; sum_act2 = 0.0;
        for (int t = 0; t < n_tok; t++)
            for (int w = 0; w < ROW_WORDS; w++) begin
                cpu_mem_read(tcdm_addr(word_off + w, 16 + t), rw);
                for (int e = 0; e < 4; e++) begin
                    v_a = fp16_to_real(rw[16*e +: 16]);
                    v_g = fp16_to_real(gold[(tok_off + t)*D_MODEL + 4*w + e]);
                    sum_ref2 += v_g * v_g;
                    sum_act2 += v_a * v_a;
                    sum_err2 += (v_a - v_g) * (v_a - v_g);
                    dot_p    += v_a * v_g;
                end
            end
        sqnr  = (sum_err2 > 0.0) ? (10.0 * $log10(sum_ref2 / sum_err2)) : 999.0;
        cos_s = dot_p / ($sqrt(sum_act2) * $sqrt(sum_ref2));
        $display("    [METRIC] %-18s : SQNR = %0.2f dB, CosineSim = %0.6f", name, sqnr, cos_s);
        check(cos_s >= min_cos, name);
    endtask

    // --------------------------------------------------------- Job Launchers --
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
        cpu_csr_write(A_WO_MODE,    out_fp4 ? 32'd0 : 32'd1);
        cpu_csr_write(A_WO_BASE,    o_base);
        cpu_csr_write(A_WO_NV,      32'(k_vecs));
        cpu_csr_write(A_WO_LS,      {o_g_16b[15:0], o_t_16b[15:0]});
        cpu_csr_write(A_START,      32'h0000_0003);
        repeat (4) @(posedge clk);
        guard_s = 0;
        while (dut.sfu_seq_busy || dut.sfu_busy || dut.wo_busy || dut.u_read_fp16.busy_o || dut.u_read_fp16_b.busy_o) begin
            @(posedge clk); guard_s++;
            if (guard_s > JOB_TIMEOUT) $fatal(1, "SFU sequencer timeout");
        end
        cpu_csr_write(A_SFU_CTRL, 32'h0);
    endtask

    task automatic sfu_rms_tokens(input int x_bank, input int gamma_word, input int out_bank);
        run_sfu_seq(3'd2, EVAL_TOKENS, 1, K_PACKETS_QKV,
                    tcdm_addr(0, x_bank),          ROW_WORDS * 32, 0,
                    tcdm_addr(gamma_word, GAMMA_BANK), 0,          0,
                    tcdm_addr(0, out_bank),        ROW_WORDS * 32, 0,
                    32'h3A000000); // 1/2048
    endtask

    task automatic sfu_residual_rows(input int o_bank, input int x_bank, input int r_bank);
        run_sfu_seq(3'd1, EVAL_TOKENS, N_TILES_MODEL, 1,
                    tcdm_addr(0, o_bank), 4 * 32,         EVAL_TOKENS * 4 * 32,
                    tcdm_addr(0, x_bank), ROW_WORDS * 32, 4 * 32,
                    tcdm_addr(0, r_bank), ROW_WORDS * 32, 4 * 32,
                    32'h3F800000);
    endtask

    task automatic sfu_silu_chunk(input int chunk);
        run_sfu_seq(3'd4, EVAL_TOKENS, CHUNK_NT, 1,
                    tcdm_addr(0, G_STG_BANK), 4 * 32, EVAL_TOKENS * 4 * 32,
                    tcdm_addr(0, OUT_BANK),   4 * 32, EVAL_TOKENS * 4 * 32,
                    tcdm_addr(chunk * CHUNK_NT, FF1_BANK), K_PACKETS_DOWN * 32, 32,
                    32'h3F800000, 1'b1);
    endtask

    task automatic run_requant_addr(input logic [31:0] rd_base, input int nvec, input logic [31:0] wo_base, input string name);
        cpu_csr_write(A_CTRL,     32'h0);
        cpu_csr_write(A_SEQ_CTRL, 32'h3);
        cpu_csr_write(A_RD_BASE,  rd_base);
        cpu_csr_write(A_RD_NV,    nvec);
        cpu_csr_write(A_RD_LS,    32'h200);
        cpu_csr_write(A_WO_BASE,  wo_base);
        cpu_csr_write(A_START,    32'h0000_0003);
        wait_job(name);
    endtask

    task automatic run_requant_wrap(input logic [31:0] rd_base, input int nvec, input logic [31:0] wo_base,
                                    input int rd_wrap_words_m1, input int wo_wrap_words_m1, input string name);
        cpu_csr_write(A_CTRL,     32'h0);
        cpu_csr_write(A_WRAP_CFG, (32'(rd_wrap_words_m1) << 11) | 32'(wo_wrap_words_m1));
        cpu_csr_write(A_SEQ_CTRL, (32'd1 << 27) | (32'd1 << 26) | 32'h3);
        cpu_csr_write(A_RD_BASE,  rd_base);
        cpu_csr_write(A_RD_NV,    nvec);
        cpu_csr_write(A_RD_LS,    32'h200);
        cpu_csr_write(A_WO_BASE,  wo_base);
        cpu_csr_write(A_START,    32'h0000_0003);
        wait_job(name);
    endtask

    task automatic run_requant(input int rd_bank, input int nvec, input int wo_bank, input string name);
        run_requant_addr(tcdm_addr(0, rd_bank), nvec, tcdm_addr(0, wo_bank), name);
    endtask

    task automatic run_fused_qk_rope_dma_ex(input int n_tiles, input logic [31:0] dram_base,
                                            input logic [31:0] q_base, input logic [31:0] out_base,
                                            input logic [31:0] rope_base, input bit act_16bank, input string name,
                                            input bit wrap_en = 1'b0, input int wo_wrap_m1 = 0, input int rd_b_wrap_m1 = 0);
        int nvec;
        nvec = (IS_DECODE ? 1 : M_BATCH) * n_tiles;
        cpu_csr_write(A_CTRL,       32'h0);
        cpu_csr_write(A_WRAP_CFG,   (32'(rd_b_wrap_m1) << 22) | 32'(wo_wrap_m1));
        cpu_csr_write(A_SEQ_CTRL,   (wrap_en ? ((32'd1 << 28) | (32'd1 << 26)) : 32'd0) |
                                    (act_16bank ? (32'd1 << 24) : 32'd0) | (IS_DECODE ? 32'h98 : 32'h18));
        cpu_csr_write(A_GEMM_NT,    n_tiles);
        cpu_csr_write(A_GEMM_MT,    32'd1);
        cpu_csr_write(A_GEMM_KB,    K_PACKETS_QKV);
        cpu_csr_write(A_RING_BASE,  32'h0);
        cpu_csr_write(A_DRAM_KV,    dram_base);
        cpu_csr_write(A_TILE_ST,    K_PACKETS_QKV * 144);
        cpu_csr_write(A_Q_BASE,     q_base);
        cpu_csr_write(A_OUT_BASE,   out_base);
        cpu_csr_write(A_SFU_CTRL,   32'h0020_0000 | (32'(nvec) << 5) | (32'd3 << 2) | 32'h1);
        cpu_csr_write(A_SFU_B_BASE, rope_base);
        cpu_csr_write(A_SFU_B_NV,   32'(nvec));
        cpu_csr_write(A_SFU_B_LS,   32'h200);
        cpu_csr_write(A_START,      32'h0000_0183);
        wait_job(name);
        while (dut.sfu_busy || dut.rd_b_busy) @(posedge clk);
        cpu_csr_write(A_SFU_CTRL,   32'h0);
    endtask

    task automatic run_fused_qk_rope_dma(input int n_tiles, input logic [31:0] dram_base,
                                         input logic [31:0] q_base, input int out_bank, input string name);
        run_fused_qk_rope_dma_ex(n_tiles, dram_base, q_base, tcdm_addr(0, out_bank), tcdm_addr(0, ROPE_BANK), 1'b0, name);
    endtask

    task automatic run_gemm_v_dma_ex(input int n_tile, input int v_word, input logic [31:0] ring_base,
                                     input logic [31:0] q_base, input bit act_16bank, input string name);
        cpu_csr_write(A_CTRL,      32'h0);
        cpu_csr_write(A_SEQ_CTRL,  (act_16bank ? (32'd1 << 24) : 32'd0) | 32'h38); // op 00, out_mode=11 (fused V), enable_dma=1
        cpu_csr_write(A_GEMM_NT,   32'd1);
        cpu_csr_write(A_GEMM_MT,   32'd1);
        cpu_csr_write(A_GEMM_KB,   K_PACKETS_QKV);
        cpu_csr_write(A_RING_BASE, ring_base);
        cpu_csr_write(A_DRAM_KV,   DRAM_WV + 32'(n_tile) * 32'(K_PACKETS_QKV) * 32'd144);
        cpu_csr_write(A_TILE_ST,   K_PACKETS_QKV * 144);
        cpu_csr_write(A_Q_BASE,    q_base);
        cpu_csr_write(A_V_BASE,    tcdm_addr(v_word, 0));
        cpu_csr_write(A_V_NB,      32'd1);
        cpu_csr_write(A_START,     32'h0000_0003);
        wait_job(name);
    endtask

    task automatic run_gemm_v_dma(input int n_tile, input int v_word, input string name);
        run_gemm_v_dma_ex(n_tile, v_word, tcdm_addr(256, 0), tcdm_addr(0, ACT_BANK), 1'b0, name);
    endtask

    task automatic run_gemm_dma_ex(input int n_tiles, input int k_blks_per_tile, input int k_tiles_m1,
                                   input logic [31:0] dram_base, input int tile_stride_bytes,
                                   input logic [31:0] q_base, input logic [31:0] out_base,
                                   input bit act_16bank, input string name,
                                   input bit out_16bank = 1'b0, input logic [1:0] out_mode = 2'd0);
        cpu_csr_write(A_CTRL,      32'h0);
        cpu_csr_write(A_SEQ_CTRL,  (out_16bank ? (32'd1 << 25) : 32'd0) |
                                   (act_16bank ? (32'd1 << 24) : 32'd0) |
                                   (32'(k_tiles_m1) << 8) |
                                   ({26'd0, out_mode, 4'b1000}) |
                                   (IS_DECODE ? 32'h80 : 32'h0));
        cpu_csr_write(A_GEMM_NT,   n_tiles);
        cpu_csr_write(A_GEMM_MT,   32'd1);
        cpu_csr_write(A_GEMM_KB,   k_blks_per_tile);
        cpu_csr_write(A_RING_BASE, 32'h0);
        cpu_csr_write(A_DRAM_KV,   dram_base);
        cpu_csr_write(A_TILE_ST,   tile_stride_bytes);
        cpu_csr_write(A_Q_BASE,    q_base);
        cpu_csr_write(A_OUT_BASE,  out_base);
        cpu_csr_write(A_START,     32'h0000_0003);
        wait_job(name);
    endtask

    task automatic run_gemm_dma(input int n_tiles, input int k_blks_per_tile, input int k_tiles_m1,
                                input logic [31:0] dram_base, input int tile_stride_bytes,
                                input logic [31:0] q_base, input logic [31:0] out_base, input string name);
        run_gemm_dma_ex(n_tiles, k_blks_per_tile, k_tiles_m1, dram_base, tile_stride_bytes, q_base, out_base, 1'b0, name);
    endtask

    task automatic run_attn_gqa_ex(input int h_cnt, input logic [31:0] dram_kv_base,
                                   input logic [31:0] q_base, input logic [31:0] out_base, input string name);
        cpu_csr_write(A_CTRL,      32'h0);
        cpu_csr_write(A_SEQ_CTRL,  (32'(GQA_M1) << 16) | (IS_DECODE ? 32'h29 : 32'h2D)); // op 01, enable_dma=1, causal=(!IS_DECODE), 4D FP4 out, GQA 4:1
        cpu_csr_write(A_ATTN_H,    h_cnt);
        cpu_csr_write(A_ATTN_QT,   Q_TILES);
        cpu_csr_write(A_ATTN_MQ,   M_Q);
        cpu_csr_write(A_ATTN_DV,   DV_TILES);
        cpu_csr_write(A_ATTN_KV,   KV_TILES);
        cpu_csr_write(A_ATTN_KB,   K_PACKETS_QK);
        cpu_csr_write(A_RING_BASE, 32'h0);
        cpu_csr_write(A_DRAM_KV,   dram_kv_base);
        cpu_csr_write(A_TILE_ST,   32'd768);
        cpu_csr_write(A_Q_BASE,    q_base);
        cpu_csr_write(A_OUT_BASE,  out_base);
        cpu_csr_write(A_START,     32'h0000_0003);
        wait_job(name);
    endtask

    task automatic run_attn_gqa(input string name);
        run_attn_gqa_ex(H_Q, DRAM_KV, tcdm_addr(0, QPKT_BANK), tcdm_addr(0, O_ACT_BANK), name);
    endtask

    task automatic run_dma_writeback(input logic [1:0] dir, input logic [31:0] tcdm_base,
                                     input logic [31:0] dram_base, input int size_bytes, input string name);
        cpu_csr_write(A_CTRL,      32'h0);
        cpu_csr_write(A_SEQ_CTRL,  32'h40 | (32'(dir) << 4)); // bit[6]=dma_wb, bits[5:4]=dir
        cpu_csr_write(A_OUT_BASE,  tcdm_base);
        cpu_csr_write(A_DRAM_KV,   dram_base);
        cpu_csr_write(A_TILE_ST,   size_bytes);
        cpu_csr_write(A_START,     32'h0000_0003);
        wait_job(name);
    endtask

    task automatic verify_dram_vs_tcdm(input logic linear_mode, input int tcdm_word0, input int tcdm_bank0,
                                       input logic [31:0] dram_base, input int n_words, input string name);
        int bad;
        int base_beat;
        logic [71:0] exp_w, act_w;
        int cur_w, cur_b;
        bad = 0;
        base_beat = dram_base >> 6;
        cur_w = tcdm_word0;
        cur_b = tcdm_bank0;
        for (int w = 0; w < n_words; w++) begin
            cpu_mem_read(tcdm_addr(cur_w, cur_b), exp_w);
            for (int bit_i = 0; bit_i < 72; bit_i++) begin
                int gbit;
                gbit = w * 72 + bit_i;
                act_w[bit_i] = dram_mem[base_beat + (gbit >> 9)][gbit & 511];
            end
            if (act_w !== exp_w) begin
                if (bad < 5)
                    $display("    [WB MISMATCH] %s word=%0d dram=%018X tcdm=%018X", name, w, act_w, exp_w);
                bad++;
            end
            if (linear_mode) begin
                cur_w++;
            end else begin
                if (cur_b == 15 || cur_b == 31) begin
                    cur_b = (cur_b == 31) ? 16 : 0;
                    cur_w++;
                end else begin
                    cur_b++;
                end
            end
        end
        $display("    [DMA WB] %-22s : %0d/%0d words bit-exact in DRAM", name, n_words - bad, n_words);
        check(bad == 0, name);
    endtask

    // =========================================================================
    // Main Llama-3.2-1B End-to-End Prefill Layer Sequence
    // =========================================================================
    initial begin
        longint t0;
        logic [31:0] rd32;
        real sum_ref2, sum_err2, dot_p, sum_rtl2, v_rtl, v_fp32, sqnr, cos_s;
        logic [71:0] rw;

        if ($test$plusargs("DUMP")) begin
            $dumpfile("tb_llama3_1b_layer.fst");
            $dumpvars(0, dut);
        end

        csr_req_valid = 0; csr_req_write = 0; csr_req_addr = 0; csr_req_data = 0; csr_rsp_ready = 1;
        cpu_mem_req_valid = 0; cpu_mem_req_write = 0; cpu_mem_req_addr = 0;
        cpu_mem_req_wdata = 0; cpu_mem_req_strb = 0;
        checks_pass = 0; checks_fail = 0;
        cyc_total = 0; cyc_seq = 0; cyc_sfu = 0; cyc_core_compute = 0; cyc_dma = 0; job_cycles_total = 0;

        $display("Loading Llama-3.2-1B DRAM weights (34.2 MB) & TCDM preload...");
        $readmemh({`DATA_DIR, "/dram_weights.hex"},      dram_mem);
        if (IS_DECODE)
            $readmemh({`DATA_DIR, "/kv_dram.hex"},       dram_mem, DRAM_KV >> 6, (DRAM_KV >> 6) + KV_DRAM_BEATS - 1);
        $readmemh({`DATA_DIR, "/tcdm_preload_addr.hex"}, pre_addr);
        $readmemh({`DATA_DIR, "/tcdm_preload_data.hex"}, pre_data);
        if (!IS_DECODE)
            $readmemh({`DATA_DIR, "/rope_mbatch.hex"},   rope_mbatch);
        $readmemh({`DATA_DIR, "/g_xnorm.hex"},           g_xnorm);
        $readmemh({`DATA_DIR, "/g_res1.hex"},            g_res1);
        $readmemh({`DATA_DIR, "/g_n2.hex"},              g_n2);
        $readmemh({`DATA_DIR, "/g_final_fp16.hex"},      g_final_fp16);
        $readmemh({`DATA_DIR, "/g_final_fp32.hex"},      g_final_fp32);
        $readmemh({`DATA_DIR, "/g_res1_fq.hex"},         g_res1_fq);
        $readmemh({`DATA_DIR, "/g_n2_fq.hex"},           g_n2_fq);
        $readmemh({`DATA_DIR, "/g_final_fq.hex"},        g_final_fq);

        rst_n = 1'b0;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        if (IS_DECODE) begin
            for (int i = 0; i < NUM_PRELOAD; i++)
                cpu_mem_write(pre_addr[i], pre_data[i]);
        end else if (M_CHUNKS == 1) begin
            // True 288 KB TCDM (BANK_DEPTH=1024): preload X into banks 16..31 (1 token/bank, words 0..511)
            // and gamma1 into bank 0 (words 0..511) while DMA is idle.
            for (int t = 0; t < M_BATCH; t++)
                for (int w = 0; w < ROW_WORDS; w++)
                    cpu_mem_write(tcdm_addr(w, 16 + t), pre_data[t*ROW_WORDS + w]);
            for (int w = 0; w < ROW_WORDS; w++)
                cpu_mem_write(tcdm_addr(w, 0), pre_data[M_TOKENS*ROW_WORDS + w]);
        end

        cpu_csr_read(A_RO_ID, rd32);
        check(rd32 == CSR_ID_VALUE, "CSR ID (SNAF)");

        $display("\n========================================================================");
        $display("  NousResearch/Llama-3.2-1B - FULL %s LAYER END-TO-END (288 KB TCDM)",
                 IS_DECODE ? "DECODE" : "PREFILL");
        $display("  M_TOKENS=%0d D_MODEL=%0d D_FF=%0d H_Q=%0d H_KV=%0d (GQA 4:1) D_K=%0d",
                 EVAL_TOKENS, D_MODEL, D_FF, H_Q, H_KV, D_K);
        $display("  Weights: ~34.2 MB in 512b DRAM -> 4-Port DMA -> 4x128KB TCDM Ring");
        $display("========================================================================\n");

        if (IS_DECODE) begin
        t0 = cyc_total;
        $display("[1/12] SFU RMSNorm 1 (M=1, D=2048)...");
        sfu_rms_tokens(X_BANK, GAMMA1_WORD, XN_BANK);
        check_sqnr_rows(XN_BANK, g_xnorm, "RMSNorm 1", 0.999);
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[2/12] HW Requant Xn -> ACT_BANK...");
        run_requant(XN_BANK, NUM_VEC_XN, ACT_BANK, "Requant Xn");
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[3/12] Fused Q (2048x2048, DMA) + RoPE + LogQ...");
        run_fused_qk_rope_dma(N_TILES_MODEL, DRAM_WQ, tcdm_addr(0, ACT_BANK), OUT_BANK, "Fused Q+RoPE");
        for (int h = 0; h < H_Q; h++)
            for (int t = 0; t < EVAL_TOKENS; t++)
                for (int kp = 0; kp < K_PACKETS_QK; kp++) begin
                    logic [71:0] qp;
                    cpu_mem_read(tcdm_addr((h*K_PACKETS_QK + kp)*EVAL_TOKENS + t, OUT_BANK), qp);
                    cpu_mem_write(tcdm_addr((h*(Q_TILES*M_Q) + t)*K_PACKETS_QK + kp, QPKT_BANK), qp);
                end
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[4/12] Fused K (512x2048, GQA H_KV=8, DMA) + RoPE + LogQ...");
        run_fused_qk_rope_dma(N_TILES_KV, DRAM_WK, tcdm_addr(0, ACT_BANK), OUT_BANK, "Fused K+RoPE");
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[5/12] Decode GEMM V (M=1, FP16 -> Bank 15 words 640..767) + HW V-Transpose (op=2'b10) + Update DRAM KV (N_KV=%0d)...", N_KV);
        begin
            logic [71:0] v_tail_fp16 [0:15 * N_TILES_KV * 4 - 1];
            $readmemh({`DATA_DIR, "/v_tail_past_fp16.hex"}, v_tail_fp16);
            for (int tok = 0; tok < 15; tok++)
                for (int w = 0; w < N_TILES_KV * 4; w++)
                    cpu_mem_write(tcdm_addr(640 + w, tok), v_tail_fp16[tok * (N_TILES_KV * 4) + w]);
        end
        // Decode GEMM V outputs FP16 (out_mode=00) into Bank 15 words 640..767 directly
        run_gemm_dma_ex(N_TILES_KV, K_PACKETS_QKV, 0, DRAM_WV, K_PACKETS_QKV * 144,
                        tcdm_addr(0, ACT_BANK), tcdm_addr(640, 15), 1'b0, "Decode GEMM V (FP16)", 1'b0, 2'd0);

        // Hardware Decode V-Transpose (op_mode=2'b10, act_bank_layout=1): Banks 0..15 words 640..767 -> Banks 0..15 words 128..159
        cpu_csr_write(A_CTRL,      32'h0);
        cpu_csr_write(A_RD_LS,     32'd0);
        cpu_csr_write(A_SEQ_CTRL,  (32'd1 << 24) | 32'h2);
        cpu_csr_write(A_Q_BASE,    tcdm_addr(640, 0));
        cpu_csr_write(A_V_BASE,    tcdm_addr(128, 0));
        cpu_csr_write(A_V_NB,      N_TILES_KV);
        cpu_csr_write(A_TILE_ST,   32'h200);
        cpu_csr_write(A_START,     32'h0000_0003);
        wait_job("Decode V-Transpose");

        // Update DRAM_KV tail tile (kj = KV_TILES_PAST) with new K row (l=15) and HW-transposed V packets (l=0..15)
        for (int hkv = 0; hkv < H_KV; hkv++) begin
            for (int kp = 0; kp < K_PACKETS_QK; kp++) begin
                logic [71:0] kp_pkt;
                cpu_mem_read(tcdm_addr(hkv*K_PACKETS_QK + kp, OUT_BANK), kp_pkt);
                for (int dvi = 0; dvi < DV_TILES; dvi++) begin
                    int tile_off, abs_byte;
                    tile_off = ((hkv * DV_TILES + dvi) * KV_TILES + KV_TILES_PAST) * 768;
                    abs_byte = tile_off + (kp * 16 + 15) * 9;
                    for (int b = 0; b < 9; b++) begin
                        int beat, boff;
                        beat = (DRAM_KV >> 6) + ((abs_byte + b) >> 6);
                        boff = (abs_byte + b) & 63;
                        dram_mem[beat][boff*8 +: 8] = kp_pkt[b*8 +: 8];
                    end
                end
            end
            for (int dvi = 0; dvi < DV_TILES; dvi++) begin
                int tile_off;
                tile_off = ((hkv * DV_TILES + dvi) * KV_TILES + KV_TILES_PAST) * 768 + 576;
                for (int l = 0; l < 16; l++) begin
                    logic [71:0] vp_pkt;
                    int abs_byte;
                    cpu_mem_read(tcdm_addr(128 + hkv * DV_TILES + dvi, l), vp_pkt);
                    abs_byte = tile_off + l * 9;
                    for (int b = 0; b < 9; b++) begin
                        int beat, boff;
                        beat = (DRAM_KV >> 6) + ((abs_byte + b) >> 6);
                        boff = (abs_byte + b) & 63;
                        dram_mem[beat][boff*8 +: 8] = vp_pkt[b*8 +: 8];
                    end
                end
            end
        end
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[6/12] HW FlashAttention (N_KV=%0d, GQA 4:1, H_Q=32, H_KV=8, D=64)...", N_KV);
        run_attn_gqa("FlashAttention GQA 4:1");
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[7/12] GEMM O (2048x2048, DMA)...");
        run_gemm_dma(N_TILES_MODEL, K_PACKETS_QKV, 0, DRAM_WO, K_PACKETS_QKV * 144,
                     tcdm_addr(0, O_ACT_BANK), tcdm_addr(0, OUT_BANK), "GEMM O");
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[8/12] SFU Residual 1 + RMSNorm 2...");
        sfu_residual_rows(OUT_BANK, X_BANK, RES1_BANK);
        check_sqnr_rows(RES1_BANK, g_res1_fq, "Residual 1 vs FQ", 0.96);
        check_sqnr_rows(RES1_BANK, g_res1,    "Residual 1 vs FP32", 0.92);
        sfu_rms_tokens(RES1_BANK, GAMMA2_WORD, XN_BANK);
        check_sqnr_rows(XN_BANK, g_n2_fq, "RMSNorm 2 vs FQ", 0.96);
        check_sqnr_rows(XN_BANK, g_n2,    "RMSNorm 2 vs FP32", 0.92);
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[9/12] HW Requant Norm2 -> ACT_BANK...");
        run_requant(XN_BANK, NUM_VEC_XN, ACT_BANK, "Requant Norm2");
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[10/12] FFN Gate & Up (2 x 8192x2048, DMA) + SiLU*Up (32 chunks)...");
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
        $display("[11/12] GEMM Down (2048x8192, DMA + 2x4096 K-Tiling)...");
        run_gemm_dma(N_TILES_MODEL, K_TILE_DOWN, 1,
                     DRAM_WD, K_PACKETS_DOWN * 144,
                     tcdm_addr(0, FF1_BANK), tcdm_addr(0, OUT_BANK), "GEMM Down");
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[12/12] SFU Residual 2 -> Final Output Check...");
        sfu_residual_rows(OUT_BANK, RES1_BANK, FIN_BANK);
        check_sqnr_rows(FIN_BANK, g_final_fq, "Final vs PyFQ", 0.91);

        sum_ref2 = 0.0; sum_err2 = 0.0; dot_p = 0.0; sum_rtl2 = 0.0;
        for (int t = 0; t < EVAL_TOKENS; t++)
            for (int w = 0; w < ROW_WORDS; w++) begin
                cpu_mem_read(tcdm_addr(t*ROW_WORDS + w, FIN_BANK), rw);
                for (int e = 0; e < 4; e++) begin
                    v_rtl  = fp16_to_real(rw[16*e +: 16]);
                    v_fp32 = fp32_to_real(g_final_fp32[t*D_MODEL + 4*w + e]);
                    sum_ref2 += v_fp32 * v_fp32;
                    sum_rtl2 += v_rtl  * v_rtl;
                    sum_err2 += (v_rtl - v_fp32) * (v_rtl - v_fp32);
                    dot_p    += v_rtl * v_fp32;
                end
            end
        sqnr  = (sum_err2 > 0.0) ? (10.0 * $log10(sum_ref2 / sum_err2)) : 999.0;
        cos_s = dot_p / ($sqrt(sum_rtl2) * $sqrt(sum_ref2));
        check(cos_s >= 0.82, "Final Layer Output vs FP32");

        run_dma_writeback(2'b11, tcdm_addr(0, FIN_BANK), DRAM_OUT, 512 * 9, "DMA WB Final Out");
        verify_dram_vs_tcdm(1'b1, 0, FIN_BANK, DRAM_OUT, 512, "Verify DRAM Final Out");
        end else if (M_CHUNKS == 1) begin
        // =====================================================================
        // PREFILL ON TRUE 288 KB TCDM (BANK_DEPTH = 1024)
        //   Banks 0..15 : 4-slot x 128KB Ring Buffer + V_BASE_BLOCK(128..159)
        //   Banks 16..31: 1 token/bank for X/XN/RES1/FIN (512w) & ACT/FF1 (128w/512w)
        // =====================================================================
        t0 = cyc_total;
        $display("[1/12] SFU RMSNorm 1 (M=16, D=2048, 288KB TCDM)...");
        run_sfu_seq(3'd2, 16, 1, 128,
                    tcdm_addr(0, 16),   1, 0,
                    tcdm_addr(0, 0),    0, 0,
                    tcdm_addr(512, 16), 1, 0, 32'h3A000000);
        check_sqnr_rows_16bank(512, g_xnorm, "RMSNorm 1", 0.999);
        run_dma_writeback(2'b01, tcdm_addr(0, 16), DRAM_X, 8192 * 9, "DMA Save X");
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[2/12] HW Requant Xn -> ACT (banks 16..31, words 0..127, 1 launch)...");
        run_requant_wrap(tcdm_addr(512, 16), M_TOKENS * 128, tcdm_addr(0, 16), 511, 127, "Requant Xn");
        for (int i = 0; i < 256; i++)
            cpu_mem_write(tcdm_addr(128 + i, 29), pre_data[9216 + i]);
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[3/12] Fused Q (2048x2048, DMA, 1 launch 32 heads) + RoPE + LogQ...");
        run_fused_qk_rope_dma_ex(128, DRAM_WQ, tcdm_addr(0, 16), tcdm_addr(128, 16),
                                 tcdm_addr(128, 29), 1'b1, "Fused Q+RoPE", 1'b1, 255, 255);
        for (int h = 0; h < H_Q; h++)
            for (int t = 0; t < M_TOKENS; t++)
                for (int kp = 0; kp < K_PACKETS_QK; kp++) begin
                    logic [71:0] qp;
                    int lin, idx, abs_byte;
                    lin = (h * K_PACKETS_QK + kp) * M_TOKENS + t;
                    cpu_mem_read(tcdm_addr(128 + (lin & 255), 16 + (lin >> 8)), qp);
                    idx = (h*M_TOKENS + t)*K_PACKETS_QK + kp;
                    abs_byte = idx * 9;
                    for (int b = 0; b < 9; b++) begin
                        int beat, boff;
                        beat = (DRAM_QPKT >> 6) + ((abs_byte + b) >> 6);
                        boff = (abs_byte + b) & 63;
                        dram_mem[beat][boff*8 +: 8] = qp[b*8 +: 8];
                    end
                end
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[4/12] Fused K (512x2048, GQA H_KV=8, DMA, 1 launch) + RoPE + LogQ...");
        run_fused_qk_rope_dma_ex(32, DRAM_WK, tcdm_addr(0, 16), tcdm_addr(128, 16),
                                 tcdm_addr(128, 29), 1'b1, "Fused K+RoPE", 1'b1, 255, 255);
        for (int hkv = 0; hkv < H_KV; hkv++)
            for (int kp = 0; kp < K_PACKETS_QK; kp++)
                for (int l = 0; l < 16; l++) begin
                    logic [71:0] kp_pkt;
                    int lin;
                    lin = (hkv * K_PACKETS_QK + kp) * 16 + l;
                    cpu_mem_read(tcdm_addr(128 + (lin & 255), 16 + (lin >> 8)), kp_pkt);
                    for (int dvi = 0; dvi < DV_TILES; dvi++) begin
                        int tile_off, abs_byte;
                        tile_off = (hkv * DV_TILES + dvi) * 768;
                        abs_byte = tile_off + (kp * 16 + l) * 9;
                        for (int b = 0; b < 9; b++) begin
                            int beat, boff;
                            beat = (DRAM_KV >> 6) + ((abs_byte + b) >> 6);
                            boff = (abs_byte + b) & 63;
                            dram_mem[beat][boff*8 +: 8] = kp_pkt[b*8 +: 8];
                        end
                    end
                    begin
                        int kc_byte;
                        kc_byte = (hkv*K_PACKETS_QK*16 + kp*16 + l) * 9;
                        for (int b = 0; b < 9; b++) begin
                            int beat, boff;
                            beat = (DRAM_KC >> 6) + ((kc_byte + b) >> 6);
                            boff = (kc_byte + b) & 63;
                            dram_mem[beat][boff*8 +: 8] = kp_pkt[b*8 +: 8];
                        end
                    end
                end
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[5/12] Fused V + Transpose (32 jobs, DMA)...");
        for (int hkv = 0; hkv < H_KV; hkv++)
            for (int dvi = 0; dvi < DV_TILES; dvi++) begin
                run_gemm_v_dma_ex(hkv*DV_TILES + dvi, 128 + hkv*DV_TILES + dvi,
                                  32'h0, tcdm_addr(0, 16), 1'b1, "GEMM V");
                for (int l = 0; l < 16; l++) begin
                    logic [71:0] vp_pkt;
                    int tile_off, abs_byte;
                    cpu_mem_read(tcdm_addr(128 + hkv*DV_TILES + dvi, l), vp_pkt);
                    tile_off = (hkv * DV_TILES + dvi) * 768;
                    abs_byte = tile_off + (4 * 16 + l) * 9;
                    for (int b = 0; b < 9; b++) begin
                        int beat, boff;
                        beat = (DRAM_KV >> 6) + ((abs_byte + b) >> 6);
                        boff = (abs_byte + b) & 63;
                        dram_mem[beat][boff*8 +: 8] = vp_pkt[b*8 +: 8];
                    end
                end
            end
        run_dma_writeback(2'b01, tcdm_addr(128, 0), DRAM_VC, 512 * 9, "DMA WB V-cache");
        verify_dram_vs_tcdm(1'b0, 128, 0,           DRAM_VC, 512,     "Verify DRAM V-cache");
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[6/12] HW Causal FlashAttention (enable_dma=1 32'h2D, 2x16 heads, 288KB TCDM)...");
        run_dma_writeback(2'b10, tcdm_addr(0, 30), DRAM_QPKT,          1024 * 9, "DMA Load QPKT0");
        run_dma_writeback(2'b10, tcdm_addr(0, 31), DRAM_QPKT + 1024*9, 1024 * 9, "DMA Load QPKT1");
        run_attn_gqa_ex(16, DRAM_KV,                 tcdm_addr(0, 30), tcdm_addr(0, 28), "Attn H0..15");
        run_attn_gqa_ex(16, DRAM_KV + 4 * 4 * 768,   tcdm_addr(0, 31), tcdm_addr(0, 29), "Attn H16..31");
        for (int t = 0; t < M_TOKENS; t++) begin
            for (int i = 0; i < 64; i++) begin
                logic [71:0] opkt;
                cpu_mem_read(tcdm_addr(t*64 + i, 28), opkt);
                cpu_mem_write(tcdm_addr(i, 16 + t), opkt);
                cpu_mem_read(tcdm_addr(t*64 + i, 29), opkt);
                cpu_mem_write(tcdm_addr(64 + i, 16 + t), opkt);
            end
        end
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[7/12 & 8/12] GEMM O (2x64 N-tiles, 16-bank) + Residual 1 + RMSNorm 2...");
        run_dma_writeback(2'b00, tcdm_addr(512, 16), DRAM_X, 8192 * 9, "DMA Load X");
        for (int c = 0; c < 2; c++) begin
            run_gemm_dma_ex(64, K_PACKETS_QKV, 0,
                            DRAM_WO + 32'(c) * 32'd64 * 32'(K_PACKETS_QKV) * 32'd144,
                            K_PACKETS_QKV * 144,
                            tcdm_addr(0, 16), tcdm_addr(128, 16), 1'b1, "GEMM O chunk", 1'b1);
            run_sfu_seq(3'd1, 16, 64, 1,
                        tcdm_addr(128, 16),         1, 4 * 32,
                        tcdm_addr(512 + c*256, 16), 1, 4 * 32,
                        tcdm_addr(512 + c*256, 16), 1, 4 * 32, 32'h3F800000);
        end
        check_sqnr_rows_16bank(512, g_res1_fq, "Residual 1 vs FQ", 0.96);
        check_sqnr_rows_16bank(512, g_res1,    "Residual 1 vs FP32", 0.95);
        run_dma_writeback(2'b01, tcdm_addr(512, 16), DRAM_RES1, 8192 * 9, "DMA Save RES1");
        for (int w = 0; w < ROW_WORDS; w++)
            cpu_mem_write(tcdm_addr(w, 0), pre_data[8704 + w]);
        run_sfu_seq(3'd2, 16, 1, 128,
                    tcdm_addr(512, 16), 1, 0,
                    tcdm_addr(0, 0),    0, 0,
                    tcdm_addr(0, 16),   1, 0, 32'h3A000000);
        check_sqnr_rows_16bank(0, g_n2_fq, "RMSNorm 2 vs FQ", 0.96);
        check_sqnr_rows_16bank(0, g_n2,    "RMSNorm 2 vs FP32", 0.95);
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[9/12] HW Requant Norm2 -> ACT (banks 16..31, words 0..127, 1 launch)...");
        run_requant_wrap(tcdm_addr(0, 16), M_TOKENS * 128, tcdm_addr(0, 16), 511, 127, "Requant Norm2");
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[10/12] FFN Gate & Up (16x32 N-tiles, 16-bank) + SiLU*Up -> FF1 (words 512..1023)...");
        for (int c = 0; c < 16; c++) begin
            run_gemm_dma_ex(32, K_PACKETS_QKV, 0,
                            DRAM_WG + 32'(c) * 32'd32 * 32'(K_PACKETS_QKV) * 32'd144,
                            K_PACKETS_QKV * 144,
                            tcdm_addr(0, 16), tcdm_addr(128, 16), 1'b1, "GEMM Gate", 1'b1);
            run_gemm_dma_ex(32, K_PACKETS_QKV, 0,
                            DRAM_WU + 32'(c) * 32'd32 * 32'(K_PACKETS_QKV) * 32'd144,
                            K_PACKETS_QKV * 144,
                            tcdm_addr(0, 16), tcdm_addr(256, 16), 1'b1, "GEMM Up", 1'b1);
            run_sfu_seq(3'd4, 16, 32, 1,
                        tcdm_addr(128, 16),         1, 4 * 32,
                        tcdm_addr(256, 16),         1, 4 * 32,
                        tcdm_addr(512 + c*32, 16),  1, 32, 32'h3F800000, 1'b1);
        end
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[11/12] GEMM Down (1x128 N-tiles, DMA + 2x4096 K-Tiling, 288KB TCDM)...");
        run_gemm_dma_ex(128, K_TILE_DOWN, 1,
                        DRAM_WD, K_PACKETS_DOWN * 144,
                        tcdm_addr(512, 16), tcdm_addr(0, 16), 1'b1, "GEMM Down", 1'b1);
        $display("       cycles=%0d", cyc_total - t0);

        t0 = cyc_total;
        $display("[12/12] SFU Residual 2 -> Final Output Check & DRAM Writeback...");
        run_dma_writeback(2'b00, tcdm_addr(512, 16), DRAM_RES1, 8192 * 9, "DMA Load RES1");
        run_sfu_seq(3'd1, 16, 128, 1,
                    tcdm_addr(0, 16),   1, 4 * 32,
                    tcdm_addr(512, 16), 1, 4 * 32,
                    tcdm_addr(0, 16),   1, 4 * 32, 32'h3F800000);
        check_sqnr_rows_16bank(0, g_final_fq, "Final vs PyFQ", 0.91);

        sum_ref2 = 0.0; sum_err2 = 0.0; dot_p = 0.0; sum_rtl2 = 0.0;
        for (int t = 0; t < EVAL_TOKENS; t++)
            for (int w = 0; w < ROW_WORDS; w++) begin
                cpu_mem_read(tcdm_addr(w, 16 + t), rw);
                for (int e = 0; e < 4; e++) begin
                    v_rtl  = fp16_to_real(rw[16*e +: 16]);
                    v_fp32 = fp32_to_real(g_final_fp32[t*D_MODEL + 4*w + e]);
                    sum_ref2 += v_fp32 * v_fp32;
                    sum_rtl2 += v_rtl  * v_rtl;
                    sum_err2 += (v_rtl - v_fp32) * (v_rtl - v_fp32);
                    dot_p    += v_rtl * v_fp32;
                end
            end
        sqnr  = (sum_err2 > 0.0) ? (10.0 * $log10(sum_ref2 / sum_err2)) : 999.0;
        cos_s = dot_p / ($sqrt(sum_rtl2) * $sqrt(sum_ref2));
        check(cos_s >= 0.88, "Final Layer Output vs FP32");

        run_dma_writeback(2'b01, tcdm_addr(0, 16), DRAM_OUT, 8192 * 9, "DMA WB Final Out");
        verify_dram_vs_tcdm(1'b0, 0, 16,           DRAM_OUT, 8192,     "Verify DRAM Final Out");
        end else begin
        // =====================================================================
        // PREFILL M > 16 (M_TOKENS = 32, 64, ...) ON TRUE 288 KB TCDM
        // M-batched (16 tokens/batch) Linear/SFU + Full-Sequence FlashAttention
        // =====================================================================
        $display("[1/12 .. 5/12] Pre-Attention M-batched (%0d x 16 tokens)...", M_CHUNKS);
        for (int mb = 0; mb < M_CHUNKS; mb++) begin
            for (int t = 0; t < M_BATCH; t++)
                for (int w = 0; w < ROW_WORDS; w++)
                    cpu_mem_write(tcdm_addr(w, 16 + t), pre_data[(mb*M_BATCH + t)*ROW_WORDS + w]);
            for (int w = 0; w < ROW_WORDS; w++)
                cpu_mem_write(tcdm_addr(w, 0), pre_data[M_TOKENS*ROW_WORDS + w]);

            run_sfu_seq(3'd2, M_BATCH, 1, 128,
                        tcdm_addr(0, 16), 1, 0, tcdm_addr(0, 0), 0, 0,
                        tcdm_addr(512, 16), 1, 0, 32'h3A000000);
            check_sqnr_rows_16bank(512, g_xnorm, "RMSNorm 1", 0.999, mb * M_BATCH);
            run_dma_writeback(2'b01, tcdm_addr(0, 16), DRAM_X + mb * 8192 * 9, 8192 * 9, "DMA Save X");

            run_requant_wrap(tcdm_addr(512, 16), M_BATCH * 128, tcdm_addr(0, 16), 511, 127, "Requant Xn");
            for (int i = 0; i < 256; i++)
                cpu_mem_write(tcdm_addr(128 + i, 29), rope_mbatch[mb * 256 + i]);

            run_fused_qk_rope_dma_ex(128, DRAM_WQ, tcdm_addr(0, 16), tcdm_addr(128, 16),
                                     tcdm_addr(128, 29), 1'b1, "Fused Q+RoPE", 1'b1, 255, 255);
            for (int h = 0; h < H_Q; h++)
                for (int t = 0; t < M_BATCH; t++)
                    for (int kp = 0; kp < K_PACKETS_QK; kp++) begin
                        logic [71:0] qp;
                        int lin, idx, abs_byte;
                        lin = (h * K_PACKETS_QK + kp) * M_BATCH + t;
                        cpu_mem_read(tcdm_addr(128 + (lin & 255), 16 + (lin >> 8)), qp);
                        idx = (h * M_TOKENS + mb * M_BATCH + t) * K_PACKETS_QK + kp;
                        abs_byte = idx * 9;
                        for (int b = 0; b < 9; b++) begin
                            int beat, boff;
                            beat = (DRAM_QPKT >> 6) + ((abs_byte + b) >> 6);
                            boff = (abs_byte + b) & 63;
                            dram_mem[beat][boff*8 +: 8] = qp[b*8 +: 8];
                        end
                    end

            run_fused_qk_rope_dma_ex(32, DRAM_WK, tcdm_addr(0, 16), tcdm_addr(128, 16),
                                     tcdm_addr(128, 29), 1'b1, "Fused K+RoPE", 1'b1, 255, 255);
            for (int hkv = 0; hkv < H_KV; hkv++)
                for (int kp = 0; kp < K_PACKETS_QK; kp++)
                    for (int l = 0; l < 16; l++) begin
                        logic [71:0] kp_pkt;
                        int lin;
                        lin = (hkv * K_PACKETS_QK + kp) * 16 + l;
                        cpu_mem_read(tcdm_addr(128 + (lin & 255), 16 + (lin >> 8)), kp_pkt);
                        for (int dvi = 0; dvi < DV_TILES; dvi++) begin
                            int tile_off, abs_byte;
                            tile_off = ((hkv * DV_TILES + dvi) * KV_TILES + mb) * 768;
                            abs_byte = tile_off + (kp * 16 + l) * 9;
                            for (int b = 0; b < 9; b++) begin
                                int beat, boff;
                                beat = (DRAM_KV >> 6) + ((abs_byte + b) >> 6);
                                boff = (abs_byte + b) & 63;
                                dram_mem[beat][boff*8 +: 8] = kp_pkt[b*8 +: 8];
                            end
                        end
                    end

            for (int hkv = 0; hkv < H_KV; hkv++)
                for (int dvi = 0; dvi < DV_TILES; dvi++) begin
                    run_gemm_v_dma_ex(hkv*DV_TILES + dvi, 128 + hkv*DV_TILES + dvi,
                                      32'h0, tcdm_addr(0, 16), 1'b1, "GEMM V");
                    for (int l = 0; l < 16; l++) begin
                        logic [71:0] vp_pkt;
                        int tile_off, abs_byte;
                        cpu_mem_read(tcdm_addr(128 + hkv*DV_TILES + dvi, l), vp_pkt);
                        tile_off = ((hkv * DV_TILES + dvi) * KV_TILES + mb) * 768;
                        abs_byte = tile_off + (4 * 16 + l) * 9;
                        for (int b = 0; b < 9; b++) begin
                            int beat, boff;
                            beat = (DRAM_KV >> 6) + ((abs_byte + b) >> 6);
                            boff = (abs_byte + b) & 63;
                            dram_mem[beat][boff*8 +: 8] = vp_pkt[b*8 +: 8];
                        end
                    end
                end
        end

        t0 = cyc_total;
        $display("[6/12] HW Causal FlashAttention (N_KV=%0d, %0d heads/grp, 288KB TCDM)...",
                 N_KV, 1024 / (M_TOKENS * K_PACKETS_QK));
        begin
            int h_grp, n_hgrp, q_words_grp;
            h_grp = 1024 / (M_TOKENS * K_PACKETS_QK);
            if (h_grp > 16) h_grp = 16;
            n_hgrp = H_Q / h_grp;
            q_words_grp = h_grp * M_TOKENS * K_PACKETS_QK;
            for (int hg = 0; hg < n_hgrp; hg++) begin
                run_dma_writeback(2'b10, tcdm_addr(0, 30), DRAM_QPKT + hg * q_words_grp * 9,
                                  q_words_grp * 9, "DMA Load QPKT");
                run_attn_gqa_ex(h_grp, DRAM_KV + hg * (h_grp / 4) * DV_TILES * KV_TILES * 768,
                                tcdm_addr(0, 30), tcdm_addr(0, 28), "Attn GQA");
                for (int t = 0; t < M_TOKENS; t++)
                    for (int hl = 0; hl < h_grp; hl++)
                        for (int dvi = 0; dvi < DV_TILES; dvi++) begin
                            logic [71:0] opkt;
                            cpu_mem_read(tcdm_addr((t*h_grp + hl)*DV_TILES + dvi, 28), opkt);
                            o_act_buf[t][(hg*h_grp + hl)*DV_TILES + dvi] = opkt;
                        end
            end
        end
        $display("       Attn cycles=%0d", cyc_total - t0);

        $display("[7/12 .. 12/12] Post-Attention & FFN M-batched (%0d x 16 tokens)...", M_CHUNKS);
        sum_ref2 = 0.0; sum_err2 = 0.0; dot_p = 0.0; sum_rtl2 = 0.0;
        for (int mb = 0; mb < M_CHUNKS; mb++) begin
            for (int t = 0; t < M_BATCH; t++)
                for (int i = 0; i < 128; i++)
                    cpu_mem_write(tcdm_addr(i, 16 + t), o_act_buf[mb*M_BATCH + t][i]);
            run_dma_writeback(2'b00, tcdm_addr(512, 16), DRAM_X + mb * 8192 * 9, 8192 * 9, "DMA Load X");
            for (int c = 0; c < 2; c++) begin
                run_gemm_dma_ex(64, K_PACKETS_QKV, 0,
                                DRAM_WO + 32'(c) * 32'd64 * 32'(K_PACKETS_QKV) * 32'd144,
                                K_PACKETS_QKV * 144,
                                tcdm_addr(0, 16), tcdm_addr(128, 16), 1'b1, "GEMM O chunk", 1'b1);
                run_sfu_seq(3'd1, M_BATCH, 64, 1,
                            tcdm_addr(128, 16),         1, 4 * 32,
                            tcdm_addr(512 + c*256, 16), 1, 4 * 32,
                            tcdm_addr(512 + c*256, 16), 1, 4 * 32, 32'h3F800000);
            end
            check_sqnr_rows_16bank(512, g_res1_fq, "Residual 1 vs FQ", 0.96, mb * M_BATCH);
            check_sqnr_rows_16bank(512, g_res1,    "Residual 1 vs FP32", 0.95, mb * M_BATCH);
            run_dma_writeback(2'b01, tcdm_addr(512, 16), DRAM_RES1 + mb * 8192 * 9, 8192 * 9, "DMA Save RES1");
            for (int w = 0; w < ROW_WORDS; w++)
                cpu_mem_write(tcdm_addr(w, 0), pre_data[M_TOKENS*ROW_WORDS + ROW_WORDS + w]);
            run_sfu_seq(3'd2, M_BATCH, 1, 128,
                        tcdm_addr(512, 16), 1, 0, tcdm_addr(0, 0), 0, 0,
                        tcdm_addr(0, 16),   1, 0, 32'h3A000000);
            check_sqnr_rows_16bank(0, g_n2_fq, "RMSNorm 2 vs FQ", 0.96, mb * M_BATCH);
            check_sqnr_rows_16bank(0, g_n2,    "RMSNorm 2 vs FP32", 0.95, mb * M_BATCH);

            run_requant_wrap(tcdm_addr(0, 16), M_BATCH * 128, tcdm_addr(0, 16), 511, 127, "Requant Norm2");
            for (int c = 0; c < 16; c++) begin
                run_gemm_dma_ex(32, K_PACKETS_QKV, 0,
                                DRAM_WG + 32'(c) * 32'd32 * 32'(K_PACKETS_QKV) * 32'd144,
                                K_PACKETS_QKV * 144,
                                tcdm_addr(0, 16), tcdm_addr(128, 16), 1'b1, "GEMM Gate", 1'b1);
                run_gemm_dma_ex(32, K_PACKETS_QKV, 0,
                                DRAM_WU + 32'(c) * 32'd32 * 32'(K_PACKETS_QKV) * 32'd144,
                                K_PACKETS_QKV * 144,
                                tcdm_addr(0, 16), tcdm_addr(256, 16), 1'b1, "GEMM Up", 1'b1);
                run_sfu_seq(3'd4, M_BATCH, 32, 1,
                            tcdm_addr(128, 16),         1, 4 * 32,
                            tcdm_addr(256, 16),         1, 4 * 32,
                            tcdm_addr(512 + c*32, 16),  1, 32, 32'h3F800000, 1'b1);
            end
            run_gemm_dma_ex(128, K_TILE_DOWN, 1,
                            DRAM_WD, K_PACKETS_DOWN * 144,
                            tcdm_addr(512, 16), tcdm_addr(0, 16), 1'b1, "GEMM Down", 1'b1);
            run_dma_writeback(2'b00, tcdm_addr(512, 16), DRAM_RES1 + mb * 8192 * 9, 8192 * 9, "DMA Load RES1");
            run_sfu_seq(3'd1, M_BATCH, 128, 1,
                        tcdm_addr(0, 16),   1, 4 * 32,
                        tcdm_addr(512, 16), 1, 4 * 32,
                        tcdm_addr(0, 16),   1, 4 * 32, 32'h3F800000);
            check_sqnr_rows_16bank(0, g_final_fq, "Final vs PyFQ", 0.91, mb * M_BATCH);

            for (int t = 0; t < M_BATCH; t++)
                for (int w = 0; w < ROW_WORDS; w++) begin
                    cpu_mem_read(tcdm_addr(w, 16 + t), rw);
                    for (int e = 0; e < 4; e++) begin
                        v_rtl  = fp16_to_real(rw[16*e +: 16]);
                        v_fp32 = fp32_to_real(g_final_fp32[(mb*M_BATCH + t)*D_MODEL + 4*w + e]);
                        sum_ref2 += v_fp32 * v_fp32;
                        sum_rtl2 += v_rtl  * v_rtl;
                        sum_err2 += (v_rtl - v_fp32) * (v_rtl - v_fp32);
                        dot_p    += v_rtl * v_fp32;
                    end
                end
            run_dma_writeback(2'b01, tcdm_addr(0, 16), DRAM_OUT + mb * 8192 * 9, 8192 * 9, "DMA WB Final Out");
            verify_dram_vs_tcdm(1'b0, 0, 16,           DRAM_OUT + mb * 8192 * 9, 8192,     "Verify DRAM Final Out");
        end
        sqnr  = (sum_err2 > 0.0) ? (10.0 * $log10(sum_ref2 / sum_err2)) : 999.0;
        cos_s = dot_p / ($sqrt(sum_rtl2) * $sqrt(sum_ref2));
        check(cos_s >= 0.88, "Final Layer Output vs FP32");
        end

        $display("\n========================================================================");
        $display("  LLAMA-3.2-1B FULL PREFILL LAYER RESULT");
        $display("========================================================================");
        $display("  RTL vs FP32       : SQNR = %0.2f dB, CosineSim = %0.6f", sqnr, cos_s);
        $display("  Total cycles      : %0d (seq=%0d, sfu=%0d, mac=%0d, dma=%0d)",
                 cyc_total, cyc_seq, cyc_sfu, cyc_core_compute, cyc_dma);
        $display("  DRAM 512b rd beats: %0d (%0d bytes)", stat_rd_beats, stat_rd_beats * 64);
        $display("  DRAM 512b wr beats: %0d (%0d bytes)", stat_wr_beats, stat_wr_beats * 64);
        $display("  Checks            : %0d passed, %0d failed", checks_pass, checks_fail);
        $display("========================================================================\n");

        if (checks_fail == 0)
            $display(">>> LLAMA-3.2-1B FULL LAYER PREFILL PASSED (0 errors) <<<\n");
        else
            $fatal(1, "LLAMA-3.2-1B FULL LAYER FAILED (%0d check fails)", checks_fail);

        $finish;
    end

endmodule
