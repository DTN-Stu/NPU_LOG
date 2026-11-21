// =============================================================================
// tb_llama3_1b_configurable.sv
//
// Configurable Real NousResearch/Llama-3.2-1B Full-Layer End-to-End Testbench
// on snax_llm_top_clean (288 KB TCDM, BANK_DEPTH=1024) + 8-port snax_simple_dma
// + 64 MB Byte-Strobed 512-bit External DRAM:
//
//   Runtime plusargs (no recompile needed across sizes):
//     +DECODE=0|1         (0 = Prefill, 1 = Decode M=1; default 0)
//     +M_TOKENS=<int>     (Prefill: multiple of 16 up to 512; Decode: 1)
//     +N_KV=<int>         (KV context multiple of 16 up to 2048; default M_TOKENS / 64)
//     +Q_SUB=<int>        (Optional Q macro slice; default min(Q_TILES, 8))
//     +DATA_DIR=<path>    (Directory with generated stimulus & golden .hex)
//     +WEIGHTS_HEX=<path> (Shared 34.2 MB dram_weights_l<L>.hex)
//     +DUMP               (Optional FST waveform dump)
// =============================================================================
`timescale 1ns/1ps

module tb_llama3_1b_configurable;

    // --------------------------------------------------------- Fixed Arch ----
    localparam int D_MODEL        = 2048;
    localparam int D_FF           = 8192;
    localparam int H_Q            = 32;
    localparam int H_KV           = 8;
    localparam int GQA_M1         = (H_Q / H_KV) - 1; // 3
    localparam int D_K            = 64;
    localparam int D_V            = 64;
    localparam int DV_TILES       = D_V / 16;         // 4
    localparam int K_PACKETS_QKV  = D_MODEL / 16;     // 128
    localparam int K_PACKETS_QK   = D_K / 16;         // 4
    localparam int K_PACKETS_DOWN = D_FF / 16;        // 512
    localparam int K_TILE_DOWN    = 256;              // 256 pkts = 4096/K-tile
    localparam int N_TILES_MODEL  = D_MODEL / 16;     // 128
    localparam int N_TILES_KV     = (H_KV * D_K) / 16;// 32
    localparam int N_TILES_FF     = D_FF / 16;        // 512
    localparam int CHUNK_NT       = 16;
    localparam int N_CHUNKS       = N_TILES_FF / CHUNK_NT; // 32
    localparam int ROW_WORDS      = D_MODEL / 4;      // 512
    localparam int M_BATCH        = 16;
    localparam int TB_BANK_DEPTH  = 1024;             // True 288 KB TCDM

    localparam int MAX_M          = 512;
    localparam int MAX_KV         = 2048;
    localparam int MAX_FP16       = MAX_M * D_MODEL;  // 1,048,576
    localparam int MAX_X_WORDS    = MAX_M * ROW_WORDS;// 262,144
    localparam int MAX_ROPE_WORDS = (MAX_M / 16) * 256; // 8,192 (also >= 512 decode)
    localparam int MAX_K_WORDS    = H_KV * (MAX_KV / 16) * K_PACKETS_QK * 16; // 65,536
    localparam int MAX_V_WORDS    = H_KV * DV_TILES * (MAX_KV / 16) * 16;     // 65,536
    localparam int DRAM_WT_BEATS  = 534528;           // 34,209,792 / 64

    localparam int CLK_PERIOD     = 10;
    localparam int DRAM_LATENCY   = 6;
    localparam int JOB_TIMEOUT    = 50_000_000;

    // --------------------------------------------------------- TCDM Map ------
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

    // --------------------------------------------------------- 64MB DRAM Map --
    localparam logic [31:0] DRAM_WQ   = 32'h0000_0000;
    localparam logic [31:0] DRAM_WK   = 32'h0024_0000;
    localparam logic [31:0] DRAM_WV   = 32'h002D_0000;
    localparam logic [31:0] DRAM_WO   = 32'h0036_0000;
    localparam logic [31:0] DRAM_WG   = 32'h005A_0000;
    localparam logic [31:0] DRAM_WU   = 32'h00EA_0000;
    localparam logic [31:0] DRAM_WD   = 32'h017A_0000;
    localparam logic [31:0] DRAM_X    = 32'h0210_0000; // Up to 2.5 MB (M<=512)
    localparam logic [31:0] DRAM_RES1 = 32'h0238_0000; // Up to 2.5 MB
    localparam logic [31:0] DRAM_OUT  = 32'h0260_0000; // Up to 2.5 MB
    localparam logic [31:0] DRAM_QPKT = 32'h0288_0000; // Up to 1.0 MB
    localparam logic [31:0] DRAM_OPKT = 32'h0298_0000; // Up to 1.0 MB
    localparam logic [31:0] DRAM_K    = 32'h02A8_0000; // Up to 1.0 MB (N_KV<=2048)
    localparam logic [31:0] DRAM_V    = 32'h02B8_0000; // Up to 1.0 MB (N_KV<=2048)

    // --------------------------------------------------------- CSR Map -------
    localparam logic [11:0] A_CTRL          = 12'd0;
    localparam logic [11:0] A_WO_MODE       = 12'd12;
    localparam logic [11:0] A_WO_BASE       = 12'd13;
    localparam logic [11:0] A_WO_NV         = 12'd14;
    localparam logic [11:0] A_WO_LS         = 12'd15;
    localparam logic [11:0] A_V_BASE        = 12'd25;
    localparam logic [11:0] A_V_NB          = 12'd26;
    localparam logic [11:0] A_RD_BASE       = 12'd27;
    localparam logic [11:0] A_RD_NV         = 12'd28;
    localparam logic [11:0] A_RD_LS         = 12'd29;
    localparam logic [11:0] A_SEQ_CTRL      = 12'd31;
    localparam logic [11:0] A_ATTN_H        = 12'd32;
    localparam logic [11:0] A_ATTN_QT       = 12'd33;
    localparam logic [11:0] A_ATTN_MQ       = 12'd34;
    localparam logic [11:0] A_ATTN_DV       = 12'd35;
    localparam logic [11:0] A_ATTN_KV       = 12'd36;
    localparam logic [11:0] A_ATTN_KB       = 12'd37;
    localparam logic [11:0] A_GEMM_NT       = 12'd38;
    localparam logic [11:0] A_GEMM_MT       = 12'd39;
    localparam logic [11:0] A_GEMM_KB       = 12'd40;
    localparam logic [11:0] A_RING_BASE     = 12'd41;
    localparam logic [11:0] A_Q_BASE        = 12'd42;
    localparam logic [11:0] A_OUT_BASE      = 12'd43;
    localparam logic [11:0] A_DRAM_KV       = 12'd44;
    localparam logic [11:0] A_TILE_ST       = 12'd45;
    localparam logic [11:0] A_SFU_CTRL      = 12'd46;
    localparam logic [11:0] A_SFU_INV_N     = 12'd47;
    localparam logic [11:0] A_SFU_EPS       = 12'd48;
    localparam logic [11:0] A_SFU_B_BASE    = 12'd49;
    localparam logic [11:0] A_SFU_B_NV      = 12'd50;
    localparam logic [11:0] A_SFU_B_LS      = 12'd51;
    localparam logic [11:0] A_WRAP_CFG      = 12'd52;
    localparam logic [11:0] A_START         = 12'd53;
    localparam logic [11:0] A_RO_ID         = 12'd61;
    localparam logic [11:0] A_EXT_MACRO_SUB = 12'd64;
    localparam logic [11:0] A_EXT_MACRO_ACT = 12'd65;
    localparam logic [11:0] A_EXT_MACRO_OUT = 12'd66;
    localparam logic [11:0] A_EXT_MACRO_ASTR= 12'd67;
    localparam logic [11:0] A_EXT_MACRO_OSTR= 12'd68;
    localparam logic [11:0] A_EXT_DRAM_V    = 12'd69;
    localparam logic [31:0] CSR_ID_VALUE    = 32'h534E_4146;

    // --------------------------------------------------------- Runtime Cfg ---
    bit    IS_DECODE;
    int    M_TOKENS, N_KV, EVAL_TOKENS, M_CHUNKS, M_Q, Q_TILES, Q_SUB;
    int    KV_TILES, KV_TILES_PAST, NUM_VEC_XN;
    bit    KV_PIN_EN;
    string DATA_DIR, WEIGHTS_HEX;

    // --------------------------------------------------------- DUT Signals ---
    logic clk, rst_n;
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD / 2) clk = ~clk;
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

    // 64 MB Byte-Strobed 512-bit External DRAM (1,048,576 x 64B beats)
    localparam int DRAM_WORDS = 1048576;
    logic [511:0] dram_mem [0:DRAM_WORDS-1];
    logic [511:0] dram_wr_mask;
    logic [511:0] lat_data [0:DRAM_LATENCY-1];
    logic         lat_vld  [0:DRAM_LATENCY-1];
    assign dram_req_ready = 1'b1;

    always_comb begin
        for (int b = 0; b < 64; b++)
            dram_wr_mask[b*8 +: 8] = {8{dram_req_strb[b]}};
    end

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
                    dram_mem[dram_req_addr[25:6]] <= (dram_mem[dram_req_addr[25:6]] & ~dram_wr_mask) |
                                                     (dram_req_wdata & dram_wr_mask);
                end else begin
                    lat_vld[0]  <= 1'b1;
                    lat_data[0] <= dram_mem[dram_req_addr[25:6]];
                end
            end
        end
    end

    // --------------------------------------------------------- Hex Buffers ---
    logic [71:0] x_fp16_mem   [0:MAX_X_WORDS-1];
    logic [71:0] gamma1_mem   [0:ROW_WORDS-1];
    logic [71:0] gamma2_mem   [0:ROW_WORDS-1];
    logic [71:0] rope_mb_mem  [0:MAX_ROPE_WORDS-1];
    logic [71:0] k_past_mem   [0:MAX_K_WORDS-1];
    logic [71:0] v_past_mem   [0:MAX_V_WORDS-1];
    logic [71:0] v_tail_fp16  [0:15 * N_TILES_KV * 4 - 1];

    logic [15:0] g_xnorm      [0:MAX_FP16-1];
    logic [15:0] g_res1       [0:MAX_FP16-1];
    logic [15:0] g_n2         [0:MAX_FP16-1];
    logic [15:0] g_res1_fq    [0:MAX_FP16-1];
    logic [15:0] g_n2_fq      [0:MAX_FP16-1];
    logic [15:0] g_final_fq   [0:MAX_FP16-1];
    logic [15:0] g_final_fp16 [0:MAX_FP16-1];
    logic [31:0] g_final_fp32 [0:MAX_FP16-1];

    // --------------------------------------------------------- Helpers -------
    function automatic logic [31:0] tcdm_addr(input int word, input int bank);
        tcdm_addr = (32'(word) << 9) | (32'(bank) << 4);
    endfunction

    task automatic write_dram_9b(input logic [31:0] dram_base, input int word_idx, input logic [71:0] pkt);
        int abs_byte;
        abs_byte = word_idx * 9;
        for (int b = 0; b < 9; b++) begin
            int beat, boff;
            beat = (dram_base >> 6) + ((abs_byte + b) >> 6);
            boff = (abs_byte + b) & 63;
            dram_mem[beat][boff*8 +: 8] = pkt[b*8 +: 8];
        end
    endtask

    task automatic read_dram_9b(input logic [31:0] dram_base, input int word_idx, output logic [71:0] pkt);
        int abs_byte;
        abs_byte = word_idx * 9;
        for (int b = 0; b < 9; b++) begin
            int beat, boff;
            beat = (dram_base >> 6) + ((abs_byte + b) >> 6);
            boff = (abs_byte + b) & 63;
            pkt[b*8 +: 8] = dram_mem[beat][boff*8 +: 8];
        end
    endtask

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

    // --------------------------------------------------------- Telemetry -----
    longint cyc_total, cyc_seq, cyc_sfu, cyc_core_compute, cyc_dma, job_cycles_total;
    int checks_pass, checks_fail;

    always @(posedge clk) begin
        if (rst_n) begin
            cyc_total <= cyc_total + 1;
            if (dut.seq_busy)        cyc_seq <= cyc_seq + 1;
            if (dut.sfu_busy)        cyc_sfu <= cyc_sfu + 1;
            if (|dut.lane_valid)     cyc_core_compute <= cyc_core_compute + 1;
            if (|dma_tcdm_req_valid) cyc_dma <= cyc_dma + 1;
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
        job_cycles_total = job_cycles_total + 64'(cycle_cnt);
        check(seen_done, name);
        if (!seen_done)
            $fatal(1, "Timeout on job: %s (seq_state=%0d, busy=%b, done_irq=%b)",
                   name, dut.u_seq.state_q, busy, done_irq);
    endtask

    // Accumulated sequence-wide error accumulators
    real acc_n1_ref2, acc_n1_err2, acc_n1_dot, acc_n1_act2, acc_n1_max;
    real acc_r1fq_ref2, acc_r1fq_err2, acc_r1fq_dot, acc_r1fq_act2, acc_r1fq_max;
    real acc_r1fp_ref2, acc_r1fp_err2, acc_r1fp_dot, acc_r1fp_act2, acc_r1fp_max;
    real acc_n2fq_ref2, acc_n2fq_err2, acc_n2fq_dot, acc_n2fq_act2, acc_n2fq_max;
    real acc_n2fp_ref2, acc_n2fp_err2, acc_n2fp_dot, acc_n2fp_act2, acc_n2fp_max;
    real acc_fnfq_ref2, acc_fnfq_err2, acc_fnfq_dot, acc_fnfq_act2, acc_fnfq_max;
    real acc_fnfp_ref2, acc_fnfp_err2, acc_fnfp_dot, acc_fnfp_act2, acc_fnfp_max;

    function automatic logic [15:0] get_ref_fp16(input int ref_sel, input int idx);
        case (ref_sel)
            0: get_ref_fp16 = g_xnorm[idx];
            1: get_ref_fp16 = g_res1_fq[idx];
            2: get_ref_fp16 = g_res1[idx];
            3: get_ref_fp16 = g_n2_fq[idx];
            4: get_ref_fp16 = g_n2[idx];
            default: get_ref_fp16 = g_final_fq[idx];
        endcase
    endfunction

    task automatic accum_rows_1bank(
        input int bank, input int n_tok, input int tok_off,
        input int ref_sel,
        inout real s_ref2, inout real s_err2, inout real s_dot, inout real s_act2, inout real m_abs
    );
        logic [71:0] rw;
        for (int t = 0; t < n_tok; t++)
            for (int w = 0; w < ROW_WORDS; w++) begin
                cpu_mem_read(tcdm_addr(t*ROW_WORDS + w, bank), rw);
                for (int e = 0; e < 4; e++) begin
                    real v_a, v_g, df;
                    v_a = fp16_to_real(rw[16*e +: 16]);
                    v_g = fp16_to_real(get_ref_fp16(ref_sel, (tok_off + t)*D_MODEL + 4*w + e));
                    df  = (v_a > v_g) ? (v_a - v_g) : (v_g - v_a);
                    if (df > m_abs) m_abs = df;
                    s_ref2 += v_g * v_g;
                    s_act2 += v_a * v_a;
                    s_err2 += (v_a - v_g) * (v_a - v_g);
                    s_dot  += v_a * v_g;
                end
            end
    endtask

    task automatic accum_rows_16bank(
        input int word_off, input int n_tok, input int tok_off,
        input int ref_sel,
        inout real s_ref2, inout real s_err2, inout real s_dot, inout real s_act2, inout real m_abs
    );
        logic [71:0] rw;
        for (int t = 0; t < n_tok; t++)
            for (int w = 0; w < ROW_WORDS; w++) begin
                cpu_mem_read(tcdm_addr(word_off + w, 16 + t), rw);
                for (int e = 0; e < 4; e++) begin
                    real v_a, v_g, df;
                    v_a = fp16_to_real(rw[16*e +: 16]);
                    v_g = fp16_to_real(get_ref_fp16(ref_sel, (tok_off + t)*D_MODEL + 4*w + e));
                    df  = (v_a > v_g) ? (v_a - v_g) : (v_g - v_a);
                    if (df > m_abs) m_abs = df;
                    s_ref2 += v_g * v_g;
                    s_act2 += v_a * v_a;
                    s_err2 += (v_a - v_g) * (v_a - v_g);
                    s_dot  += v_a * v_g;
                end
            end
    endtask

    task automatic accum_final_fp32_1bank(input int bank, input int n_tok, input int tok_off);
        logic [71:0] rw;
        for (int t = 0; t < n_tok; t++)
            for (int w = 0; w < ROW_WORDS; w++) begin
                cpu_mem_read(tcdm_addr(t*ROW_WORDS + w, bank), rw);
                for (int e = 0; e < 4; e++) begin
                    real v_a, v_g, df;
                    v_a = fp16_to_real(rw[16*e +: 16]);
                    v_g = fp32_to_real(g_final_fp32[(tok_off + t)*D_MODEL + 4*w + e]);
                    df  = (v_a > v_g) ? (v_a - v_g) : (v_g - v_a);
                    if (df > acc_fnfp_max) acc_fnfp_max = df;
                    acc_fnfp_ref2 += v_g * v_g;
                    acc_fnfp_act2 += v_a * v_a;
                    acc_fnfp_err2 += (v_a - v_g) * (v_a - v_g);
                    acc_fnfp_dot  += v_a * v_g;
                end
            end
    endtask

    task automatic accum_final_fp32_16bank(input int word_off, input int n_tok, input int tok_off);
        logic [71:0] rw;
        for (int t = 0; t < n_tok; t++)
            for (int w = 0; w < ROW_WORDS; w++) begin
                cpu_mem_read(tcdm_addr(word_off + w, 16 + t), rw);
                for (int e = 0; e < 4; e++) begin
                    real v_a, v_g, df;
                    v_a = fp16_to_real(rw[16*e +: 16]);
                    v_g = fp32_to_real(g_final_fp32[(tok_off + t)*D_MODEL + 4*w + e]);
                    df  = (v_a > v_g) ? (v_a - v_g) : (v_g - v_a);
                    if (df > acc_fnfp_max) acc_fnfp_max = df;
                    acc_fnfp_ref2 += v_g * v_g;
                    acc_fnfp_act2 += v_a * v_a;
                    acc_fnfp_err2 += (v_a - v_g) * (v_a - v_g);
                    acc_fnfp_dot  += v_a * v_g;
                end
            end
    endtask

    task automatic report_metric(
        input string name,
        input real s_ref2, input real s_err2, input real s_dot, input real s_act2, input real m_abs,
        input real min_cos
    );
        real sqnr, cos_s;
        sqnr  = (s_err2 > 0.0) ? (10.0 * $log10(s_ref2 / s_err2)) : 999.0;
        cos_s = s_dot / ($sqrt(s_act2) * $sqrt(s_ref2));
        $display("    [METRIC] %-20s : SQNR = %6.2f dB | CosineSim = %0.6f | MaxAbs = %0.4f",
                 name, sqnr, cos_s, m_abs);
        check(cos_s >= min_cos, name);
    endtask

    // --------------------------------------------------------- Job Launchers -
    task automatic run_sfu_seq(
        input logic [2:0] op,
        input int m_tokens, input int n_tiles, input int k_vecs,
        input logic [31:0] a_base, input int a_t_16b, input int a_g_16b,
        input logic [31:0] b_base, input int b_t_16b, input int b_g_16b,
        input logic [31:0] o_base, input int o_t_16b, input int o_g_16b,
        input logic [31:0] inv_n, input logic out_fp4 = 1'b0
    );
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

    task automatic run_requant_addr(input logic [31:0] rd_base, input int nvec, input logic [31:0] wo_base, input string name);
        cpu_csr_write(A_CTRL,     32'h0);
        cpu_csr_write(A_WRAP_CFG, 32'h0);
        cpu_csr_write(A_SEQ_CTRL, 32'h3);
        cpu_csr_write(A_RD_BASE,  rd_base);
        cpu_csr_write(A_RD_NV,    nvec);
        cpu_csr_write(A_RD_LS,    32'h200);
        cpu_csr_write(A_WO_BASE,  wo_base);
        cpu_csr_write(A_START,    32'h0000_0003);
        wait_job(name);
    endtask

    task automatic run_requant_wrap(
        input logic [31:0] rd_base, input int nvec, input logic [31:0] wo_base,
        input int rd_wrap_words_m1, input int wo_wrap_words_m1, input string name
    );
        cpu_csr_write(A_CTRL,     32'h0);
        cpu_csr_write(A_WRAP_CFG, (32'(rd_wrap_words_m1) << 11) | 32'(wo_wrap_words_m1));
        cpu_csr_write(A_SEQ_CTRL, (32'd1 << 27) | (32'd1 << 26) | 32'h3);
        cpu_csr_write(A_RD_BASE,  rd_base);
        cpu_csr_write(A_RD_NV,    nvec);
        cpu_csr_write(A_RD_LS,    32'h200);
        cpu_csr_write(A_WO_BASE,  wo_base);
        cpu_csr_write(A_START,    32'h0000_0003);
        wait_job(name);
        cpu_csr_write(A_WRAP_CFG, 32'h0);
    endtask

    task automatic run_fused_qk_rope_dma_ex(
        input int n_tiles, input logic [31:0] dram_base,
        input logic [31:0] q_base, input logic [31:0] out_base,
        input logic [31:0] rope_base, input bit act_16bank, input string name,
        input bit wrap_en = 1'b0, input int wo_wrap_m1 = 0, input int rd_b_wrap_m1 = 0
    );
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
        cpu_csr_write(A_WRAP_CFG,   32'h0);
    endtask

    task automatic run_gemm_v_dma_ex(
        input int n_tile, input int v_word, input logic [31:0] ring_base,
        input logic [31:0] q_base, input bit act_16bank, input string name
    );
        cpu_csr_write(A_CTRL,      32'h0);
        cpu_csr_write(A_SEQ_CTRL,  (act_16bank ? (32'd1 << 24) : 32'd0) | 32'h38);
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

    task automatic run_gemm_dma_ex(
        input int n_tiles, input int k_blks_per_tile, input int k_tiles_m1,
        input logic [31:0] dram_base, input int tile_stride_bytes,
        input logic [31:0] q_base, input logic [31:0] out_base,
        input bit act_16bank, input string name,
        input bit out_16bank = 1'b0, input logic [1:0] out_mode = 2'd0
    );
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

    task automatic run_dma_writeback(
        input logic [1:0] dir, input logic [31:0] tcdm_base,
        input logic [31:0] dram_base, input int size_bytes, input string name
    );
        cpu_csr_write(A_CTRL,      32'h0);
        cpu_csr_write(A_SEQ_CTRL,  32'h40 | (32'(dir) << 4));
        cpu_csr_write(A_OUT_BASE,  tcdm_base);
        cpu_csr_write(A_DRAM_KV,   dram_base);
        cpu_csr_write(A_TILE_ST,   size_bytes);
        cpu_csr_write(A_START,     32'h0000_0003);
        wait_job(name);
    endtask

    task automatic verify_dram_vs_tcdm(
        input logic linear_mode, input int tcdm_word0, input int tcdm_bank0,
        input logic [31:0] dram_base, input int n_words, input string name
    );
        int bad, base_beat, cur_w, cur_b;
        logic [71:0] exp_w, act_w;
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

    task automatic clear_ext_macro_csrs();
        cpu_csr_write(A_EXT_MACRO_SUB,  32'h0);
        cpu_csr_write(A_EXT_MACRO_ACT,  32'h0);
        cpu_csr_write(A_EXT_MACRO_OUT,  32'h0);
        cpu_csr_write(A_EXT_MACRO_ASTR, 32'h0);
        cpu_csr_write(A_EXT_MACRO_OSTR, 32'h0);
        cpu_csr_write(A_EXT_DRAM_V,     32'h0);
    endtask

    // --------------------------------------------------------- Main Execution -
    initial begin
        int dec_arg, m_arg, kv_arg, qs_arg;
        longint t0, t_attn;
        logic [31:0] rd32;
        int n_fp16;

        IS_DECODE = 1'b0;
        if ($value$plusargs("DECODE=%d", dec_arg)) IS_DECODE = (dec_arg != 0);

        M_TOKENS = IS_DECODE ? 1 : 16;
        if ($value$plusargs("M_TOKENS=%d", m_arg)) M_TOKENS = m_arg;
        if (IS_DECODE) M_TOKENS = 1;

        N_KV = IS_DECODE ? 64 : M_TOKENS;
        if ($value$plusargs("N_KV=%d", kv_arg)) N_KV = kv_arg;

        EVAL_TOKENS   = M_TOKENS;
        M_CHUNKS      = IS_DECODE ? 1 : (M_TOKENS / M_BATCH);
        M_Q           = IS_DECODE ? 1 : 8;
        Q_TILES       = IS_DECODE ? 1 : (M_TOKENS / M_Q);
        KV_TILES      = N_KV / 16;
        KV_TILES_PAST = (N_KV - M_TOKENS) / 16;
        NUM_VEC_XN    = EVAL_TOKENS * K_PACKETS_QKV;
        KV_PIN_EN     = ((KV_TILES * 8) <= 1024);

        Q_SUB = (Q_TILES > 8) ? 8 : Q_TILES;
        if ($value$plusargs("Q_SUB=%d", qs_arg) && (qs_arg > 0)) Q_SUB = qs_arg;

        DATA_DIR = $sformatf("sim_llama3_1b_%s_m%0d_kv%0d", IS_DECODE ? "decode" : "prefill", M_TOKENS, N_KV);
        void'($value$plusargs("DATA_DIR=%s", DATA_DIR));
        WEIGHTS_HEX = "sim_llama3_1b_weights/dram_weights_l1.hex";
        void'($value$plusargs("WEIGHTS_HEX=%s", WEIGHTS_HEX));

        if ((M_TOKENS > MAX_M) || (N_KV > MAX_KV) || (N_KV < M_TOKENS) || (N_KV % 16 != 0))
            $fatal(1, "Invalid dimensions M_TOKENS=%0d N_KV=%0d (MAX_M=%0d MAX_KV=%0d)", M_TOKENS, N_KV, MAX_M, MAX_KV);

        if ($test$plusargs("DUMP")) begin
            $dumpfile("tb_llama3_1b_configurable.fst");
            $dumpvars(0, dut);
        end

        csr_req_valid = 0; csr_req_write = 0; csr_req_addr = 0; csr_req_data = 0; csr_rsp_ready = 1;
        cpu_mem_req_valid = 0; cpu_mem_req_write = 0; cpu_mem_req_addr = 0;
        cpu_mem_req_wdata = 0; cpu_mem_req_strb = 0;
        checks_pass = 0; checks_fail = 0;
        cyc_total = 0; cyc_seq = 0; cyc_sfu = 0; cyc_core_compute = 0; cyc_dma = 0; job_cycles_total = 0;

        acc_n1_ref2   = 0.0; acc_n1_err2   = 0.0; acc_n1_dot   = 0.0; acc_n1_act2   = 0.0; acc_n1_max   = 0.0;
        acc_r1fq_ref2 = 0.0; acc_r1fq_err2 = 0.0; acc_r1fq_dot = 0.0; acc_r1fq_act2 = 0.0; acc_r1fq_max = 0.0;
        acc_r1fp_ref2 = 0.0; acc_r1fp_err2 = 0.0; acc_r1fp_dot = 0.0; acc_r1fp_act2 = 0.0; acc_r1fp_max = 0.0;
        acc_n2fq_ref2 = 0.0; acc_n2fq_err2 = 0.0; acc_n2fq_dot = 0.0; acc_n2fq_act2 = 0.0; acc_n2fq_max = 0.0;
        acc_n2fp_ref2 = 0.0; acc_n2fp_err2 = 0.0; acc_n2fp_dot = 0.0; acc_n2fp_act2 = 0.0; acc_n2fp_max = 0.0;
        acc_fnfq_ref2 = 0.0; acc_fnfq_err2 = 0.0; acc_fnfq_dot = 0.0; acc_fnfq_act2 = 0.0; acc_fnfq_max = 0.0;
        acc_fnfp_ref2 = 0.0; acc_fnfp_err2 = 0.0; acc_fnfp_dot = 0.0; acc_fnfp_act2 = 0.0; acc_fnfp_max = 0.0;

        n_fp16 = EVAL_TOKENS * D_MODEL;

        $display("Loading weights (%s) & stimulus (%s)...", WEIGHTS_HEX, DATA_DIR);
        $readmemh(WEIGHTS_HEX,                        dram_mem,    0, DRAM_WT_BEATS - 1);
        $readmemh({DATA_DIR, "/x_fp16.hex"},          x_fp16_mem,  0, EVAL_TOKENS * ROW_WORDS - 1);
        $readmemh({DATA_DIR, "/gamma1.hex"},          gamma1_mem,  0, ROW_WORDS - 1);
        $readmemh({DATA_DIR, "/gamma2.hex"},          gamma2_mem,  0, ROW_WORDS - 1);
        $readmemh({DATA_DIR, "/rope_mbatch.hex"},     rope_mb_mem, 0, (IS_DECODE ? 512 : M_CHUNKS * 256) - 1);
        $readmemh({DATA_DIR, "/g_xnorm.hex"},         g_xnorm,     0, n_fp16 - 1);
        $readmemh({DATA_DIR, "/g_res1.hex"},          g_res1,      0, n_fp16 - 1);
        $readmemh({DATA_DIR, "/g_n2.hex"},            g_n2,        0, n_fp16 - 1);
        $readmemh({DATA_DIR, "/g_final_fp16.hex"},    g_final_fp16,0, n_fp16 - 1);
        $readmemh({DATA_DIR, "/g_final_fp32.hex"},    g_final_fp32,0, n_fp16 - 1);
        $readmemh({DATA_DIR, "/g_res1_fq.hex"},       g_res1_fq,   0, n_fp16 - 1);
        $readmemh({DATA_DIR, "/g_n2_fq.hex"},         g_n2_fq,     0, n_fp16 - 1);
        $readmemh({DATA_DIR, "/g_final_fq.hex"},      g_final_fq,  0, n_fp16 - 1);

        if (IS_DECODE) begin
            $readmemh({DATA_DIR, "/k_past_words.hex"}, k_past_mem,  0, H_KV * KV_TILES * 64 - 1);
            $readmemh({DATA_DIR, "/v_past_words.hex"}, v_past_mem,  0, H_KV * DV_TILES * KV_TILES * 16 - 1);
            $readmemh({DATA_DIR, "/v_tail_past_fp16.hex"}, v_tail_fp16, 0, 15 * N_TILES_KV * 4 - 1);
            for (int w = 0; w < H_KV * KV_TILES * 64; w++)
                write_dram_9b(DRAM_K, w, k_past_mem[w]);
            for (int w = 0; w < H_KV * DV_TILES * KV_TILES * 16; w++)
                write_dram_9b(DRAM_V, w, v_past_mem[w]);
        end else if (KV_TILES_PAST > 0) begin
            $readmemh({DATA_DIR, "/k_past_words.hex"}, k_past_mem, 0, H_KV * KV_TILES_PAST * 64 - 1);
            $readmemh({DATA_DIR, "/v_past_words.hex"}, v_past_mem, 0, H_KV * DV_TILES * KV_TILES_PAST * 16 - 1);
            // Place past tiles at kj = 0 .. KV_TILES_PAST-1 inside full-context KV_TILES stride!
            for (int hkv = 0; hkv < H_KV; hkv++) begin
                for (int kj = 0; kj < KV_TILES_PAST; kj++) begin
                    for (int kp = 0; kp < K_PACKETS_QK; kp++)
                        for (int l = 0; l < 16; l++)
                            write_dram_9b(DRAM_K, ((hkv * KV_TILES + kj) * K_PACKETS_QK + kp) * 16 + l,
                                          k_past_mem[((hkv * KV_TILES_PAST + kj) * K_PACKETS_QK + kp) * 16 + l]);
                    for (int dvi = 0; dvi < DV_TILES; dvi++)
                        for (int l = 0; l < 16; l++)
                            write_dram_9b(DRAM_V, ((hkv * DV_TILES + dvi) * KV_TILES + kj) * 16 + l,
                                          v_past_mem[((hkv * DV_TILES + dvi) * KV_TILES_PAST + kj) * 16 + l]);
                end
            end
        end

        rst_n = 1'b0;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        cpu_csr_read(A_RO_ID, rd32);
        check(rd32 == CSR_ID_VALUE, "CSR ID (SNAF)");

        $display("\n========================================================================");
        $display("  REAL NousResearch/Llama-3.2-1B %s LAYER (288KB TCDM, 8-Port DMA)",
                 IS_DECODE ? "DECODE" : "PREFILL");
        $display("  M_TOKENS=%0d | N_KV=%0d (PAST=%0d) | Q_TILES=%0d (Q_SUB=%0d) | KV_PIN=%0d",
                 EVAL_TOKENS, N_KV, KV_TILES_PAST * 16, Q_TILES, Q_SUB, KV_PIN_EN);
        $display("========================================================================\n");

        if (IS_DECODE) begin
            // -----------------------------------------------------------------
            // DECODE FLOW (M = 1, N_KV = configurable)
            // -----------------------------------------------------------------
            for (int w = 0; w < ROW_WORDS; w++) begin
                cpu_mem_write(tcdm_addr(w, X_BANK),                  x_fp16_mem[w]);
                cpu_mem_write(tcdm_addr(GAMMA1_WORD + w, GAMMA_BANK), gamma1_mem[w]);
                cpu_mem_write(tcdm_addr(GAMMA2_WORD + w, GAMMA_BANK), gamma2_mem[w]);
            end
            for (int w = 0; w < 512; w++)
                cpu_mem_write(tcdm_addr(w, ROPE_BANK), rope_mb_mem[w]);

            $display("[1/12] SFU RMSNorm 1 (M=1, D=2048)...");
            run_sfu_seq(3'd2, 1, 1, K_PACKETS_QKV,
                        tcdm_addr(0, X_BANK),               ROW_WORDS * 32, 0,
                        tcdm_addr(GAMMA1_WORD, GAMMA_BANK), 0,              0,
                        tcdm_addr(0, XN_BANK),              ROW_WORDS * 32, 0, 32'h3A000000);
            accum_rows_1bank(XN_BANK, 1, 0, 0, acc_n1_ref2, acc_n1_err2, acc_n1_dot, acc_n1_act2, acc_n1_max);

            $display("[2/12] HW Requant Xn -> ACT_BANK...");
            run_requant_addr(tcdm_addr(0, XN_BANK), NUM_VEC_XN, tcdm_addr(0, ACT_BANK), "Requant Xn");

            $display("[3/12] Fused Q (2048x2048, DMA) + RoPE + LogQ...");
            run_fused_qk_rope_dma_ex(N_TILES_MODEL, DRAM_WQ, tcdm_addr(0, ACT_BANK),
                                     tcdm_addr(0, OUT_BANK), tcdm_addr(0, ROPE_BANK), 1'b0, "Fused Q+RoPE");
            for (int h = 0; h < H_Q; h++)
                for (int kp = 0; kp < K_PACKETS_QK; kp++) begin
                    logic [71:0] qp;
                    cpu_mem_read(tcdm_addr(h*K_PACKETS_QK + kp, OUT_BANK), qp);
                    cpu_mem_write(tcdm_addr(h*K_PACKETS_QK + kp, QPKT_BANK), qp);
                end

            $display("[4/12] Fused K (512x2048, GQA H_KV=8, DMA) + RoPE + LogQ...");
            run_fused_qk_rope_dma_ex(N_TILES_KV, DRAM_WK, tcdm_addr(0, ACT_BANK),
                                     tcdm_addr(0, OUT_BANK), tcdm_addr(0, ROPE_BANK), 1'b0, "Fused K+RoPE");

            $display("[5/12] Decode GEMM V (FP16) + HW Decode V-Transpose (op=2'b10)...");
            for (int tok = 0; tok < 15; tok++)
                for (int w = 0; w < N_TILES_KV * 4; w++)
                    cpu_mem_write(tcdm_addr(640 + w, tok), v_tail_fp16[tok * (N_TILES_KV * 4) + w]);
            run_gemm_dma_ex(N_TILES_KV, K_PACKETS_QKV, 0, DRAM_WV, K_PACKETS_QKV * 144,
                            tcdm_addr(0, ACT_BANK), tcdm_addr(640, 15), 1'b0, "Decode GEMM V", 1'b0, 2'd0);

            cpu_csr_write(A_CTRL,     32'h0);
            cpu_csr_write(A_RD_LS,    32'd0);
            cpu_csr_write(A_SEQ_CTRL, (32'd1 << 24) | 32'h2);
            cpu_csr_write(A_Q_BASE,   tcdm_addr(640, 0));
            cpu_csr_write(A_V_BASE,   tcdm_addr(128, 0));
            cpu_csr_write(A_V_NB,     N_TILES_KV);
            cpu_csr_write(A_TILE_ST,  32'h200);
            cpu_csr_write(A_START,    32'h0000_0003);
            wait_job("Decode V-Transpose");

            for (int hkv = 0; hkv < H_KV; hkv++) begin
                for (int kp = 0; kp < K_PACKETS_QK; kp++) begin
                    logic [71:0] kp_pkt;
                    cpu_mem_read(tcdm_addr(hkv*K_PACKETS_QK + kp, OUT_BANK), kp_pkt);
                    write_dram_9b(DRAM_K, ((hkv * KV_TILES + (KV_TILES - 1)) * K_PACKETS_QK + kp) * 16 + 15, kp_pkt);
                end
                for (int dvi = 0; dvi < DV_TILES; dvi++)
                    for (int l = 0; l < 16; l++) begin
                        logic [71:0] vp_pkt;
                        cpu_mem_read(tcdm_addr(128 + hkv * DV_TILES + dvi, l), vp_pkt);
                        write_dram_9b(DRAM_V, ((hkv * DV_TILES + dvi) * KV_TILES + (KV_TILES - 1)) * 16 + l, vp_pkt);
                    end
            end

            t_attn = cyc_total;
            $display("[6/12] HW Decode FlashAttention (N_KV=%0d, Sep-KV + KV_Pin=%0d)...", N_KV, KV_PIN_EN);
            clear_ext_macro_csrs();
            cpu_csr_write(A_CTRL,         32'h0);
            cpu_csr_write(A_EXT_MACRO_SUB,(32'd1 << 17) | (32'(KV_PIN_EN) << 16)); // sep_kv=1, kv_pin_en
            cpu_csr_write(A_EXT_DRAM_V,   DRAM_V);
            cpu_csr_write(A_SEQ_CTRL,     (32'(GQA_M1) << 16) | 32'h29); // op=01, dma=1, causal=0, 4D FP4 out, GQA 4:1
            cpu_csr_write(A_ATTN_H,       H_Q);
            cpu_csr_write(A_ATTN_QT,      1);
            cpu_csr_write(A_ATTN_MQ,      1);
            cpu_csr_write(A_ATTN_DV,      DV_TILES);
            cpu_csr_write(A_ATTN_KV,      KV_TILES);
            cpu_csr_write(A_ATTN_KB,      K_PACKETS_QK);
            cpu_csr_write(A_RING_BASE,    32'h0);
            cpu_csr_write(A_DRAM_KV,      DRAM_K);
            cpu_csr_write(A_TILE_ST,      32'd720);
            cpu_csr_write(A_Q_BASE,       tcdm_addr(0, QPKT_BANK));
            cpu_csr_write(A_OUT_BASE,     tcdm_addr(0, O_ACT_BANK));
            cpu_csr_write(A_START,        32'h0000_0003);
            wait_job("Decode FlashAttention");
            clear_ext_macro_csrs();
            $display("       Attn cycles=%0d", cyc_total - t_attn);

            $display("[7/12] GEMM O (2048x2048, DMA)...");
            run_gemm_dma_ex(N_TILES_MODEL, K_PACKETS_QKV, 0, DRAM_WO, K_PACKETS_QKV * 144,
                            tcdm_addr(0, O_ACT_BANK), tcdm_addr(0, OUT_BANK), 1'b0, "GEMM O");

            $display("[8/12] SFU Residual 1 + RMSNorm 2...");
            run_sfu_seq(3'd1, 1, N_TILES_MODEL, 1,
                        tcdm_addr(0, OUT_BANK), 4 * 32,         4 * 32,
                        tcdm_addr(0, X_BANK),   ROW_WORDS * 32, 4 * 32,
                        tcdm_addr(0, RES1_BANK),ROW_WORDS * 32, 4 * 32, 32'h3F800000);
            accum_rows_1bank(RES1_BANK, 1, 0, 1, acc_r1fq_ref2, acc_r1fq_err2, acc_r1fq_dot, acc_r1fq_act2, acc_r1fq_max);
            accum_rows_1bank(RES1_BANK, 1, 0, 2, acc_r1fp_ref2, acc_r1fp_err2, acc_r1fp_dot, acc_r1fp_act2, acc_r1fp_max);

            run_sfu_seq(3'd2, 1, 1, K_PACKETS_QKV,
                        tcdm_addr(0, RES1_BANK),            ROW_WORDS * 32, 0,
                        tcdm_addr(GAMMA2_WORD, GAMMA_BANK), 0,              0,
                        tcdm_addr(0, XN_BANK),              ROW_WORDS * 32, 0, 32'h3A000000);
            accum_rows_1bank(XN_BANK, 1, 0, 3, acc_n2fq_ref2, acc_n2fq_err2, acc_n2fq_dot, acc_n2fq_act2, acc_n2fq_max);
            accum_rows_1bank(XN_BANK, 1, 0, 4, acc_n2fp_ref2, acc_n2fp_err2, acc_n2fp_dot, acc_n2fp_act2, acc_n2fp_max);

            $display("[9/12] HW Requant Norm2 -> ACT_BANK...");
            run_requant_addr(tcdm_addr(0, XN_BANK), NUM_VEC_XN, tcdm_addr(0, ACT_BANK), "Requant Norm2");

            $display("[10/12] FFN Gate & Up (32 chunks) + SiLU*Up...");
            for (int c = 0; c < N_CHUNKS; c++) begin
                run_gemm_dma_ex(CHUNK_NT, K_PACKETS_QKV, 0,
                                DRAM_WG + 32'(c) * 32'(CHUNK_NT) * 32'(K_PACKETS_QKV) * 32'd144,
                                K_PACKETS_QKV * 144,
                                tcdm_addr(0, ACT_BANK), tcdm_addr(0, G_STG_BANK), 1'b0, "GEMM Gate");
                run_gemm_dma_ex(CHUNK_NT, K_PACKETS_QKV, 0,
                                DRAM_WU + 32'(c) * 32'(CHUNK_NT) * 32'(K_PACKETS_QKV) * 32'd144,
                                K_PACKETS_QKV * 144,
                                tcdm_addr(0, ACT_BANK), tcdm_addr(0, OUT_BANK), 1'b0, "GEMM Up");
                run_sfu_seq(3'd4, 1, CHUNK_NT, 1,
                            tcdm_addr(0, G_STG_BANK),           4 * 32, 4 * 32,
                            tcdm_addr(0, OUT_BANK),             4 * 32, 4 * 32,
                            tcdm_addr(c * CHUNK_NT, FF1_BANK),  K_PACKETS_DOWN * 32, 32, 32'h3F800000, 1'b1);
            end

            $display("[11/12] GEMM Down (2048x8192, K-Tiling 2x4096)...");
            run_gemm_dma_ex(N_TILES_MODEL, K_TILE_DOWN, 1,
                            DRAM_WD, K_PACKETS_DOWN * 144,
                            tcdm_addr(0, FF1_BANK), tcdm_addr(0, OUT_BANK), 1'b0, "GEMM Down");

            $display("[12/12] SFU Residual 2 -> DRAM WB Verify...");
            run_sfu_seq(3'd1, 1, N_TILES_MODEL, 1,
                        tcdm_addr(0, OUT_BANK),  4 * 32,         4 * 32,
                        tcdm_addr(0, RES1_BANK), ROW_WORDS * 32, 4 * 32,
                        tcdm_addr(0, FIN_BANK),  ROW_WORDS * 32, 4 * 32, 32'h3F800000);
            accum_rows_1bank(FIN_BANK, 1, 0, 5, acc_fnfq_ref2, acc_fnfq_err2, acc_fnfq_dot, acc_fnfq_act2, acc_fnfq_max);
            accum_final_fp32_1bank(FIN_BANK, 1, 0);

            run_dma_writeback(2'b11, tcdm_addr(0, FIN_BANK), DRAM_OUT, 512 * 9, "DMA WB Final Out");
            verify_dram_vs_tcdm(1'b1, 0, FIN_BANK, DRAM_OUT, 512, "Verify DRAM Final Out");
        end else begin
            // -----------------------------------------------------------------
            // PREFILL FLOW (Arbitrary M = 16 * M_CHUNKS, N_KV >= M)
            //   1) M-batched Pre-Attn projections -> DRAM_QPKT / DRAM_K / DRAM_V
            //   2) ONE single Sequencer launch for full FlashAttention (Macro Q/O + KV-Pin SepKV + Causal Offset)
            //   3) M-batched Post-Attn & FFN -> DRAM_OUT + bit-exact verify
            // -----------------------------------------------------------------
            t0 = cyc_total;
            $display("[1/12 .. 5/12] Pre-Attention M-batched (%0d x 16 tokens)...", M_CHUNKS);
            for (int mb = 0; mb < M_CHUNKS; mb++) begin
                for (int t = 0; t < M_BATCH; t++)
                    for (int w = 0; w < ROW_WORDS; w++)
                        cpu_mem_write(tcdm_addr(w, 16 + t), x_fp16_mem[(mb * M_BATCH + t) * ROW_WORDS + w]);
                for (int w = 0; w < ROW_WORDS; w++)
                    cpu_mem_write(tcdm_addr(w, 0), gamma1_mem[w]);

                run_sfu_seq(3'd2, M_BATCH, 1, K_PACKETS_QKV,
                            tcdm_addr(0, 16), 1, 0, tcdm_addr(0, 0), 0, 0,
                            tcdm_addr(512, 16), 1, 0, 32'h3A000000);
                accum_rows_16bank(512, M_BATCH, mb * M_BATCH, 0,
                                  acc_n1_ref2, acc_n1_err2, acc_n1_dot, acc_n1_act2, acc_n1_max);
                run_dma_writeback(2'b01, tcdm_addr(0, 16), DRAM_X + 32'(mb) * 32'(8192 * 9), 8192 * 9, "DMA Save X");

                run_requant_wrap(tcdm_addr(512, 16), M_BATCH * K_PACKETS_QKV, tcdm_addr(0, 16), 511, 127, "Requant Xn");
                for (int i = 0; i < 256; i++)
                    cpu_mem_write(tcdm_addr(128 + i, ROPE_BANK), rope_mb_mem[mb * 256 + i]);

                // Fused Q+RoPE -> pack into DRAM_QPKT in [h][t_abs][kp] order
                run_fused_qk_rope_dma_ex(N_TILES_MODEL, DRAM_WQ, tcdm_addr(0, 16), tcdm_addr(128, 16),
                                         tcdm_addr(128, ROPE_BANK), 1'b1, "Fused Q+RoPE", 1'b1, 255, 255);
                for (int h = 0; h < H_Q; h++)
                    for (int t = 0; t < M_BATCH; t++)
                        for (int kp = 0; kp < K_PACKETS_QK; kp++) begin
                            logic [71:0] qp;
                            int lin;
                            lin = (h * K_PACKETS_QK + kp) * M_BATCH + t;
                            cpu_mem_read(tcdm_addr(128 + (lin & 255), 16 + (lin >> 8)), qp);
                            write_dram_9b(DRAM_QPKT, (h * M_TOKENS + mb * M_BATCH + t) * K_PACKETS_QK + kp, qp);
                        end

                // Fused K+RoPE -> pack into DRAM_K at kj = KV_TILES_PAST + mb
                run_fused_qk_rope_dma_ex(N_TILES_KV, DRAM_WK, tcdm_addr(0, 16), tcdm_addr(128, 16),
                                         tcdm_addr(128, ROPE_BANK), 1'b1, "Fused K+RoPE", 1'b1, 255, 255);
                for (int hkv = 0; hkv < H_KV; hkv++)
                    for (int kp = 0; kp < K_PACKETS_QK; kp++)
                        for (int l = 0; l < 16; l++) begin
                            logic [71:0] kp_pkt;
                            int lin;
                            lin = (hkv * K_PACKETS_QK + kp) * 16 + l;
                            cpu_mem_read(tcdm_addr(128 + (lin & 255), 16 + (lin >> 8)), kp_pkt);
                            write_dram_9b(DRAM_K, ((hkv * KV_TILES + (KV_TILES_PAST + mb)) * K_PACKETS_QK + kp) * 16 + l, kp_pkt);
                        end

                // Fused V+Transpose -> pack into DRAM_V at kj = KV_TILES_PAST + mb
                for (int hkv = 0; hkv < H_KV; hkv++)
                    for (int dvi = 0; dvi < DV_TILES; dvi++) begin
                        run_gemm_v_dma_ex(hkv * DV_TILES + dvi, 512 + hkv * DV_TILES + dvi,
                                          32'h0, tcdm_addr(0, 16), 1'b1, "GEMM V");
                        for (int l = 0; l < 16; l++) begin
                            logic [71:0] vp_pkt;
                            cpu_mem_read(tcdm_addr(512 + hkv * DV_TILES + dvi, l), vp_pkt);
                            write_dram_9b(DRAM_V, ((hkv * DV_TILES + dvi) * KV_TILES + (KV_TILES_PAST + mb)) * 16 + l, vp_pkt);
                        end
                    end
            end
            $display("       Pre-Attn cycles=%0d", cyc_total - t0);

            t_attn = cyc_total;
            $display("[6/12] Single-Launch HW Causal FlashAttention (M=%0d, N_KV=%0d, Q_T=%0d, Q_SUB=%0d, Sep-KV, KV_Pin=%0d)...",
                     M_TOKENS, N_KV, Q_TILES, Q_SUB, KV_PIN_EN);
            clear_ext_macro_csrs();
            cpu_csr_write(A_CTRL,           32'h0);
            cpu_csr_write(A_EXT_MACRO_SUB,  (32'(KV_TILES_PAST[13:0]) << 18) | (32'd1 << 17) | (32'(KV_PIN_EN) << 16) | 32'(Q_SUB[7:0]));
            cpu_csr_write(A_EXT_MACRO_ACT,  DRAM_QPKT);
            cpu_csr_write(A_EXT_MACRO_OUT,  DRAM_OPKT);
            cpu_csr_write(A_EXT_MACRO_ASTR, 32'h0004_0000); // Bank 30 words 0..255 <-> 512..767
            cpu_csr_write(A_EXT_MACRO_OSTR, 32'h0004_0000); // Bank 28 words 0..255 <-> 512..767
            cpu_csr_write(A_EXT_DRAM_V,     DRAM_V);
            cpu_csr_write(A_SEQ_CTRL,       (32'd1 << 30) | (32'(GQA_M1) << 16) | 32'h2D); // macro_dma=1, op=01, dma=1, causal=1, out=2 (4D FP4)
            cpu_csr_write(A_ATTN_H,         H_Q);
            cpu_csr_write(A_ATTN_QT,        Q_TILES);
            cpu_csr_write(A_ATTN_MQ,        M_Q);
            cpu_csr_write(A_ATTN_DV,        DV_TILES);
            cpu_csr_write(A_ATTN_KV,        KV_TILES);
            cpu_csr_write(A_ATTN_KB,        K_PACKETS_QK);
            cpu_csr_write(A_RING_BASE,      tcdm_addr(0, 0));
            cpu_csr_write(A_DRAM_KV,        DRAM_K);
            cpu_csr_write(A_TILE_ST,        32'd720);
            cpu_csr_write(A_Q_BASE,         tcdm_addr(0, 30));
            cpu_csr_write(A_OUT_BASE,       tcdm_addr(0, 28));
            cpu_csr_write(A_START,          32'h0000_0003);
            wait_job("FlashAttention Single-Launch");
            clear_ext_macro_csrs();
            $display("       Attn cycles=%0d", cyc_total - t_attn);

            t0 = cyc_total;
            $display("[7/12 .. 12/12] Post-Attention & FFN M-batched (%0d x 16 tokens)...", M_CHUNKS);
            for (int mb = 0; mb < M_CHUNKS; mb++) begin
                // Scatter Attn O for chunk mb from DRAM_OPKT ([h][t_abs][dvi]) to banks 16..31 words 0..127
                for (int t = 0; t < M_BATCH; t++)
                    for (int h = 0; h < H_Q; h++)
                        for (int dvi = 0; dvi < DV_TILES; dvi++) begin
                            logic [71:0] opkt;
                            read_dram_9b(DRAM_OPKT, (h * M_TOKENS + mb * M_BATCH + t) * DV_TILES + dvi, opkt);
                            cpu_mem_write(tcdm_addr(h * DV_TILES + dvi, 16 + t), opkt);
                        end

                run_dma_writeback(2'b00, tcdm_addr(512, 16), DRAM_X + 32'(mb) * 32'(8192 * 9), 8192 * 9, "DMA Load X");
                for (int c = 0; c < 2; c++) begin
                    run_gemm_dma_ex(64, K_PACKETS_QKV, 0,
                                    DRAM_WO + 32'(c) * 32'd64 * 32'(K_PACKETS_QKV) * 32'd144,
                                    K_PACKETS_QKV * 144,
                                    tcdm_addr(0, 16), tcdm_addr(128, 16), 1'b1, "GEMM O", 1'b1);
                    run_sfu_seq(3'd1, M_BATCH, 64, 1,
                                tcdm_addr(128, 16),         1, 4 * 32,
                                tcdm_addr(512 + c*256, 16), 1, 4 * 32,
                                tcdm_addr(512 + c*256, 16), 1, 4 * 32, 32'h3F800000);
                end
                accum_rows_16bank(512, M_BATCH, mb * M_BATCH, 1,
                                  acc_r1fq_ref2, acc_r1fq_err2, acc_r1fq_dot, acc_r1fq_act2, acc_r1fq_max);
                accum_rows_16bank(512, M_BATCH, mb * M_BATCH, 2,
                                  acc_r1fp_ref2, acc_r1fp_err2, acc_r1fp_dot, acc_r1fp_act2, acc_r1fp_max);

                run_dma_writeback(2'b01, tcdm_addr(512, 16), DRAM_RES1 + 32'(mb) * 32'(8192 * 9), 8192 * 9, "DMA Save RES1");
                for (int w = 0; w < ROW_WORDS; w++)
                    cpu_mem_write(tcdm_addr(w, 0), gamma2_mem[w]);
                run_sfu_seq(3'd2, M_BATCH, 1, K_PACKETS_QKV,
                            tcdm_addr(512, 16), 1, 0, tcdm_addr(0, 0), 0, 0,
                            tcdm_addr(0, 16),   1, 0, 32'h3A000000);
                accum_rows_16bank(0, M_BATCH, mb * M_BATCH, 3,
                                  acc_n2fq_ref2, acc_n2fq_err2, acc_n2fq_dot, acc_n2fq_act2, acc_n2fq_max);
                accum_rows_16bank(0, M_BATCH, mb * M_BATCH, 4,
                                  acc_n2fp_ref2, acc_n2fp_err2, acc_n2fp_dot, acc_n2fp_act2, acc_n2fp_max);

                run_requant_wrap(tcdm_addr(0, 16), M_BATCH * K_PACKETS_QKV, tcdm_addr(0, 16), 511, 127, "Requant Norm2");
                for (int c = 0; c < N_CHUNKS; c++) begin
                    run_gemm_dma_ex(CHUNK_NT, K_PACKETS_QKV, 0,
                                    DRAM_WG + 32'(c) * 32'(CHUNK_NT) * 32'(K_PACKETS_QKV) * 32'd144,
                                    K_PACKETS_QKV * 144,
                                    tcdm_addr(0, 16), tcdm_addr(128, 16), 1'b1, "GEMM Gate", 1'b1);
                    run_gemm_dma_ex(CHUNK_NT, K_PACKETS_QKV, 0,
                                    DRAM_WU + 32'(c) * 32'(CHUNK_NT) * 32'(K_PACKETS_QKV) * 32'd144,
                                    K_PACKETS_QKV * 144,
                                    tcdm_addr(0, 16), tcdm_addr(256, 16), 1'b1, "GEMM Up", 1'b1);
                    run_sfu_seq(3'd4, M_BATCH, CHUNK_NT, 1,
                                tcdm_addr(128, 16),             1, 4 * 32,
                                tcdm_addr(256, 16),             1, 4 * 32,
                                tcdm_addr(512 + c*CHUNK_NT, 16),1, 32, 32'h3F800000, 1'b1);
                end

                run_gemm_dma_ex(N_TILES_MODEL, K_TILE_DOWN, 1,
                                DRAM_WD, K_PACKETS_DOWN * 144,
                                tcdm_addr(512, 16), tcdm_addr(0, 16), 1'b1, "GEMM Down", 1'b1);
                run_dma_writeback(2'b00, tcdm_addr(512, 16), DRAM_RES1 + 32'(mb) * 32'(8192 * 9), 8192 * 9, "DMA Load RES1");
                run_sfu_seq(3'd1, M_BATCH, N_TILES_MODEL, 1,
                            tcdm_addr(0, 16),   1, 4 * 32,
                            tcdm_addr(512, 16), 1, 4 * 32,
                            tcdm_addr(0, 16),   1, 4 * 32, 32'h3F800000);

                accum_rows_16bank(0, M_BATCH, mb * M_BATCH, 5,
                                  acc_fnfq_ref2, acc_fnfq_err2, acc_fnfq_dot, acc_fnfq_act2, acc_fnfq_max);
                accum_final_fp32_16bank(0, M_BATCH, mb * M_BATCH);

                run_dma_writeback(2'b01, tcdm_addr(0, 16), DRAM_OUT + 32'(mb) * 32'(8192 * 9), 8192 * 9, "DMA WB Final Out");
                verify_dram_vs_tcdm(1'b0, 0, 16,           DRAM_OUT + 32'(mb) * 32'(8192 * 9), 8192,     "Verify DRAM Final Out");
            end
            $display("       Post-Attn/FFN cycles=%0d", cyc_total - t0);
        end

        $display("\n========================================================================");
        $display("  SEQUENCE-WIDE ACCURACY & PERFORMANCE SUMMARY (%s M=%0d, N_KV=%0d)",
                 IS_DECODE ? "DECODE" : "PREFILL", EVAL_TOKENS, N_KV);
        $display("========================================================================");
        report_metric("RMSNorm 1 vs FP16",  acc_n1_ref2,   acc_n1_err2,   acc_n1_dot,   acc_n1_act2,   acc_n1_max,   0.998);
        report_metric("Residual 1 vs PyFQ", acc_r1fq_ref2, acc_r1fq_err2, acc_r1fq_dot, acc_r1fq_act2, acc_r1fq_max, 0.90);
        report_metric("Residual 1 vs FP32", acc_r1fp_ref2, acc_r1fp_err2, acc_r1fp_dot, acc_r1fp_act2, acc_r1fp_max, 0.85);
        report_metric("RMSNorm 2 vs PyFQ",  acc_n2fq_ref2, acc_n2fq_err2, acc_n2fq_dot, acc_n2fq_act2, acc_n2fq_max, 0.90);
        report_metric("RMSNorm 2 vs FP32",  acc_n2fp_ref2, acc_n2fp_err2, acc_n2fp_dot, acc_n2fp_act2, acc_n2fp_max, 0.85);
        report_metric("Final vs PyFQ",      acc_fnfq_ref2, acc_fnfq_err2, acc_fnfq_dot, acc_fnfq_act2, acc_fnfq_max, 0.85);
        report_metric("Final vs FP32",      acc_fnfp_ref2, acc_fnfp_err2, acc_fnfp_dot, acc_fnfp_act2, acc_fnfp_max, 0.78);
        $display("  Total cycles      : %0d (seq=%0d, sfu=%0d, mac=%0d, dma=%0d)",
                 cyc_total, cyc_seq, cyc_sfu, cyc_core_compute, cyc_dma);
        $display("  DRAM 512b rd beats: %0d (%0d bytes)", stat_rd_beats, stat_rd_beats * 64);
        $display("  DRAM 512b wr beats: %0d (%0d bytes)", stat_wr_beats, stat_wr_beats * 64);
        $display("  Checks            : %0d passed, %0d failed", checks_pass, checks_fail);
        $display("========================================================================\n");

        if (checks_fail == 0)
            $display(">>> CONFIGURABLE LLAMA-3.2-1B %s (M=%0d, N_KV=%0d) PASSED (0 errors) <<<\n",
                     IS_DECODE ? "DECODE" : "PREFILL", EVAL_TOKENS, N_KV);
        else
            $fatal(1, "CONFIGURABLE LLAMA-3.2-1B FAILED (%0d check fails)", checks_fail);

        $finish;
    end

endmodule
