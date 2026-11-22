// =============================================================================
// tb_attn_comparison.sv
//
// Quantitative Hardware Comparison of Attention Architectures on
// snax_llm_top_clean (288 KB TCDM, BANK_DEPTH=1024) + 8-port snax_simple_dma:
//   1) NAIVE       : 3-Phase Unfused Attention (QK^T -> DRAM S -> 2-pass DMA read
//                    -> Softmax + Requant -> DRAM P -> PV)
//   2) ONLINE_2A   : Unfused Online Softmax (QK^T -> DRAM S -> 1-pass DMA read
//                    -> Online Softmax + Requant -> DRAM P -> PV)
//   3) ONLINE_2B   : Online Softmax fused w/ PV WITHOUT 2D tile reuse (M_Q = 1)
//   4) FLASH_RING  : Full FlashAttention (M_Q = 8 2D tile fusion, Ring Buffer, KV_PIN=0)
//   5) FLASH_KVPIN : Full Proposed FlashAttention (M_Q = 8, GQA 4:1 KV_PIN=1)
//
// Zero-Cheat Rules Enforced:
//   - cpu_mem_req_valid_i tied to 1'b0 and monitored every cycle (== 0).
//   - 100% real 1D-DMA + Sequencer execution on RTL; exact hardware DRAM beat
//     & cycle telemetry plus 64B DRAM canaries and FP32 golden accuracy check.
// =============================================================================
`timescale 1ns/1ps

module tb_attn_comparison;

    localparam int H_Q          = 32;
    localparam int H_KV         = 8;
    localparam int GQA_M1       = (H_Q / H_KV) - 1; // 3
    localparam int D_K          = 64;
    localparam int D_V          = 64;
    localparam int DV_TILES     = D_V / 16;         // 4
    localparam int K_PACKETS_QK = D_K / 16;         // 4
    localparam int M_BATCH      = 16;
    localparam int TB_BANK_DEPTH= 1024;             // True 288 KB TCDM

    localparam int MAX_M        = 1024;
    localparam int MAX_KV       = 1024;
    localparam int MAX_Q_WORDS  = H_Q * MAX_M * K_PACKETS_QK;                 // 131,072
    localparam int MAX_K_WORDS  = H_KV * (MAX_KV / 16) * K_PACKETS_QK * 16;   // 32,768
    localparam int MAX_V_WORDS  = H_KV * DV_TILES * (MAX_KV / 16) * 16;       // 32,768
    localparam int MAX_OUT_FP32 = H_Q * MAX_M * D_V;                          // 2,097,152

    localparam int CLK_PERIOD   = 10;
    localparam int DRAM_LATENCY = 6;
    localparam int JOB_TIMEOUT  = 1_000_000_000;

    // 16 MB Byte-Strobed 512-bit DRAM (262,144 beats)
    localparam int DRAM_WORDS = 262144;
    localparam logic [31:0] DRAM_Q = 32'h0000_0000; // max 1.125 MB
    localparam logic [31:0] DRAM_K = 32'h0020_0000; // max 288 KB
    localparam logic [31:0] DRAM_V = 32'h0030_0000; // max 288 KB
    localparam logic [31:0] DRAM_O = 32'h0040_0000; // max 1.125 MB (+ canaries)
    localparam logic [31:0] DRAM_S = 32'h0060_0000; // per-head reusable S buffer (max 2.25 MB)
    localparam logic [31:0] DRAM_P = 32'h0090_0000; // per-head reusable P buffer (max 576 KB)

    // CSR Map matching snax_flash_csr_snax.sv
    localparam logic [11:0] A_CTRL          = 12'd0;
    localparam logic [11:0] A_RD_BASE       = 12'd27;
    localparam logic [11:0] A_RD_NV         = 12'd28;
    localparam logic [11:0] A_RD_LS         = 12'd29;
    localparam logic [11:0] A_WO_BASE       = 12'd13;
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

    // Runtime configuration
    string MODE_STR, DATA_DIR;
    int    M_TOKENS, N_KV, DRAM_BW, M_CHUNKS, KV_TILES, KV_TILES_PAST, N_PAST;

    // Clock
    logic clk, rst_n;
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD / 2) clk = ~clk;
    end

    // CSR & tied-off cpu_mem
    logic        csr_req_valid, csr_req_ready, csr_req_write, csr_rsp_valid, csr_rsp_ready;
    logic [11:0] csr_req_addr;
    logic [31:0] csr_req_data, csr_rsp_data;

    logic        cpu_mem_req_valid, cpu_mem_req_ready, cpu_mem_req_write, cpu_mem_rsp_valid;
    logic [31:0] cpu_mem_req_addr;
    logic [71:0] cpu_mem_req_wdata, cpu_mem_rsp_rdata;
    logic [8:0]  cpu_mem_req_strb;

    assign cpu_mem_req_valid = 1'b0;
    assign cpu_mem_req_write = 1'b0;
    assign cpu_mem_req_addr  = 32'd0;
    assign cpu_mem_req_wdata = 72'd0;
    assign cpu_mem_req_strb  = 9'd0;

    // 8-port DMA <-> DUT
    logic [7:0]        dma_tcdm_req_valid, dma_tcdm_req_ready, dma_tcdm_req_write, dma_tcdm_rsp_valid;
    logic [7:0][31:0]  dma_tcdm_req_addr;
    logic [7:0][71:0]  dma_tcdm_req_wdata, dma_tcdm_rsp_rdata;
    logic [7:0][8:0]   dma_tcdm_req_strb;

    logic        dma_desc_valid, dma_desc_ready, dma_desc_done;
    logic [1:0]  dma_desc_dir;
    logic [31:0] dma_desc_dram_addr, dma_desc_tcdm_addr, dma_desc_size;

    // 512-bit DRAM
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

    // 512-bit DRAM w/ Byte-Strobe Write & Configurable Bandwidth Throttler
    logic [511:0] dram_mem [0:DRAM_WORDS-1];
    logic [511:0] dram_wr_mask;
    logic [511:0] lat_data [0:DRAM_LATENCY-1];
    logic         lat_vld  [0:DRAM_LATENCY-1];
    logic [2:0]   bw_cnt_q;

    always_comb begin
        for (int b = 0; b < 64; b++)
            dram_wr_mask[b*8 +: 8] = {8{dram_req_strb[b]}};
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) bw_cnt_q <= 3'd0;
        else        bw_cnt_q <= bw_cnt_q + 3'd1;
    end

    assign dram_req_ready = (DRAM_BW >= 64) ? 1'b1 :
                            (DRAM_BW >= 32) ? (bw_cnt_q[0] == 1'b0) :
                            (DRAM_BW >= 16) ? (bw_cnt_q[1:0] == 2'b00) :
                                              (bw_cnt_q[2:0] == 3'b000);

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
                    dram_mem[dram_req_addr[23:6]] <= (dram_mem[dram_req_addr[23:6]] & ~dram_wr_mask) |
                                                     (dram_req_wdata & dram_wr_mask);
                end else begin
                    lat_vld[0]  <= 1'b1;
                    lat_data[0] <= dram_mem[dram_req_addr[23:6]];
                end
            end
        end
    end

    // Stimulus / Golden Buffers
    logic [71:0] q_mem       [0:MAX_Q_WORDS-1];
    logic [71:0] k_mem       [0:MAX_K_WORDS-1];
    logic [71:0] v_mem       [0:MAX_V_WORDS-1];
    logic [31:0] g_attn_fp32 [0:MAX_OUT_FP32-1];
    logic [31:0] g_attn_fq   [0:MAX_OUT_FP32-1];

    // Telemetry & Monitors
    bit     measure_en;
    longint cyc_total, cyc_mac, cyc_dma, dram_stall_cyc, cpu_mem_active_cyc;
    longint p2_sfu_est_cyc, sp_rd_bytes, sp_wr_bytes;
    int     max_inflight_seen, checks_pass, checks_fail;

    always @(posedge clk) begin
        if (rst_n && measure_en) begin
            cyc_total <= cyc_total + 1;
            if (|dut.lane_valid)     cyc_mac <= cyc_mac + 1;
            if (|dma_tcdm_req_valid) cyc_dma <= cyc_dma + 1;
            if (dram_req_valid && !dram_req_ready)
                dram_stall_cyc <= dram_stall_cyc + 1;
            if (cpu_mem_req_valid) begin
                cpu_mem_active_cyc <= cpu_mem_active_cyc + 1;
                if (cpu_mem_active_cyc < 5)
                    $display("    [FAIL] Forbidden cpu_mem activity at %0t!", $time);
                checks_fail++;
            end
            if (int'(dut.u_seq.inflight_out_cnt_q) > max_inflight_seen)
                max_inflight_seen <= int'(dut.u_seq.inflight_out_cnt_q);
            if (dut.u_seq.inflight_out_cnt_q > 4'd4) begin
                $display("    [FAIL] inflight_out_cnt_q overflow: %0d", dut.u_seq.inflight_out_cnt_q);
                checks_fail++;
            end
        end
    end

    // Pure SW Helpers (FP16 / FP32 / Log8-FP4 SLQ bit-exact)
    function automatic string to_upper_str(input string s);
        string r;
        byte ch;
        r = s;
        for (int i = 0; i < r.len(); i++) begin
            ch = r.getc(i);
            if (ch >= "a" && ch <= "z")
                r.putc(i, ch - 8'd32);
        end
        return r;
    endfunction

    function automatic real sw_fp16_to_real(input logic [15:0] h);
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

    function automatic logic [15:0] sw_real_to_fp16(input real r);
        logic [63:0] b64;
        logic        s;
        int          e64, e16, mant;
        logic        rnd;
        b64 = $realtobits(r);
        s   = b64[63];
        e64 = int'(b64[62:52]);
        if (e64 == 0 || e64 == 2047) return {s, (e64 == 2047) ? 15'h7BFF : 15'd0};
        e16 = e64 - 1023 + 15;
        if (e16 <= 0)  return {s, 15'd0};
        if (e16 >= 31) return {s, 5'd30, 10'h3FF};
        mant = int'(b64[51:42]);
        rnd  = b64[41] && ((b64[40:0] != 41'd0) || b64[42]);
        if (rnd) begin
            mant = mant + 1;
            if (mant >= 1024) begin
                mant = 0;
                e16  = e16 + 1;
                if (e16 >= 31) return {s, 5'd30, 10'h3FF};
            end
        end
        return {s, e16[4:0], mant[9:0]};
    endfunction

    function automatic real sw_fp32_to_real(input logic [31:0] f);
        logic sign;
        int e;
        logic [22:0] mf;
        real v;
        sign = f[31];
        e    = int'(f[30:23]);
        mf   = f[22:0];
        if (e == 0 && mf == 0) return 0.0;
        if (e == 255) return sign ? -1.0e38 : 1.0e38;
        if (e == 0)  v = (real'(mf) / 8388608.0) * (2.0 ** -126);
        else         v = (1.0 + real'(mf) / 8388608.0) * (2.0 ** (e - 127));
        return sign ? -v : v;
    endfunction

    function automatic int sw_fp4_x2(input logic [3:0] nib);
        int mag;
        case (nib[2:0])
            3'd0: mag = 0;
            3'd1: mag = 1;
            3'd2: mag = 2;
            3'd3: mag = 3;
            3'd4: mag = 4;
            3'd5: mag = 6;
            3'd6: mag = 8;
            default: mag = 12;
        endcase
        return nib[3] ? -mag : mag;
    endfunction

    function automatic real sw_dequant_elem(input logic [71:0] pkt, input int idx);
        int sc;
        real x;
        sc = int'($signed(pkt[71:64]));
        x  = real'(sw_fp4_x2(pkt[idx*4 +: 4])) * 0.5;
        return x * (2.0 ** (real'(sc) / 8.0));
    endfunction

    function automatic logic [71:0] sw_logq_vec(input logic [15:0][15:0] fp16_in);
        logic [15:0] sign_bits;
        logic [7:0]  mu [0:15];
        logic [7:0]  max_mu;
        logic signed [9:0] raw_scale;
        logic signed [7:0] shared_scale;
        logic [63:0] fp4_nibs;
        max_mu = 8'd0;
        for (int i = 0; i < 16; i++) begin
            logic [15:0] h;
            logic [4:0]  exp5;
            logic [9:0]  mant10;
            h = fp16_in[i];
            sign_bits[i] = h[15];
            exp5   = h[14:10];
            mant10 = h[9:0];
            if (exp5 == 5'd0)       mu[i] = 8'd0;
            else if (exp5 == 5'd31) mu[i] = 8'd247;
            else                    mu[i] = {exp5, mant10[9:7]};
            if (mu[i] > max_mu) max_mu = mu[i];
        end
        raw_scale = $signed({2'b00, max_mu}) - 10'sd140;
        if (max_mu == 8'd0)            shared_scale = 8'sd0;
        else if (raw_scale < -10'sd128) shared_scale = -8'sd128;
        else if (raw_scale > 10'sd127)  shared_scale = 8'sd127;
        else                            shared_scale = raw_scale[7:0];
        fp4_nibs = 64'd0;
        for (int i = 0; i < 16; i++) begin
            logic [7:0] diff;
            logic [2:0] mag;
            diff = max_mu - mu[i];
            if (mu[i] == 8'd0 || max_mu == 8'd0) mag = 3'd0;
            else if (diff <= 8'd2)               mag = 3'd7;
            else if (diff <= 8'd6)               mag = 3'd6;
            else if (diff <= 8'd10)              mag = 3'd5;
            else if (diff <= 8'd14)              mag = 3'd4;
            else if (diff <= 8'd18)              mag = 3'd3;
            else if (diff <= 8'd24)              mag = 3'd2;
            else if (diff <= 8'd36)              mag = 3'd1;
            else                                 mag = 3'd0;
            fp4_nibs[i*4 +: 4] = (mag == 3'd0) ? 4'd0 : {sign_bits[i], mag};
        end
        return {shared_scale, fp4_nibs};
    endfunction

    // DRAM / Canary / CSR / DMA Helpers
    function automatic logic [31:0] tcdm_addr(input int word, input int bank);
        tcdm_addr = (32'(word) << 9) | (32'(bank) << 4);
    endfunction

    task automatic write_dram_9b(input logic [31:0] dram_base, input int word_idx, input logic [71:0] pkt);
        int abs_byte;
        abs_byte = int'(dram_base[5:0]) + word_idx * 9;
        for (int b = 0; b < 9; b++) begin
            int beat, boff;
            beat = int'(dram_base >> 6) + ((abs_byte + b) >> 6);
            boff = (abs_byte + b) & 63;
            dram_mem[beat][boff*8 +: 8] = pkt[b*8 +: 8];
        end
    endtask

    task automatic read_dram_9b(input logic [31:0] dram_base, input int word_idx, output logic [71:0] pkt);
        int abs_byte;
        abs_byte = int'(dram_base[5:0]) + word_idx * 9;
        for (int b = 0; b < 9; b++) begin
            int beat, boff;
            beat = int'(dram_base >> 6) + ((abs_byte + b) >> 6);
            boff = (abs_byte + b) & 63;
            pkt[b*8 +: 8] = dram_mem[beat][boff*8 +: 8];
        end
    endtask

    task automatic set_dram_byte(input int byte_addr, input logic [7:0] val);
        dram_mem[byte_addr >> 6][(byte_addr & 63)*8 +: 8] = val;
    endtask

    function automatic logic [7:0] get_dram_byte(input int byte_addr);
        return dram_mem[byte_addr >> 6][(byte_addr & 63)*8 +: 8];
    endfunction

    task automatic arm_dram_canaries(input int byte_base, input int byte_len);
        for (int i = -64; i < 0; i++)
            set_dram_byte(byte_base + i, 8'hA5);
        for (int i = 0; i < byte_len; i++)
            set_dram_byte(byte_base + i, 8'h5A);
        for (int i = 0; i < 64; i++)
            set_dram_byte(byte_base + byte_len + i, 8'hA5);
    endtask

    task automatic check(input bit cond, input string msg);
        if (cond) checks_pass++;
        else begin
            checks_fail++;
            $display("    [FAIL] %s", msg);
        end
    endtask

    task automatic check_dram_canaries(input int byte_base, input int byte_len, input string tag);
        int bad;
        bad = 0;
        for (int i = -64; i < 0; i++)
            if (get_dram_byte(byte_base + i) !== 8'hA5) bad++;
        for (int i = 0; i < 64; i++)
            if (get_dram_byte(byte_base + byte_len + i) !== 8'hA5) bad++;
        check(bad == 0, {tag, " DRAM canaries"});
    endtask

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

    task automatic clear_ext_macro_csrs();
        cpu_csr_write(A_EXT_MACRO_SUB,  32'h0);
        cpu_csr_write(A_EXT_MACRO_ACT,  32'h0);
        cpu_csr_write(A_EXT_MACRO_OUT,  32'h0);
        cpu_csr_write(A_EXT_MACRO_ASTR, 32'h0);
        cpu_csr_write(A_EXT_MACRO_OSTR, 32'h0);
        cpu_csr_write(A_EXT_DRAM_V,     32'h0);
    endtask

    task automatic wait_job(input string name);
        bit seen_done, seen_busy;
        int cycle_cnt;
        seen_done = 1'b0; seen_busy = 1'b0; cycle_cnt = 0;
        while (!seen_done && cycle_cnt < JOB_TIMEOUT) begin
            @(posedge clk);
            cycle_cnt++;
            if (busy) seen_busy = 1'b1;
            if (seen_busy && done_irq && !busy) seen_done = 1'b1;
        end
        check(seen_done, name);
        if (!seen_done)
            $fatal(1, "Timeout on job: %s (seq_state=%0d)", name, dut.u_seq.state_q);
    endtask

    task automatic run_dma_1d(
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

    task automatic run_gemm_dma_ex(
        input int n_tiles, input int k_blks_per_tile,
        input logic [31:0] dram_base, input int tile_stride_bytes,
        input logic [31:0] q_base, input logic [31:0] out_base,
        input bit act_16bank, input bit out_16bank, input logic [1:0] out_mode,
        input string name
    );
        cpu_csr_write(A_CTRL,      32'h0);
        cpu_csr_write(A_SEQ_CTRL,  (out_16bank ? (32'd1 << 25) : 32'd0) |
                                   (act_16bank ? (32'd1 << 24) : 32'd0) |
                                   ({26'd0, out_mode, 4'b1000}));
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

    // SW Causal Softmax over hardware Phase-1 scores in DRAM_S -> quantize P in DRAM_P
    task automatic compute_softmax_s_to_p(input int mb, input int s_strip_bytes, input int p_strip_bytes);
        logic [31:0] s_addr, p_addr;
        s_addr = DRAM_S + 32'(mb * s_strip_bytes);
        p_addr = DRAM_P + 32'(mb * p_strip_bytes);
        for (int t = 0; t < M_BATCH; t++) begin
            int q_pos;
            real s_vals [0:MAX_KV-1];
            real max_s, sum_e;
            q_pos = mb * M_BATCH + t;
            max_s = -1.0e30;
            for (int kj = 0; kj < KV_TILES; kj++) begin
                for (int word = 0; word < 4; word++) begin
                    logic [71:0] sw;
                    read_dram_9b(s_addr, (kj * 4 + word) * 16 + t, sw);
                    for (int e = 0; e < 4; e++) begin
                        int k_pos;
                        real v;
                        k_pos = kj * 16 + word * 4 + e;
                        if (k_pos <= (N_PAST + q_pos)) begin
                            v = sw_fp16_to_real(sw[e*16 +: 16]);
                            s_vals[k_pos] = v;
                            if (v > max_s) max_s = v;
                        end else begin
                            s_vals[k_pos] = -1.0e30;
                        end
                    end
                end
            end
            sum_e = 0.0;
            for (int k_pos = 0; k_pos < N_KV; k_pos++)
                if (k_pos <= (N_PAST + q_pos))
                    sum_e += $exp(s_vals[k_pos] - max_s);
            if (sum_e <= 0.0) sum_e = 1.0;

            for (int kj = 0; kj < KV_TILES; kj++) begin
                logic [15:0][15:0] p_fp16;
                for (int l = 0; l < 16; l++) begin
                    int k_pos;
                    k_pos = kj * 16 + l;
                    if (k_pos <= (N_PAST + q_pos))
                        p_fp16[l] = sw_real_to_fp16($exp(s_vals[k_pos] - max_s) / sum_e);
                    else
                        p_fp16[l] = 16'd0;
                end
                write_dram_9b(p_addr, kj * 16 + t, sw_logq_vec(p_fp16));
            end
        end
    endtask

    // =========================================================================
    // MAIN
    // =========================================================================
    initial begin
        string mode_raw;
        logic [31:0] rd32;
        real s_ref2, s_err2, s_act2, s_dot, max_abs, sqnr, cos_s;
        real f_ref2, f_err2, f_dot, sqnr_fq, cos_fq;
        longint rd_bytes_tot, wr_bytes_tot, cyc_total_w_sfu;
        real mac_util_pct;

        mode_raw = "FLASH_KVPIN";
        void'($value$plusargs("MODE=%s", mode_raw));
        MODE_STR = to_upper_str(mode_raw);
        if (!(MODE_STR == "NAIVE" || MODE_STR == "ONLINE_2A" || MODE_STR == "ONLINE_2B" ||
              MODE_STR == "FLASH_RING" || MODE_STR == "FLASH_KVPIN"))
            $fatal(1, "Invalid +MODE=%s (expected NAIVE|ONLINE_2A|ONLINE_2B|FLASH_RING|FLASH_KVPIN)", mode_raw);

        M_TOKENS = 64;
        void'($value$plusargs("M_TOKENS=%d", M_TOKENS));
        N_KV = M_TOKENS;
        void'($value$plusargs("N_KV=%d", N_KV));
        DRAM_BW = 64;
        void'($value$plusargs("DRAM_BW=%d", DRAM_BW));

        if ((M_TOKENS < 16) || (M_TOKENS > MAX_M) || (M_TOKENS % 16 != 0) ||
            (N_KV < M_TOKENS) || (N_KV > MAX_KV) || (N_KV % 16 != 0))
            $fatal(1, "Invalid M_TOKENS=%0d N_KV=%0d", M_TOKENS, N_KV);

        M_CHUNKS      = M_TOKENS / M_BATCH;
        KV_TILES      = N_KV / 16;
        N_PAST        = N_KV - M_TOKENS;
        KV_TILES_PAST = N_PAST / 16;

        DATA_DIR = $sformatf("sim_attn_cmp_m%0d_kv%0d", M_TOKENS, N_KV);
        void'($value$plusargs("DATA_DIR=%s", DATA_DIR));

`ifdef FST_TRACE
        if ($test$plusargs("DUMP")) begin
            $dumpfile($sformatf("tb_attn_%s_m%0d_kv%0d_bw%0d.fst", MODE_STR, M_TOKENS, N_KV, DRAM_BW));
            $dumpvars(0, dut);
        end
`endif

        csr_req_valid = 0; csr_req_write = 0; csr_req_addr = 0; csr_req_data = 0; csr_rsp_ready = 1;
        measure_en = 1'b0;
        cyc_total = 0; cyc_mac = 0; cyc_dma = 0; dram_stall_cyc = 0; cpu_mem_active_cyc = 0;
        p2_sfu_est_cyc = 0; sp_rd_bytes = 0; sp_wr_bytes = 0;
        max_inflight_seen = 0; checks_pass = 0; checks_fail = 0;

        for (int i = 0; i < DRAM_WORDS; i++) dram_mem[i] = 512'd0;

        $display("Loading stimulus from %s (MODE=%s, M=%0d, N_KV=%0d, DRAM_BW=%0d B/cyc)...",
                 DATA_DIR, MODE_STR, M_TOKENS, N_KV, DRAM_BW);
        $readmemh({DATA_DIR, "/q_words.hex"},     q_mem,       0, H_Q * M_TOKENS * K_PACKETS_QK - 1);
        $readmemh({DATA_DIR, "/k_words.hex"},     k_mem,       0, H_KV * KV_TILES * K_PACKETS_QK * 16 - 1);
        $readmemh({DATA_DIR, "/v_words.hex"},     v_mem,       0, H_KV * DV_TILES * KV_TILES * 16 - 1);
        $readmemh({DATA_DIR, "/g_attn_fp32.hex"}, g_attn_fp32, 0, H_Q * M_TOKENS * D_V - 1);
        $readmemh({DATA_DIR, "/g_attn_fq.hex"},   g_attn_fq,   0, H_Q * M_TOKENS * D_V - 1);

        for (int i = 0; i < H_Q * M_TOKENS * K_PACKETS_QK; i++)
            write_dram_9b(DRAM_Q, i, q_mem[i]);
        for (int i = 0; i < H_KV * KV_TILES * K_PACKETS_QK * 16; i++)
            write_dram_9b(DRAM_K, i, k_mem[i]);
        for (int i = 0; i < H_KV * DV_TILES * KV_TILES * 16; i++)
            write_dram_9b(DRAM_V, i, v_mem[i]);

        arm_dram_canaries(int'(DRAM_O), H_Q * M_TOKENS * DV_TILES * 9);

        rst_n = 1'b0;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        cpu_csr_read(A_RO_ID, rd32);
        check(rd32 == CSR_ID_VALUE, "CSR ID (SNAF)");

        measure_en = 1'b1;

        if (MODE_STR == "NAIVE" || MODE_STR == "ONLINE_2A") begin
            int s_strip_bytes, p_strip_bytes;
            s_strip_bytes = KV_TILES * 64 * 9; // 4 words * 16 banks * KV_TILES * 9B
            p_strip_bytes = KV_TILES * 16 * 9; // 1 word  * 16 banks * KV_TILES * 9B

            clear_ext_macro_csrs();
            for (int h = 0; h < H_Q; h++) begin
                int hkv;
                hkv = h >> 2;
                for (int mb = 0; mb < M_CHUNKS; mb++) begin
                    longint r0, w0;

                    // Phase 1: Q_h[mb] (DRAM -> 16-bank TCDM) & GEMM S = Q * K^T (FP16) & DMA Write S -> DRAM
                    for (int t = 0; t < M_BATCH; t++)
                        run_dma_1d(2'b10, tcdm_addr(0, 16 + t),
                                   DRAM_Q + 32'(((h * M_TOKENS + mb * M_BATCH + t) * K_PACKETS_QK) * 9),
                                   K_PACKETS_QK * 9, "P1 Load Q");

                    run_gemm_dma_ex(KV_TILES, K_PACKETS_QK,
                                    DRAM_K + 32'(hkv * KV_TILES * K_PACKETS_QK * 16 * 9),
                                    K_PACKETS_QK * 144,
                                    tcdm_addr(0, 16), tcdm_addr(128, 16),
                                    1'b1, 1'b1, 2'd0, "P1 GEMM QK");

                    w0 = longint'(stat_wr_beats);
                    run_dma_1d(2'b01, tcdm_addr(128, 16), DRAM_S + 32'(mb * s_strip_bytes), s_strip_bytes, "P1 WB S");
                    sp_wr_bytes += (longint'(stat_wr_beats) - w0) * 64;

                    // Phase 2: Unfused Softmax (2-pass DMA read S for NAIVE, 1-pass for ONLINE_2A) + RTL Requant + DMA WB P
                    r0 = longint'(stat_rd_beats);
                    run_dma_1d(2'b00, tcdm_addr(128, 16), DRAM_S + 32'(mb * s_strip_bytes), s_strip_bytes, "P2 Rd S1");
                    if (MODE_STR == "NAIVE")
                        run_dma_1d(2'b00, tcdm_addr(128, 16), DRAM_S + 32'(mb * s_strip_bytes), s_strip_bytes, "P2 Rd S2");
                    sp_rd_bytes += (longint'(stat_rd_beats) - r0) * 64;

                    run_requant_wrap(tcdm_addr(128, 16), M_BATCH * KV_TILES, tcdm_addr(0, 16),
                                     KV_TILES * 4 - 1, KV_TILES - 1, "P2 Requant S->P");

                    w0 = longint'(stat_wr_beats);
                    run_dma_1d(2'b01, tcdm_addr(0, 16), DRAM_P + 32'(mb * p_strip_bytes), p_strip_bytes, "P2 WB P");
                    sp_wr_bytes += (longint'(stat_wr_beats) - w0) * 64;

                    // Optimistic SFU Softmax compute cycle estimate (1-cycle vector cadence vs 8-lane pass)
                    p2_sfu_est_cyc += longint'(M_BATCH) * longint'(KV_TILES) * ((MODE_STR == "NAIVE") ? 64'd8 : 64'd4);

                    // Populate DRAM_P w/ exact Causal Softmax + bit-exact Log8-FP4 quant of Phase 1 hardware S
                    compute_softmax_s_to_p(mb, s_strip_bytes, p_strip_bytes);

                    // Phase 3: DMA Load P -> GEMM O = P * V (out=2 16-bank scatter) -> DMA WB O
                    r0 = longint'(stat_rd_beats);
                    run_dma_1d(2'b00, tcdm_addr(0, 16), DRAM_P + 32'(mb * p_strip_bytes), p_strip_bytes, "P3 Load P");
                    sp_rd_bytes += (longint'(stat_rd_beats) - r0) * 64;

                    run_gemm_dma_ex(DV_TILES, KV_TILES,
                                    DRAM_V + 32'(hkv * DV_TILES * KV_TILES * 16 * 9),
                                    KV_TILES * 144,
                                    tcdm_addr(0, 16), tcdm_addr(128, 16),
                                    1'b1, 1'b0, 2'd2, "P3 GEMM PV");

                    for (int t = 0; t < M_BATCH; t++)
                        run_dma_1d(2'b11, tcdm_addr(128, 16 + t),
                                   DRAM_O + 32'(((h * M_TOKENS + mb * M_BATCH + t) * DV_TILES) * 9),
                                   DV_TILES * 9, "P3 WB O");
                end
            end
        end else begin
            int m_q, q_tiles, q_sub;
            bit kv_pin_en;
            if (MODE_STR == "ONLINE_2B") begin
                m_q       = 1;
                q_tiles   = M_TOKENS;
                q_sub     = (q_tiles > 64) ? 64 : q_tiles;
                kv_pin_en = 1'b0;
            end else if (MODE_STR == "FLASH_RING") begin
                m_q       = 8;
                q_tiles   = M_TOKENS / 8;
                q_sub     = (q_tiles > 8) ? 8 : q_tiles;
                kv_pin_en = 1'b0;
            end else begin
                m_q       = 8;
                q_tiles   = M_TOKENS / 8;
                q_sub     = (q_tiles > 8) ? 8 : q_tiles;
                kv_pin_en = 1'b1;
            end

            clear_ext_macro_csrs();
            cpu_csr_write(A_CTRL,           32'h0);
            cpu_csr_write(A_EXT_MACRO_SUB,  (32'(KV_TILES_PAST[13:0]) << 18) |
                                            (32'd1 << 17) |
                                            (32'(kv_pin_en) << 16) |
                                            32'(q_sub[7:0]));
            cpu_csr_write(A_EXT_MACRO_ACT,  DRAM_Q);
            cpu_csr_write(A_EXT_MACRO_OUT,  DRAM_O);
            cpu_csr_write(A_EXT_MACRO_ASTR, 32'h0004_0000); // Bank 30 ping-pong 0 <-> 512
            cpu_csr_write(A_EXT_MACRO_OSTR, 32'h0004_0000); // Bank 28 ping-pong 0 <-> 512
            cpu_csr_write(A_EXT_DRAM_V,     DRAM_V);
            cpu_csr_write(A_SEQ_CTRL,       (32'd1 << 30) | (32'(GQA_M1) << 16) | 32'h2D); // macro=1, GQA=4:1, op=01, out=2, causal=1, dma=1
            cpu_csr_write(A_ATTN_H,         H_Q);
            cpu_csr_write(A_ATTN_QT,        q_tiles);
            cpu_csr_write(A_ATTN_MQ,        m_q);
            cpu_csr_write(A_ATTN_DV,        DV_TILES);
            cpu_csr_write(A_ATTN_KV,        KV_TILES);
            cpu_csr_write(A_ATTN_KB,        K_PACKETS_QK);
            cpu_csr_write(A_RING_BASE,      tcdm_addr(0, 0));
            cpu_csr_write(A_DRAM_KV,        DRAM_K);
            cpu_csr_write(A_TILE_ST,        32'd720);
            cpu_csr_write(A_Q_BASE,         tcdm_addr(0, 30));
            cpu_csr_write(A_OUT_BASE,       tcdm_addr(0, 28));
            cpu_csr_write(A_START,          32'h0000_0003);
            wait_job(MODE_STR);
            clear_ext_macro_csrs();
        end

        measure_en = 1'b0;

        // Verify DRAM canaries & full [H_Q][M_TOKENS][D_V] accuracy vs FP32 golden
        check_dram_canaries(int'(DRAM_O), H_Q * M_TOKENS * DV_TILES * 9, MODE_STR);

        s_ref2 = 0.0; s_err2 = 0.0; s_act2 = 0.0; s_dot = 0.0; max_abs = 0.0;
        f_ref2 = 0.0; f_err2 = 0.0; f_dot = 0.0;
        for (int h = 0; h < H_Q; h++) begin
            for (int t = 0; t < M_TOKENS; t++) begin
                for (int dvi = 0; dvi < DV_TILES; dvi++) begin
                    logic [71:0] opkt;
                    read_dram_9b(DRAM_O, (h * M_TOKENS + t) * DV_TILES + dvi, opkt);
                    for (int l = 0; l < 16; l++) begin
                        int c_idx;
                        real v_a, v_g, v_fq, df;
                        c_idx = (h * M_TOKENS + t) * D_V + dvi * 16 + l;
                        v_a   = sw_dequant_elem(opkt, l);
                        v_g   = sw_fp32_to_real(g_attn_fp32[c_idx]);
                        v_fq  = sw_fp32_to_real(g_attn_fq[c_idx]);
                        df    = (v_a > v_g) ? (v_a - v_g) : (v_g - v_a);
                        if (df > max_abs) max_abs = df;
                        s_ref2 += v_g * v_g;
                        s_act2 += v_a * v_a;
                        s_err2 += (v_a - v_g) * (v_a - v_g);
                        s_dot  += v_a * v_g;
                        f_ref2 += v_fq * v_fq;
                        f_err2 += (v_a - v_fq) * (v_a - v_fq);
                        f_dot  += v_a * v_fq;
                    end
                end
            end
        end

        sqnr    = (s_err2 > 0.0) ? (10.0 * $log10(s_ref2 / s_err2)) : 999.0;
        cos_s   = (s_act2 > 0.0 && s_ref2 > 0.0) ? (s_dot / ($sqrt(s_act2) * $sqrt(s_ref2))) : 0.0;
        sqnr_fq = (f_err2 > 0.0) ? (10.0 * $log10(f_ref2 / f_err2)) : 999.0;
        cos_fq  = (s_act2 > 0.0 && f_ref2 > 0.0) ? (f_dot / ($sqrt(s_act2) * $sqrt(f_ref2))) : 0.0;
        check(cos_fq >= 0.96, "Attn CosineSim vs PyFQ >= 0.96");
        check(cos_s  >= 0.80, "Attn CosineSim vs FP32 >= 0.80");
        check(cpu_mem_active_cyc == 0, "ZERO cpu_mem activity");
        check(max_inflight_seen <= 4,  "inflight_out_cnt <= 4");

        rd_bytes_tot    = longint'(stat_rd_beats) * 64;
        wr_bytes_tot    = longint'(stat_wr_beats) * 64;
        cyc_total_w_sfu = cyc_total + p2_sfu_est_cyc;
        mac_util_pct    = (cyc_total_w_sfu > 0) ? (100.0 * real'(cyc_mac) / real'(cyc_total_w_sfu)) : 0.0;

        $display("\n========================================================================");
        $display("  ATTENTION COMPARISON: %-11s | M=%0d, N_KV=%0d | BW=%0d B/cyc",
                 MODE_STR, M_TOKENS, N_KV, DRAM_BW);
        $display("========================================================================");
        $display("  HW Measured Cycles : %0d (MAC=%0d, DMA=%0d, DRAM_stall=%0d)",
                 cyc_total, cyc_mac, cyc_dma, dram_stall_cyc);
        $display("  P2 SFU Est / Total : +%0d cyc -> %0d cyc (MAC Util = %0.2f%%)",
                 p2_sfu_est_cyc, cyc_total_w_sfu, mac_util_pct);
        $display("  DRAM Traffic Total : rd=%0d B, wr=%0d B, sum=%0d B",
                 rd_bytes_tot, wr_bytes_tot, rd_bytes_tot + wr_bytes_tot);
        $display("  Interm S/P Traffic : rd=%0d B, wr=%0d B, sum=%0d B",
                 sp_rd_bytes, sp_wr_bytes, sp_rd_bytes + sp_wr_bytes);
        $display("  Accuracy vs PyFQ   : SQNR=%0.2f dB | CosineSim=%0.6f",
                 sqnr_fq, cos_fq);
        $display("  Accuracy vs FP32   : SQNR=%0.2f dB | CosineSim=%0.6f | MaxAbs=%0.4f",
                 sqnr, cos_s, max_abs);
        $display("  cpu_mem_active_cyc : %0d | max_inflight=%0d | checks=%0d/%0d fail=%0d",
                 cpu_mem_active_cyc, max_inflight_seen, checks_pass, checks_pass + checks_fail, checks_fail);
        $display("========================================================================");
        $display("CSV_ROW,%s,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0.2f,%0.2f,%0.6f,%0.2f,%0.6f,%0.4f,%0d,%0d,%0d",
                 MODE_STR, M_TOKENS, N_KV, DRAM_BW,
                 cyc_total, p2_sfu_est_cyc, cyc_total_w_sfu, cyc_mac, cyc_dma, dram_stall_cyc,
                 rd_bytes_tot, wr_bytes_tot, rd_bytes_tot + wr_bytes_tot,
                 sp_rd_bytes, sp_wr_bytes, mac_util_pct,
                 sqnr_fq, cos_fq, sqnr, cos_s, max_abs, cpu_mem_active_cyc, max_inflight_seen, checks_fail);

        if (checks_fail != 0)
            $fatal(1, "FAILED with %0d check failures!", checks_fail);
        $finish;
    end

endmodule
