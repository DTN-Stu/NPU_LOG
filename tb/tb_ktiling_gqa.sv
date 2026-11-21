// =============================================================================
// tb_ktiling_gqa.sv
// Verifies the 3 sequencer/subsystem upgrades:
//   [1] Output-Stationary K-Tiling (non-DMA): 2x64 vs 1x128 bit-exact
//   [2] Output-Stationary K-Tiling + 4-Port DMA + 128KB Ring Slots: 2x256 vs 1x512
//       and full 128KB slot capacity (256 K-packets => K=4096 per slot)
//   [3] GQA 4:1 (H_Q=8, H_KV=2, G=4) FlashAttention bit-exact vs MHA 1:1
// =============================================================================
`timescale 1ns/1ps

module tb_ktiling_gqa;

    localparam int unsigned DRAM_LATENCY = 4;

    localparam logic [11:0] A_CTRL      = 12'd0;
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
    localparam logic [11:0] A_START     = 12'd53;

    logic clk = 1'b0;
    always #5 clk = ~clk;
    logic rst_n;

    logic        cpu_req_valid, cpu_req_ready, cpu_req_write, cpu_rsp_valid;
    logic [31:0] cpu_req_addr;
    logic [71:0] cpu_req_wdata, cpu_rsp_rdata;
    logic [8:0]  cpu_req_strb;

    logic        csr_req_valid, csr_req_ready, csr_req_write, csr_rsp_valid;
    logic [11:0] csr_req_addr;
    logic [31:0] csr_req_wdata, csr_rsp_rdata;
    logic        irq_o, busy_o;

    logic [7:0]        dma_tcdm_req_valid, dma_tcdm_req_ready, dma_tcdm_req_write, dma_tcdm_rsp_valid;
    logic [7:0][31:0]  dma_tcdm_req_addr;
    logic [7:0][71:0]  dma_tcdm_req_wdata, dma_tcdm_rsp_rdata;
    logic [7:0][8:0]   dma_tcdm_req_strb;

    logic        dut_desc_valid, dut_desc_ready, dut_desc_done;
    logic [1:0]  dut_desc_dir;
    logic [31:0] dut_desc_dram_addr, dut_desc_tcdm_addr, dut_desc_size;

    logic         dram_req_valid, dram_req_ready, dram_req_write, dram_rsp_valid;
    logic [31:0]  dram_req_addr;
    logic [511:0] dram_req_wdata, dram_rsp_rdata;
    logic [63:0]  dram_req_strb;

    snax_llm_top_clean dut (
        .clk                     (clk),
        .rst_n                   (rst_n),
        .cpu_csr_req_valid_i     (csr_req_valid),
        .cpu_csr_req_ready_o     (csr_req_ready),
        .cpu_csr_req_write_i     (csr_req_write),
        .cpu_csr_req_addr_i      (csr_req_addr),
        .cpu_csr_req_data_i      (csr_req_wdata),
        .cpu_csr_rsp_valid_o     (csr_rsp_valid),
        .cpu_csr_rsp_data_o      (csr_rsp_rdata),
        .cpu_csr_rsp_ready_i     (1'b1),
        .cpu_mem_req_valid_i     (cpu_req_valid),
        .cpu_mem_req_ready_o     (cpu_req_ready),
        .cpu_mem_req_addr_i      (cpu_req_addr),
        .cpu_mem_req_write_i     (cpu_req_write),
        .cpu_mem_req_wdata_i     (cpu_req_wdata),
        .cpu_mem_req_strb_i      (cpu_req_strb),
        .cpu_mem_rsp_valid_o     (cpu_rsp_valid),
        .cpu_mem_rsp_rdata_o     (cpu_rsp_rdata),
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
        .dma_desc_valid_o        (dut_desc_valid),
        .dma_desc_ready_i        (dut_desc_ready),
        .dma_desc_dir_o          (dut_desc_dir),
        .dma_desc_dram_addr_o    (dut_desc_dram_addr),
        .dma_desc_tcdm_addr_o    (dut_desc_tcdm_addr),
        .dma_desc_size_o         (dut_desc_size),
        .dma_desc_done_i         (dut_desc_done),
        .subsystem_busy_o        (busy_o),
        .subsystem_done_irq_o    (irq_o)
    );

    snax_simple_dma #(.TCDM_PORTS(8)) u_dma (
        .clk                  (clk),
        .rst_n                (rst_n),
        .desc_valid_i         (dut_desc_valid),
        .desc_ready_o         (dut_desc_ready),
        .desc_dir_i           (dut_desc_dir),
        .desc_dram_addr_i     (dut_desc_dram_addr),
        .desc_tcdm_addr_i     (dut_desc_tcdm_addr),
        .desc_size_bytes_i    (dut_desc_size),
        .desc_done_o          (dut_desc_done),
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
        .stat_dram_rd_beats_o (),
        .stat_dram_wr_beats_o (),
        .stat_tcdm_words_o    (),
        .stat_busy_cycles_o   ()
    );

    // 512-bit DRAM model (8192 x 64B = 512 KB)
    localparam int DRAM_WORDS = 8192;
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
            if (dram_req_valid && dram_req_ready && !dram_req_write) begin
                lat_vld[0]  <= 1'b1;
                lat_data[0] <= dram_mem[dram_req_addr[18:6]];
            end
        end
    end

    function automatic logic [31:0] tcdm_addr(input int word, input int bank);
        tcdm_addr = (32'(word) << 9) | (32'(bank) << 4);
    endfunction

    task automatic cpu_write(input logic [31:0] a, input logic [71:0] d);
        @(negedge clk);
        cpu_req_valid = 1'b1; cpu_req_write = 1'b1;
        cpu_req_addr  = a;    cpu_req_wdata = d; cpu_req_strb = 9'h1FF;
        while (!cpu_req_ready) @(posedge clk);
        @(posedge clk);
        cpu_req_valid = 1'b0; cpu_req_write = 1'b0; cpu_req_strb = 9'h000;
    endtask

    task automatic cpu_read(input logic [31:0] a, output logic [71:0] d);
        @(negedge clk);
        cpu_req_valid = 1'b1; cpu_req_write = 1'b0;
        cpu_req_addr  = a;    cpu_req_strb  = 9'h000;
        while (!cpu_req_ready) @(posedge clk);
        @(posedge clk);
        cpu_req_valid = 1'b0;
        while (!cpu_rsp_valid) @(posedge clk);
        d = cpu_rsp_rdata;
    endtask

    task automatic csr_write(input logic [11:0] a, input logic [31:0] d);
        @(negedge clk);
        csr_req_valid = 1'b1; csr_req_write = 1'b1;
        csr_req_addr  = a;    csr_req_wdata = d;
        while (!csr_req_ready) @(posedge clk);
        @(posedge clk);
        csr_req_valid = 1'b0; csr_req_write = 1'b0;
    endtask

    task automatic wait_seq();
        int g = 0;
        repeat (4) @(posedge clk);
        while (dut.seq_busy || dut.wo_busy) begin
            @(posedge clk); g++;
            if (g > 100000) $fatal(1, "Sequencer timeout");
        end
    endtask

    function automatic logic [71:0] make_pkt(input int seed);
        logic [63:0] nibs;
        logic [7:0]  sc;
        for (int i = 0; i < 16; i++) begin
            int v = (seed * 7 + i * 3) % 7;
            case (v)
                0: nibs[i*4 +: 4] = 4'h1;
                1: nibs[i*4 +: 4] = 4'h2;
                2: nibs[i*4 +: 4] = 4'h3;
                3: nibs[i*4 +: 4] = 4'h9;
                4: nibs[i*4 +: 4] = 4'hA;
                5: nibs[i*4 +: 4] = 4'h0;
                default: nibs[i*4 +: 4] = 4'h2;
            endcase
        end
        sc = (seed % 5) - 2;
        return {sc, nibs};
    endfunction

    task automatic pack_wt_to_dram(input int dram_byte_off, input int n_words, input logic [71:0] pkts[]);
        int total_bits = n_words * 72;
        int n_beats = (total_bits + 511) / 512;
        int base_beat = dram_byte_off >> 6;
        for (int b = 0; b < n_beats; b++) dram_mem[base_beat + b] = '0;
        for (int w = 0; w < n_words; w++) begin
            for (int bit_i = 0; bit_i < 72; bit_i++) begin
                int gbit = w * 72 + bit_i;
                dram_mem[base_beat + (gbit >> 9)][gbit & 511] = pkts[w][bit_i];
            end
        end
    endtask

    logic [71:0] gold_o [0:2047];
    logic [71:0] act_o  [0:2047];

    initial begin
        cpu_req_valid = 0; cpu_req_write = 0; cpu_req_addr = 0; cpu_req_wdata = 0; cpu_req_strb = 0;
        csr_req_valid = 0; csr_req_write = 0; csr_req_addr = 0; csr_req_wdata = 0;
        rst_n = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        // =====================================================================
        // TEST 1: Output-Stationary K-Tiling (non-DMA): 2x64 vs 1x128 (M=16, N=16)
        // =====================================================================
        for (int t = 0; t < 16; t++)
            for (int kb = 0; kb < 8; kb++)
                cpu_write(tcdm_addr(t*8 + kb, 16), make_pkt(t*13 + kb + 1));
        for (int kb = 0; kb < 8; kb++)
            for (int ln = 0; ln < 16; ln++)
                cpu_write(tcdm_addr(kb, ln), make_pkt(kb*19 + ln + 100));

        csr_write(A_CTRL,      32'h0);
        csr_write(A_SFU_CTRL,  32'h0);
        csr_write(A_SEQ_CTRL,  32'h0); // k_tiles=1, K_PACKETS=8
        csr_write(A_GEMM_NT,   32'd1);
        csr_write(A_GEMM_MT,   32'd1);
        csr_write(A_GEMM_KB,   32'd8);
        csr_write(A_RING_BASE, tcdm_addr(0, 0));
        csr_write(A_Q_BASE,    tcdm_addr(0, 16));
        csr_write(A_OUT_BASE,  tcdm_addr(0, 17));
        csr_write(A_START,     32'h3);
        wait_seq();
        for (int w = 0; w < 64; w++) cpu_read(tcdm_addr(w, 17), gold_o[w]);

        csr_write(A_SEQ_CTRL,  (32'd1 << 8)); // k_tiles_m1=1 (2 K-tiles), K_PACKETS=4
        csr_write(A_GEMM_KB,   32'd4);
        csr_write(A_OUT_BASE,  tcdm_addr(0, 18));
        csr_write(A_START,     32'h3);
        wait_seq();
        for (int w = 0; w < 64; w++) begin
            cpu_read(tcdm_addr(w, 18), act_o[w]);
            if (act_o[w] !== gold_o[w])
                $fatal(1, "TEST 1 K-Tiling non-DMA mismatch at word %0d: %h != %h", w, act_o[w], gold_o[w]);
        end
        $display("[PASS] TEST 1: K-Tiling (non-DMA) 2x64 vs 1x128 bit-exact (64/64 words)");

        // =====================================================================
        // TEST 2: K-Tiling + 4-Port DMA + 128KB Ring Slots: 2x256 (K=512) vs 1x512
        // =====================================================================
        begin
            logic [71:0] wt_pkts[];
            wt_pkts = new[32 * 16];
            for (int t = 0; t < 16; t++)
                for (int kb = 0; kb < 32; kb++)
                    cpu_write(tcdm_addr(t*32 + kb, 16), make_pkt(t*7 + kb + 50));
            for (int kb = 0; kb < 32; kb++)
                for (int ln = 0; ln < 16; ln++) begin
                    logic [71:0] p;
                    p = make_pkt(kb*11 + ln + 200);
                    cpu_write(tcdm_addr(kb, ln), p);
                    wt_pkts[kb*16 + ln] = p;
                end
            pack_wt_to_dram(0, 32 * 16, wt_pkts);

            // Golden 1x512 non-DMA
            csr_write(A_SEQ_CTRL,  32'h0);
            csr_write(A_GEMM_NT,   32'd1);
            csr_write(A_GEMM_MT,   32'd1);
            csr_write(A_GEMM_KB,   32'd32);
            csr_write(A_RING_BASE, tcdm_addr(0, 0));
            csr_write(A_Q_BASE,    tcdm_addr(0, 16));
            csr_write(A_OUT_BASE,  tcdm_addr(0, 17));
            csr_write(A_START,     32'h3);
            wait_seq();
            for (int w = 0; w < 64; w++) cpu_read(tcdm_addr(w, 17), gold_o[w]);

            // Clear TCDM banks 0..15 words 0..511 to prove DMA loads both K-tiles
            for (int w = 0; w < 512; w++)
                for (int ln = 0; ln < 16; ln++)
                    cpu_write(tcdm_addr(w, ln), 72'h0);

            // 2 K-tiles x 256 via DMA + 128KB ring slots
            csr_write(A_SEQ_CTRL,  (32'd1 << 8) | 32'h8); // k_tiles_m1=1, enable_dma=1
            csr_write(A_GEMM_KB,   32'd16);
            csr_write(A_DRAM_KV,   32'h0);
            csr_write(A_TILE_ST,   32'd32 * 32'd144);
            csr_write(A_OUT_BASE,  tcdm_addr(0, 18));
            csr_write(A_START,     32'h3);
            wait_seq();
            for (int w = 0; w < 64; w++) begin
                cpu_read(tcdm_addr(w, 18), act_o[w]);
                if (act_o[w] !== gold_o[w])
                    $fatal(1, "TEST 2 K-Tiling+DMA mismatch at word %0d: %h != %h", w, act_o[w], gold_o[w]);
            end
            $display("[PASS] TEST 2: K-Tiling + 4-Port DMA + 128KB Ring (2x256 vs 1x512) bit-exact (64/64 words)");
        end

        // =====================================================================
        // TEST 3: GQA 4:1 (H_Q=8, H_KV=2, G=4, d_k=d_v=32, M_TOKENS=16)
        // =====================================================================
        // Q: 8 heads * Q_TILES(2) * M_Q(8) * K_PKT(2) = 256 packets in bank 16
        for (int h = 0; h < 8; h++)
            for (int q = 0; q < 16; q++)
                for (int kb = 0; kb < 2; kb++)
                    cpu_write(tcdm_addr((h*16 + q)*2 + kb, 16), make_pkt(h*31 + q*5 + kb + 300));

        // First run MHA 1:1 (H=8) with 2 KV heads duplicated 4x into K[0..7], V[0..7]
        // K[h][kj=0][kb=0..1] in banks 0..15, V_BASE_BLOCK = 8*1*2 = 16
        for (int h = 0; h < 8; h++) begin
            int hkv;
            hkv = h / 4;
            for (int kb = 0; kb < 2; kb++)
                for (int ln = 0; ln < 16; ln++)
                    cpu_write(tcdm_addr(h*2 + kb, ln), make_pkt(hkv*41 + kb*7 + ln + 500));
            for (int dv = 0; dv < 2; dv++)
                for (int ln = 0; ln < 16; ln++)
                    cpu_write(tcdm_addr(16 + h*2 + dv, ln), make_pkt(hkv*53 + dv*13 + ln + 700));
        end
        csr_write(A_SEQ_CTRL,  32'h05); // op=01 Attn, causal=1, out_mode=0 (FP16), gqa_m1=0
        csr_write(A_ATTN_H,    32'd8);
        csr_write(A_ATTN_QT,   32'd2);
        csr_write(A_ATTN_MQ,   32'd8);
        csr_write(A_ATTN_DV,   32'd2);
        csr_write(A_ATTN_KV,   32'd1);
        csr_write(A_ATTN_KB,   32'd2);
        csr_write(A_RING_BASE, tcdm_addr(0, 0));
        csr_write(A_Q_BASE,    tcdm_addr(0, 16));
        csr_write(A_OUT_BASE,  tcdm_addr(0, 17));
        csr_write(A_START,     32'h3);
        wait_seq();
        // 8 heads * 2 Q_TILES * 2 DV_TILES * 8 M_Q * 4 words = 1024 words
        for (int w = 0; w < 1024; w++) cpu_read(tcdm_addr(w, 17), gold_o[w]);

        // Load KV head 1 into slot h=1 and clear duplicated slots [2..7]
        for (int kb = 0; kb < 2; kb++)
            for (int ln = 0; ln < 16; ln++)
                cpu_write(tcdm_addr(1*2 + kb, ln), make_pkt(1*41 + kb*7 + ln + 500));
        for (int dv = 0; dv < 2; dv++)
            for (int ln = 0; ln < 16; ln++)
                cpu_write(tcdm_addr(16 + 1*2 + dv, ln), make_pkt(1*53 + dv*13 + ln + 700));
        for (int h = 2; h < 8; h++) begin
            for (int kb = 0; kb < 2; kb++)
                for (int ln = 0; ln < 16; ln++)
                    cpu_write(tcdm_addr(h*2 + kb, ln), 72'h0);
            for (int dv = 0; dv < 2; dv++)
                for (int ln = 0; ln < 16; ln++)
                    cpu_write(tcdm_addr(16 + h*2 + dv, ln), 72'h0);
        end

        csr_write(A_SEQ_CTRL,  (32'd3 << 16) | 32'h05); // gqa_group_m1 = 3 (G=4)
        csr_write(A_OUT_BASE,  tcdm_addr(0, 18));
        csr_write(A_START,     32'h3);
        wait_seq();
        for (int w = 0; w < 1024; w++) begin
            cpu_read(tcdm_addr(w, 18), act_o[w]);
            if (act_o[w] !== gold_o[w])
                $fatal(1, "TEST 3 GQA 4:1 mismatch at word %0d: %h != %h", w, act_o[w], gold_o[w]);
        end
        $display("[PASS] TEST 3: GQA 4:1 (H_Q=8, H_KV=2) bit-exact vs MHA (1024/1024 words = 4096 FP16)");

        // =====================================================================
        // TEST 4: Multi-Bank Activation Layout (1 token/bank across banks 16..31)
        //         M=16, N=16, K=8192 (512 K-blocks) on true 288 KB TCDM (BANK_DEPTH=1024)
        // =====================================================================
        begin
            logic [71:0] wt_8k [0:8191];
            for (int t = 0; t < 16; t++)
                for (int kb = 0; kb < 512; kb++)
                    cpu_write(tcdm_addr(kb, 16 + t), make_pkt(t*19 + kb*3 + 77));
            for (int kb = 0; kb < 512; kb++)
                for (int ln = 0; ln < 16; ln++)
                    wt_8k[kb*16 + ln] = make_pkt(kb*11 + ln + 333);
            pack_wt_to_dram(32'h0, 8192, wt_8k);

            // Reference: 4 K-tiles x 128 K-blocks (act_bank_layout=1)
            csr_write(A_SEQ_CTRL,  (32'd1 << 24) | (32'd3 << 8) | 32'h8);
            csr_write(A_GEMM_NT,   32'd1);
            csr_write(A_GEMM_MT,   32'd1);
            csr_write(A_GEMM_KB,   32'd128);
            csr_write(A_RING_BASE, 32'h0);
            csr_write(A_DRAM_KV,   32'h0);
            csr_write(A_TILE_ST,   32'd512 * 32'd144);
            csr_write(A_Q_BASE,    tcdm_addr(0, 16));
            csr_write(A_OUT_BASE,  tcdm_addr(512, 16));
            csr_write(A_START,     32'h3);
            wait_seq();
            for (int w = 0; w < 64; w++) cpu_read(tcdm_addr(512 + w, 16), gold_o[w]);

            // DUT: 2 K-tiles x 256 K-blocks (act_bank_layout=1)
            csr_write(A_SEQ_CTRL,  (32'd1 << 24) | (32'd1 << 8) | 32'h8);
            csr_write(A_GEMM_KB,   32'd256);
            csr_write(A_OUT_BASE,  tcdm_addr(576, 16));
            csr_write(A_START,     32'h3);
            wait_seq();
            for (int w = 0; w < 64; w++) begin
                cpu_read(tcdm_addr(576 + w, 16), act_o[w]);
                if (act_o[w] !== gold_o[w])
                    $fatal(1, "TEST 4 Multi-Bank Act K=8192 mismatch at word %0d: %h != %h", w, act_o[w], gold_o[w]);
            end
            $display("[PASS] TEST 4: Multi-Bank Act (banks 16..31) K=8192 (2x256 vs 4x128) on 288KB TCDM bit-exact (64/64 words)");
        end

        $display("\n=== ALL K-TILING + 128KB RING + GQA + MULTI-BANK ACT TESTS PASSED (0 errors) ===");
        $finish;
    end
endmodule
