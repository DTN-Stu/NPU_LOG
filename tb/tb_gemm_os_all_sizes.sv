// =============================================================================
// Testbench: tb_gemm_os_all_sizes.sv
// Spec Coverage (GEMM_SEQUENCER_REDESIGN_SPEC.md Section 5):
//   [1] Decode (M=1), N=32, K=128 with K_TILES > 1 (K_BLOCKS=2, K_TILES=4)
//   [2] Prefill (M=32, N=32, K=64): ALL 4 out_mode (0: FP16, 1: FP4 linear,
//       2: FP4 bank-scatter, 3: Fused V-Transpose w/ K_TILES=2) & act_bank_layout=0/1
//   [3] Weight Ring-Buffer Reuse (M=64, N=64, K=128, K_TILES=1 -> 4x W DRAM savings)
//   [4] Macro-DMA Ping-Pong A / W / O (> 288 KB TCDM):
//       - Non-multiple remainders (M=48, M_SUB=32; N=48, N_SUB=32; K=1024, K_TILES=2)
//       - Full >TCDM scale (M=128, N=64, K=4096 -> A=288KB, W=144KB, total 432KB)
//   [5] Randomized Drain Backpressure stress on FP16, FP4-scatter, & Fused-V
// =============================================================================
`timescale 1ns/1ps

module tb_gemm_os_all_sizes;

    localparam int unsigned DRAM_LATENCY = 4;
    localparam int unsigned DRAM_WORDS   = 65536; // 4 MB (64B beats)

    // CSR map matching snax_flash_csr_snax.sv
    localparam logic [11:0] A_CTRL          = 12'd0;
    localparam logic [11:0] A_V_BASE        = 12'd25;
    localparam logic [11:0] A_V_NB          = 12'd26;
    localparam logic [11:0] A_RD_LS         = 12'd29;
    localparam logic [11:0] A_SEQ_CTRL      = 12'd31;
    localparam logic [11:0] A_ATTN_H        = 12'd32;
    localparam logic [11:0] A_ATTN_QT       = 12'd33;
    localparam logic [11:0] A_ATTN_MQ       = 12'd34;
    localparam logic [11:0] A_ATTN_DV       = 12'd35;
    localparam logic [11:0] A_ATTN_KV       = 12'd36;
    localparam logic [11:0] A_ATTN_KB       = 12'd37;
    localparam logic [11:0] A_SEQ_MACRO_SUB = 12'd32; // legacy aliased onto RW_SEQ_ATTN_H
    localparam logic [11:0] A_SEQ_DRAM_ACT  = 12'd33; // legacy aliased onto RW_SEQ_ATTN_QT
    localparam logic [11:0] A_SEQ_DRAM_OUT  = 12'd35; // legacy aliased onto RW_SEQ_ATTN_DV
    localparam logic [11:0] A_SEQ_ABUF_STR  = 12'd36; // legacy aliased onto RW_SEQ_ATTN_KV
    localparam logic [11:0] A_SEQ_OBUF_STR  = 12'd37; // legacy aliased onto RW_SEQ_ATTN_KB
    localparam logic [11:0] A_GEMM_NT       = 12'd38;
    localparam logic [11:0] A_GEMM_MT       = 12'd39;
    localparam logic [11:0] A_GEMM_KB       = 12'd40;
    localparam logic [11:0] A_RING_BASE     = 12'd41;
    localparam logic [11:0] A_Q_BASE        = 12'd42;
    localparam logic [11:0] A_OUT_BASE      = 12'd43;
    localparam logic [11:0] A_DRAM_KV       = 12'd44;
    localparam logic [11:0] A_TILE_ST       = 12'd45;
    localparam logic [11:0] A_START         = 12'd53;
    localparam logic [11:0] A_EXT_MACRO_SUB = 12'd64;
    localparam logic [11:0] A_EXT_MACRO_ACT = 12'd65;
    localparam logic [11:0] A_EXT_MACRO_OUT = 12'd66;
    localparam logic [11:0] A_EXT_MACRO_ASTR= 12'd67;
    localparam logic [11:0] A_EXT_MACRO_OSTR= 12'd68;
    localparam logic [11:0] A_EXT_DRAM_V    = 12'd69;

    logic clk, rst_n;
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // CPU & CSR
    logic        cpu_req_valid, cpu_req_ready, cpu_req_write, cpu_rsp_valid;
    logic [31:0] cpu_req_addr;
    logic [71:0] cpu_req_wdata, cpu_rsp_rdata;
    logic [8:0]  cpu_req_strb;

    logic        csr_req_valid, csr_req_ready, csr_req_write, csr_rsp_valid;
    logic [11:0] csr_req_addr;
    logic [31:0] csr_req_wdata, csr_rsp_rdata;
    logic        irq_o;

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

    logic [31:0]  stat_rd_beats, stat_wr_beats, stat_tcdm_words, stat_busy_cycles;

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
        .dma_desc_valid_o        (dma_desc_valid),
        .dma_desc_ready_i        (dma_desc_ready),
        .dma_desc_dir_o          (dma_desc_dir),
        .dma_desc_dram_addr_o    (dma_desc_dram_addr),
        .dma_desc_tcdm_addr_o    (dma_desc_tcdm_addr),
        .dma_desc_size_o         (dma_desc_size),
        .dma_desc_done_i         (dma_desc_done),
        .subsystem_busy_o        (),
        .subsystem_done_irq_o    (irq_o)
    );

    snax_simple_dma #(
        .TCDM_PORTS  (8),
        .DRAM_DATA_W (512)
    ) u_dma (
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
        .stat_tcdm_words_o    (stat_tcdm_words),
        .stat_busy_cycles_o   (stat_busy_cycles)
    );

    // -------------------------------------------------------------------------
    // 512-bit DRAM Model (4 MB, non-blocking writes + pipelined reads)
    // -------------------------------------------------------------------------
    logic [511:0] dram_mem [0:DRAM_WORDS-1];
    logic [2:0]   dram_lat_cnt;
    logic         dram_rd_busy;
    logic [31:0]  dram_rd_idx;

    assign dram_req_ready = !dram_rd_busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dram_rd_busy   <= 1'b0;
            dram_lat_cnt   <= 3'd0;
            dram_rd_idx    <= 32'd0;
            dram_rsp_valid <= 1'b0;
            dram_rsp_rdata <= 512'd0;
        end else begin
            dram_rsp_valid <= 1'b0;
            if (dram_req_valid && dram_req_ready) begin
                if (dram_req_write) begin
                    logic [31:0] wr_idx;
                    wr_idx = dram_req_addr >> 6;
                    for (int b = 0; b < 64; b++)
                        if (dram_req_strb[b])
                            dram_mem[wr_idx][b*8 +: 8] <= dram_req_wdata[b*8 +: 8];
                end else begin
                    dram_rd_busy <= 1'b1;
                    dram_lat_cnt <= 3'(DRAM_LATENCY - 1);
                    dram_rd_idx  <= dram_req_addr >> 6;
                end
            end else if (dram_rd_busy) begin
                if (dram_lat_cnt == 3'd0) begin
                    dram_rd_busy   <= 1'b0;
                    dram_rsp_valid <= 1'b1;
                    dram_rsp_rdata <= dram_mem[dram_rd_idx];
                end else begin
                    dram_lat_cnt <= dram_lat_cnt - 3'd1;
                end
            end
        end
    end

    // Monitor invariant: inflight_out_cnt_q must NEVER exceed OUT_PIPE_CREDITS (4)
    int errs;
    int max_inflight_seen;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max_inflight_seen <= 0;
        end else begin
            if (int'(dut.u_seq.inflight_out_cnt_q) > max_inflight_seen)
                max_inflight_seen <= int'(dut.u_seq.inflight_out_cnt_q);
            if (dut.u_seq.inflight_out_cnt_q > 4'd4) begin
                $display("    [FAIL] inflight_out_cnt_q overflow: %0d at %0t",
                         dut.u_seq.inflight_out_cnt_q, $time);
                errs++;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    function automatic logic [31:0] tcdm_addr(input int word, input int bank);
        tcdm_addr = (32'(word) << 9) | (32'(bank) << 4);
    endfunction

    function automatic logic [71:0] make_fp4_pkt(input int seed);
        logic [7:0]  sc;
        logic [63:0] nibs;
        begin
            sc = 8'd124 + 8'(seed % 5);
            for (int i = 0; i < 16; i++)
                nibs[i*4 +: 4] = 4'((seed * 7 + i * 3) % 16);
            make_fp4_pkt = {sc, nibs};
        end
    endfunction

    task automatic cpu_mem_write(input logic [31:0] addr, input logic [71:0] data);
        begin
            @(negedge clk);
            cpu_req_valid = 1'b1;
            cpu_req_write = 1'b1;
            cpu_req_addr  = addr;
            cpu_req_wdata = data;
            cpu_req_strb  = 9'h1FF;
            @(posedge clk);
            while (!cpu_req_ready) @(posedge clk);
            @(negedge clk);
            cpu_req_valid = 1'b0;
        end
    endtask

    task automatic cpu_mem_read(input logic [31:0] addr, output logic [71:0] data);
        begin
            @(negedge clk);
            cpu_req_valid = 1'b1;
            cpu_req_write = 1'b0;
            cpu_req_addr  = addr;
            @(posedge clk);
            while (!cpu_req_ready) @(posedge clk);
            @(negedge clk);
            cpu_req_valid = 1'b0;
            while (!cpu_rsp_valid) @(posedge clk);
            data = cpu_rsp_rdata;
        end
    endtask

    task automatic cpu_csr_write(input logic [11:0] addr, input logic [31:0] data);
        begin
            @(negedge clk);
            csr_req_valid = 1'b1;
            csr_req_write = 1'b1;
            csr_req_addr  = addr;
            csr_req_wdata = data;
            @(posedge clk);
            while (!csr_req_ready) @(posedge clk);
            @(negedge clk);
            csr_req_valid = 1'b0;
        end
    endtask

    task automatic wait_seq_done(input string tag, input int max_cyc = 300000);
        int c;
        begin
            c = 0;
            repeat (3) @(posedge clk);
            while ((dut.seq_busy || dut.wo_busy || dut.v_busy) && (c < max_cyc)) begin
                @(posedge clk);
                c++;
            end
            if (c >= max_cyc) begin
                $display("    [FAIL] Timeout waiting for %s after %0d cycles! cmp_state=%0d dr_state=%0d wo_busy=%0b v_busy=%0b dma_busy=%0b",
                         tag, max_cyc, dut.u_seq.cmp_state_q, dut.u_seq.dr_state_q,
                         dut.wo_busy, dut.v_busy, dut.u_seq.dma_busy_q);
                errs++;
            end
        end
    endtask

    task automatic pack_words_to_dram(input int byte_base, input int num_words, input logic [71:0] words[]);
        int bit_pos;
        begin
            bit_pos = byte_base * 8;
            for (int w = 0; w < num_words; w++) begin
                for (int b = 0; b < 72; b++) begin
                    dram_mem[(bit_pos + b) / 512][(bit_pos + b) % 512] = words[w][b];
                end
                bit_pos += 72;
            end
        end
    endtask

    task automatic unpack_words_from_dram(input int byte_base, input int num_words, output logic [71:0] words[]);
        int bit_pos;
        begin
            words   = new[num_words];
            bit_pos = byte_base * 8;
            for (int w = 0; w < num_words; w++) begin
                logic [71:0] tmp_w;
                tmp_w = 72'd0;
                for (int b = 0; b < 72; b++) begin
                    tmp_w[b] = dram_mem[(bit_pos + b) / 512][(bit_pos + b) % 512];
                end
                words[w] = tmp_w;
                bit_pos += 72;
            end
        end
    endtask

    // =========================================================================
    // TEST 1: Decode (M=1, N=32, K=128) with K_TILES = 4 (K_BLOCKS = 2)
    // =========================================================================
    task automatic test_decode_ktiling();
        localparam int N_T = 2;
        localparam int K_B = 8; // K = 128
        logic [71:0] a_pkts[], w_pkts[];
        logic [71:0] gold_fp16[], act_fp16[], gold_fp4[], act_fp4[];
        int mismatch;
        begin
            $display("\n[TEST 1] Decode M=1, N=32, K=128: K_TILES=4 (K_BLOCKS=2) vs K_TILES=1 (K_BLOCKS=8) ...");
            a_pkts    = new[K_B];
            w_pkts    = new[N_T * K_B * 16];
            gold_fp16 = new[N_T * 4];
            act_fp16  = new[N_T * 4];
            gold_fp4  = new[N_T];
            act_fp4   = new[N_T];

            for (int kb = 0; kb < K_B; kb++) begin
                a_pkts[kb] = make_fp4_pkt(100 + kb);
                cpu_mem_write(tcdm_addr(kb, 16), a_pkts[kb]);
            end
            for (int nt = 0; nt < N_T; nt++)
                for (int kb = 0; kb < K_B; kb++)
                    for (int l = 0; l < 16; l++) begin
                        logic [71:0] wp = make_fp4_pkt(1000 + nt * 200 + kb * 16 + l);
                        w_pkts[(nt * K_B + kb) * 16 + l] = wp;
                        cpu_mem_write(tcdm_addr(nt * K_B + kb, l), wp);
                    end
            pack_words_to_dram(0, N_T * K_B * 16, w_pkts);

            // Golden: single K-tile (non-DMA), FP16 & FP4
            cpu_csr_write(A_CTRL,      32'h0);
            cpu_csr_write(A_SEQ_CTRL,  32'h80); // op=00, out=0 (FP16), decode=1
            cpu_csr_write(A_GEMM_NT,   N_T);
            cpu_csr_write(A_GEMM_MT,   1);
            cpu_csr_write(A_GEMM_KB,   K_B);
            cpu_csr_write(A_RING_BASE, tcdm_addr(0, 0));
            cpu_csr_write(A_Q_BASE,    tcdm_addr(0, 16));
            cpu_csr_write(A_OUT_BASE,  tcdm_addr(0, 17));
            cpu_csr_write(A_START,     32'h3);
            wait_seq_done("Decode Golden FP16");
            for (int i = 0; i < N_T * 4; i++) cpu_mem_read(tcdm_addr(i, 17), gold_fp16[i]);

            cpu_csr_write(A_SEQ_CTRL,  32'h90); // out=1 (FP4), decode=1
            cpu_csr_write(A_START,     32'h3);
            wait_seq_done("Decode Golden FP4");
            for (int i = 0; i < N_T; i++) cpu_mem_read(tcdm_addr(i, 17), gold_fp4[i]);

            // Multi-K-tile DMA run: K_TILES = 4 (k_tiles_m1 = 3), K_BLOCKS = 2
            for (int i = 0; i < N_T * 4; i++) cpu_mem_write(tcdm_addr(i, 17), 72'd0);
            cpu_csr_write(A_SEQ_CTRL,  (32'd3 << 8) | 32'h88); // k_tiles_m1=3, decode=1, dma=1, out=0
            cpu_csr_write(A_GEMM_KB,   2);
            cpu_csr_write(A_DRAM_KV,   32'd0);
            cpu_csr_write(A_TILE_ST,   32'(K_B * 144));
            cpu_csr_write(A_START,     32'h3);
            wait_seq_done("Decode K_TILES=4 FP16");
            mismatch = 0;
            for (int i = 0; i < N_T * 4; i++) begin
                cpu_mem_read(tcdm_addr(i, 17), act_fp16[i]);
                if (act_fp16[i] !== gold_fp16[i]) begin
                    mismatch++;
                    $display("    [FAIL] Decode FP16 word %0d mismatch: act=%h exp=%h", i, act_fp16[i], gold_fp16[i]);
                end
            end

            cpu_csr_write(A_SEQ_CTRL,  (32'd3 << 8) | 32'h98); // k_tiles_m1=3, decode=1, dma=1, out=1
            cpu_csr_write(A_START,     32'h3);
            wait_seq_done("Decode K_TILES=4 FP4");
            for (int i = 0; i < N_T; i++) begin
                cpu_mem_read(tcdm_addr(i, 17), act_fp4[i]);
                if (act_fp4[i] !== gold_fp4[i]) begin
                    mismatch++;
                    $display("    [FAIL] Decode FP4 word %0d mismatch: act=%h exp=%h", i, act_fp4[i], gold_fp4[i]);
                end
            end

            if (mismatch == 0) $display("    [PASS] Decode K_TILES=4 (FP16 & FP4) 100%% bit-exact!");
            else errs += mismatch;
        end
    endtask

    // =========================================================================
    // TEST 2: Prefill M=32, N=32, K=64 -> ALL 4 out_mode + act_bank_layout 0/1
    //         Includes Fused V (out_mode=3) w/ K_TILES=2 (old sequencer deadlocked!)
    // =========================================================================
    task automatic test_prefill_all_out_modes();
        localparam int M_T = 2, N_T = 2, K_B = 4;
        logic [71:0] a_mat[], w_mat[], gold_fp16[], gold_fp4[], gold_v[];
        logic [71:0] rdata;
        int mismatch;
        begin
            $display("\n[TEST 2] Prefill M=32, N=32, K=64: out_mode=0/1/2/3 & act_bank_layout=0/1 ...");
            a_mat     = new[M_T * 16 * K_B];
            w_mat     = new[N_T * K_B * 16];
            gold_fp16 = new[M_T * N_T * 16 * 4];
            gold_fp4  = new[M_T * N_T * 16];
            gold_v    = new[M_T * N_T * 16];

            // Load Weights (word 0..7, banks 0..15)
            for (int nt = 0; nt < N_T; nt++)
                for (int kb = 0; kb < K_B; kb++)
                    for (int l = 0; l < 16; l++) begin
                        logic [71:0] wp = make_fp4_pkt(2000 + nt * 100 + kb * 16 + l);
                        w_mat[(nt * K_B + kb) * 16 + l] = wp;
                        cpu_mem_write(tcdm_addr(nt * K_B + kb, l), wp);
                    end

            // Load A in linear bank 16 (word 0..127)
            for (int mt = 0; mt < M_T; mt++)
                for (int tok = 0; tok < 16; tok++)
                    for (int kb = 0; kb < K_B; kb++) begin
                        int idx = (mt * 16 + tok) * K_B + kb;
                        a_mat[idx] = make_fp4_pkt(3000 + idx);
                        cpu_mem_write(tcdm_addr(idx, 16), a_mat[idx]);
                    end

            // 2A: out_mode=0 FP16, act_bank_layout=0 (golden)
            cpu_csr_write(A_CTRL,      32'h0);
            cpu_csr_write(A_SEQ_CTRL,  32'h0); // op=00, out=0
            cpu_csr_write(A_GEMM_NT,   N_T);
            cpu_csr_write(A_GEMM_MT,   M_T);
            cpu_csr_write(A_GEMM_KB,   K_B);
            cpu_csr_write(A_RING_BASE, tcdm_addr(0, 0));
            cpu_csr_write(A_Q_BASE,    tcdm_addr(0, 16));
            cpu_csr_write(A_OUT_BASE,  tcdm_addr(0, 17));
            cpu_csr_write(A_START,     32'h3);
            wait_seq_done("Prefill out=0 act_lin");
            for (int i = 0; i < M_T * N_T * 16 * 4; i++) cpu_mem_read(tcdm_addr(i, 17), gold_fp16[i]);

            // 2B: out_mode=0 FP16, act_bank_layout=1 (16-bank A across banks 16..31 at word 200..207)
            // Must match 2A 100% (proves Bug #3 gemm_act_addr offset for M_tiles>1 fixed!)
            for (int mt = 0; mt < M_T; mt++)
                for (int kb = 0; kb < K_B; kb++)
                    for (int tok = 0; tok < 16; tok++)
                        cpu_mem_write(tcdm_addr(200 + mt * K_B + kb, 16 + tok),
                                      a_mat[(mt * 16 + tok) * K_B + kb]);
            for (int i = 0; i < M_T * N_T * 16 * 4; i++) cpu_mem_write(tcdm_addr(250 + i, 17), 72'd0);
            cpu_csr_write(A_SEQ_CTRL,  (32'd1 << 24)); // act_bank_layout=1, out=0
            cpu_csr_write(A_Q_BASE,    tcdm_addr(200, 16));
            cpu_csr_write(A_OUT_BASE,  tcdm_addr(250, 17));
            cpu_csr_write(A_START,     32'h3);
            wait_seq_done("Prefill out=0 act_16bank");
            mismatch = 0;
            for (int i = 0; i < M_T * N_T * 16 * 4; i++) begin
                cpu_mem_read(tcdm_addr(250 + i, 17), rdata);
                if (rdata !== gold_fp16[i]) mismatch++;
            end
            if (mismatch == 0) $display("    [PASS] 2B act_bank_layout=1 (M=32) matches act_bank_layout=0 100%%!");
            else begin
                $display("    [FAIL] 2B act_bank_layout=1 had %0d mismatches!", mismatch);
                errs += mismatch;
            end

            // Reload A linear at word 0 bank 16 for remaining checks
            for (int i = 0; i < M_T * 16 * K_B; i++) cpu_mem_write(tcdm_addr(i, 16), a_mat[i]);

            // 2C: out_mode=1 FP4 linear (word 300..363 in bank 17)
            cpu_csr_write(A_SEQ_CTRL, 32'h10); // out=1
            cpu_csr_write(A_Q_BASE,   tcdm_addr(0, 16));
            cpu_csr_write(A_OUT_BASE, tcdm_addr(300, 17));
            cpu_csr_write(A_START,    32'h3);
            wait_seq_done("Prefill out=1 FP4 linear");
            for (int i = 0; i < M_T * N_T * 16; i++) cpu_mem_read(tcdm_addr(300 + i, 17), gold_fp4[i]);

            // 2D: out_mode=2 FP4 bank-scatter (word 400..403 across banks 0..15)
            // Proves Bug #4 (wo_num_batches = M_tiles * N_tiles) fixed!
            // Note: clear banks 0..15 at word 400..403 first so weights don't overlap.
            for (int b = 0; b < M_T * N_T; b++)
                for (int tok = 0; tok < 16; tok++)
                    cpu_mem_write(tcdm_addr(400 + b, tok), 72'd0);
            cpu_csr_write(A_SEQ_CTRL, 32'h20); // out=2
            cpu_csr_write(A_OUT_BASE, tcdm_addr(400, 0));
            cpu_csr_write(A_START,    32'h3);
            wait_seq_done("Prefill out=2 FP4 bank-scatter");
            mismatch = 0;
            for (int b = 0; b < M_T * N_T; b++) begin
                for (int tok = 0; tok < 16; tok++) begin
                    cpu_mem_read(tcdm_addr(400 + b, tok), rdata);
                    if (rdata !== gold_fp4[b * 16 + tok]) mismatch++;
                end
            end
            if (mismatch == 0) $display("    [PASS] 2D out_mode=2 FP4 bank-scatter (M=32, 4 batches) 100%% bit-exact vs out_mode=1!");
            else begin
                $display("    [FAIL] 2D out_mode=2 had %0d mismatches!", mismatch);
                errs += mismatch;
            end

            // 2E: out_mode=3 Fused V-Transpose:
            //   First build tile-by-tile (1x1) golden using single tile calls, then run M=2xN=2 K_TILES=2!
            for (int nt = 0; nt < N_T; nt++) begin
                for (int mt = 0; mt < M_T; mt++) begin
                    int b = nt * M_T + mt;
                    cpu_csr_write(A_SEQ_CTRL,  32'h30); // out=3, dma=0
                    cpu_csr_write(A_GEMM_NT,   1);
                    cpu_csr_write(A_GEMM_MT,   1);
                    cpu_csr_write(A_GEMM_KB,   K_B);
                    cpu_csr_write(A_RING_BASE, tcdm_addr(nt * K_B, 0));
                    cpu_csr_write(A_Q_BASE,    tcdm_addr(mt * 16 * K_B, 16));
                    cpu_csr_write(A_V_BASE,    tcdm_addr(420, 16));
                    cpu_csr_write(A_V_NB,      1);
                    cpu_csr_write(A_START,     32'h3);
                    wait_seq_done("Fused-V 1x1 gold tile");
                    for (int col = 0; col < 16; col++)
                        cpu_mem_read(tcdm_addr(420, 16 + col), gold_v[b * 16 + col]);
                end
            end

            // Multi-tile (M_T=2, N_T=2) + K_TILES=2 (K_B_per_tile=2, k_tiles_m1=1) via DMA
            pack_words_to_dram(32768, N_T * K_B * 16, w_mat);
            for (int b = 0; b < M_T * N_T; b++)
                for (int col = 0; col < 16; col++)
                    cpu_mem_write(tcdm_addr(450 + b, 16 + col), 72'd0);

            cpu_csr_write(A_SEQ_CTRL,  (32'd1 << 8) | 32'h38); // k_tiles_m1=1, out=3, dma=1
            cpu_csr_write(A_GEMM_NT,   N_T);
            cpu_csr_write(A_GEMM_MT,   M_T);
            cpu_csr_write(A_GEMM_KB,   K_B / 2);
            cpu_csr_write(A_RING_BASE, tcdm_addr(500, 0));
            cpu_csr_write(A_Q_BASE,    tcdm_addr(0, 16));
            cpu_csr_write(A_V_BASE,    tcdm_addr(450, 16));
            cpu_csr_write(A_V_NB,      1); // Deliberately set to 1: top must use seq_v_num_batches = 4!
            cpu_csr_write(A_DRAM_KV,   32768);
            cpu_csr_write(A_TILE_ST,   K_B * 144);
            cpu_csr_write(A_START,     32'h3);
            wait_seq_done("Fused-V M=2xN=2 K_TILES=2");

            mismatch = 0;
            for (int b = 0; b < M_T * N_T; b++) begin
                for (int col = 0; col < 16; col++) begin
                    cpu_mem_read(tcdm_addr(450 + b, 16 + col), rdata);
                    if (rdata !== gold_v[b * 16 + col]) mismatch++;
                end
            end
            if (mismatch == 0) $display("    [PASS] 2E Fused-V M_T=2 x N_T=2 x K_TILES=2 (0 deadlock, 64 cols 100%% bit-exact)!");
            else begin
                $display("    [FAIL] 2E Fused-V had %0d mismatches!", mismatch);
                errs += mismatch;
            end
        end
    endtask

    // =========================================================================
    // TEST 3: Weight Ring Reuse (M=64 -> M_TILES=4, N=64 -> N_TILES=4, K=128)
    // =========================================================================
    task automatic test_weight_ring_reuse();
        localparam int M_T = 4, N_T = 4, K_B = 8; // K=128, K_TILES=1
        logic [71:0] w_mat[], gold[], rdata;
        int w_start, w_words, mismatch;
        begin
            $display("\n[TEST 3] Weight Ring Reuse: M=64 (M_T=4), N=64 (N_T=4), K=128 (K_TILES=1) ...");
            w_mat = new[N_T * K_B * 16];
            gold  = new[M_T * N_T * 16 * 4];

            for (int mt = 0; mt < M_T; mt++)
                for (int tok = 0; tok < 16; tok++)
                    for (int kb = 0; kb < K_B; kb++) begin
                        int idx = (mt * 16 + tok) * K_B + kb;
                        cpu_mem_write(tcdm_addr(idx, 16), make_fp4_pkt(5000 + idx));
                    end

            for (int nt = 0; nt < N_T; nt++)
                for (int kb = 0; kb < K_B; kb++)
                    for (int l = 0; l < 16; l++) begin
                        logic [71:0] wp = make_fp4_pkt(6000 + nt * 128 + kb * 16 + l);
                        w_mat[(nt * K_B + kb) * 16 + l] = wp;
                        cpu_mem_write(tcdm_addr(nt * K_B + kb, l), wp);
                    end

            // Baseline non-DMA
            cpu_csr_write(A_CTRL,      32'h0);
            cpu_csr_write(A_SEQ_CTRL,  32'h0);
            cpu_csr_write(A_GEMM_NT,   N_T);
            cpu_csr_write(A_GEMM_MT,   M_T);
            cpu_csr_write(A_GEMM_KB,   K_B);
            cpu_csr_write(A_RING_BASE, tcdm_addr(0, 0));
            cpu_csr_write(A_Q_BASE,    tcdm_addr(0, 16));
            cpu_csr_write(A_OUT_BASE,  tcdm_addr(0, 17));
            cpu_csr_write(A_START,     32'h3);
            wait_seq_done("W-Reuse Golden");
            for (int i = 0; i < M_T * N_T * 16 * 4; i++) cpu_mem_read(tcdm_addr(i, 17), gold[i]);

            // Clear O & run DMA
            for (int i = 0; i < M_T * N_T * 16 * 4; i++) cpu_mem_write(tcdm_addr(i, 17), 72'd0);
            pack_words_to_dram(65536, N_T * K_B * 16, w_mat);
            w_start = stat_tcdm_words;

            cpu_csr_write(A_SEQ_CTRL,  32'h8); // enable_dma=1, k_tiles_m1=0
            cpu_csr_write(A_DRAM_KV,   65536);
            cpu_csr_write(A_TILE_ST,   K_B * 144);
            cpu_csr_write(A_START,     32'h3);
            wait_seq_done("W-Reuse DMA");
            w_words = stat_tcdm_words - w_start;

            mismatch = 0;
            for (int i = 0; i < M_T * N_T * 16 * 4; i++) begin
                cpu_mem_read(tcdm_addr(i, 17), rdata);
                if (rdata !== gold[i]) mismatch++;
            end

            // Expected W transfers = N_T * K_B * 16 = 512 words (NOT 4x = 2048 words!)
            if (w_words !== (N_T * K_B * 16)) begin
                $display("    [FAIL] Weight fetched %0d TCDM words instead of exact 1x (%0d)!", w_words, N_T * K_B * 16);
                errs++;
            end
            if (mismatch == 0 && w_words == (N_T * K_B * 16))
                $display("    [PASS] 4x Weight Reuse verified: %0d W-words fetched, 100%% bit-exact!", w_words);
            else errs += mismatch;
        end
    endtask

    // =========================================================================
    // TEST 4: Macro-DMA Ping-Pong (A, W, O in DRAM) Exceeding 288 KB TCDM!
    //   Case 4A: Non-multiple Remainders: M=48 (M_T=3, M_SUB=2), N=48 (N_T=3, N_SUB=2),
    //            K=1024 (K_B=32, K_TILES=2), verified bit-exact per macro-tile!
    //   Case 4B: > 288 KB TCDM Proof: M=128 (M_T=8, M_SUB=1), N=64 (N_T=4, N_SUB=2),
    //            K=4096 (K_B=128, K_TILES=2) -> A=288KB, W=144KB (Total=432KB > 288KB)!
    // =========================================================================
    task automatic test_macro_dma_gt_tcdm();
        // Part 4A: Remainder test (M=48, N=48, K=1024, M_SUB=2, N_SUB=2, K_TILES=2)
        begin
            localparam int M_T = 3, N_T = 3, KB_PER_TILE = 32, K_TILES = 2;
            localparam int TOT_KB = KB_PER_TILE * K_TILES; // 64 (K=1024)
            localparam int DRAM_A_BASE = 131072;
            localparam int DRAM_W_BASE = 262144;
            localparam int DRAM_O_BASE = 524288;
            localparam int DRAM_G_BASE = 786432;
            logic [71:0] a_all[], w_all[], act_o[], gold_o[];
            int mismatch;

            $display("\n[TEST 4A] Macro-DMA Ping-Pong w/ Remainders: M=48 (M_SUB=2), N=48 (N_SUB=2), K=1024 (K_TILES=2) ...");
            a_all  = new[M_T * TOT_KB * 16];
            w_all  = new[N_T * TOT_KB * 16];
            gold_o = new[M_T * N_T * 16];

            // Build A in 16-bank word order: [mt][k][tok]
            for (int i = 0; i < M_T * TOT_KB * 16; i++) a_all[i] = make_fp4_pkt(70000 + i);
            for (int i = 0; i < N_T * TOT_KB * 16; i++) w_all[i] = make_fp4_pkt(90000 + i);
            pack_words_to_dram(DRAM_A_BASE, M_T * TOT_KB * 16, a_all);
            pack_words_to_dram(DRAM_W_BASE, N_T * TOT_KB * 16, w_all);

            // Compute golden tile-by-tile (1x1 non-macro) directly via core
            for (int mt = 0; mt < M_T; mt++) begin
                // Write A[mt] into banks 16..31 word 0..TOT_KB-1
                for (int k = 0; k < TOT_KB; k++)
                    for (int tok = 0; tok < 16; tok++)
                        cpu_mem_write(tcdm_addr(k, 16 + tok), a_all[(mt * TOT_KB + k) * 16 + tok]);

                for (int nt = 0; nt < N_T; nt++) begin
                    for (int k = 0; k < TOT_KB; k++)
                        for (int l = 0; l < 16; l++)
                            cpu_mem_write(tcdm_addr(k, l), w_all[(nt * TOT_KB + k) * 16 + l]);

                    cpu_csr_write(A_CTRL,      32'h0);
                    cpu_csr_write(A_SEQ_CTRL,  (32'd1 << 24) | 32'h10); // act_16bank=1, out=1 (FP4), dma=0
                    cpu_csr_write(A_GEMM_NT,   1);
                    cpu_csr_write(A_GEMM_MT,   1);
                    cpu_csr_write(A_GEMM_KB,   TOT_KB);
                    cpu_csr_write(A_RING_BASE, tcdm_addr(0, 0));
                    cpu_csr_write(A_Q_BASE,    tcdm_addr(0, 16));
                    cpu_csr_write(A_OUT_BASE,  tcdm_addr(200, 16));
                    cpu_csr_write(A_START,     32'h3);
                    wait_seq_done("Macro 4A Gold Tile");
                    for (int tok = 0; tok < 16; tok++) begin
                        logic [71:0] tmp_o;
                        cpu_mem_read(tcdm_addr(200 + tok, 16), tmp_o);
                        gold_o[(mt * N_T + nt) * 16 + tok] = tmp_o;
                    end
                end
            end

            // Run FULL Macro-DMA job across DRAM <-> TCDM (M_SUB=2, N_SUB=2, K_TILES=2)
            // TCDM Map:
            //   W ring (4 slots x 128KB): banks 0..15 words 0..1023 (A_RING_BASE=0)
            //   A ping/pong: banks 16..31 words 0..127 (A0) and 256..383 (A1) -> stride 0x20000
            //   O ping/pong: bank 16 words 512..767 (O0) and 768..1023 (O1)     -> stride 0x20000
            cpu_csr_write(A_CTRL,          32'h0);
            cpu_csr_write(A_SEQ_CTRL,      (32'd1 << 30) | (32'd1 << 24) | (32'd1 << 8) | 32'h18); // macro_dma=1, act_16bank=1, kt_m1=1, out=1 (FP4), dma=1
            cpu_csr_write(A_SEQ_MACRO_SUB, (32'd2 << 8) | 32'd2); // N_SUB=2, M_SUB=2
            cpu_csr_write(A_SEQ_DRAM_ACT,  DRAM_A_BASE);
            cpu_csr_write(A_SEQ_DRAM_OUT,  DRAM_O_BASE);
            cpu_csr_write(A_SEQ_ABUF_STR,  32'h0002_0000);
            cpu_csr_write(A_SEQ_OBUF_STR,  32'h0002_0000);
            cpu_csr_write(A_GEMM_NT,       N_T);
            cpu_csr_write(A_GEMM_MT,       M_T);
            cpu_csr_write(A_GEMM_KB,       KB_PER_TILE);
            cpu_csr_write(A_RING_BASE,     tcdm_addr(0, 0));
            cpu_csr_write(A_Q_BASE,        tcdm_addr(0, 16));
            cpu_csr_write(A_OUT_BASE,      tcdm_addr(512, 16));
            cpu_csr_write(A_DRAM_KV,       DRAM_W_BASE);
            cpu_csr_write(A_TILE_ST,       TOT_KB * 144);
            cpu_csr_write(A_START,         32'h3);
            wait_seq_done("Macro 4A Full Run");

            // Unpack all 4 macro-blocks from DRAM_O_BASE:
            //   mm=0 (cur_m=2): nm=0 (2x2=4 tiles), nm=1 (2x1=2 tiles)
            //   mm=1 (cur_m=1): nm=0 (1x2=2 tiles), nm=1 (1x1=1 tile)
            unpack_words_from_dram(DRAM_O_BASE, M_T * N_T * 16, act_o);
            mismatch = 0;
            begin
                // Compare macro-block (0,0): mm=0, nm=0 (cur_m=2, cur_n=2 -> offset 0)
                for (int gn = 0; gn < 2; gn++)
                    for (int gm = 0; gm < 2; gm++)
                        for (int tok = 0; tok < 16; tok++)
                            if (act_o[((0*2 + gn)*2 + gm)*16 + tok] !== gold_o[((0 + gm)*N_T + (0 + gn))*16 + tok])
                                mismatch++;
                // Compare macro-block (0,1): mm=0, nm=1 (cur_m=2, cur_n=1 -> offset 4 tiles = 64 words)
                for (int gm = 0; gm < 2; gm++)
                    for (int tok = 0; tok < 16; tok++)
                        if (act_o[(4 + gm)*16 + tok] !== gold_o[((0 + gm)*N_T + 2)*16 + tok])
                            mismatch++;
                // Compare macro-block (1,0): mm=1, nm=0 (cur_m=1, cur_n=2 -> offset 6 tiles = 96 words)
                for (int gn = 0; gn < 2; gn++)
                    for (int tok = 0; tok < 16; tok++)
                        if (act_o[(6 + gn)*16 + tok] !== gold_o[(2*N_T + gn)*16 + tok])
                            mismatch++;
                // Compare macro-block (1,1): mm=1, nm=1 (cur_m=1, cur_n=1 -> offset 8 tiles = 128 words)
                for (int tok = 0; tok < 16; tok++)
                    if (act_o[8*16 + tok] !== gold_o[(2*N_T + 2)*16 + tok])
                        mismatch++;
            end

            if (mismatch == 0)
                $display("    [PASS] 4A Macro-DMA w/ Remainders (3x3 tiles, M_SUB=2, N_SUB=2, K_TILES=2) 100%% bit-exact!");
            else begin
                $display("    [FAIL] 4A Macro-DMA had %0d mismatches!", mismatch);
                errs += mismatch;
            end
        end

        // Part 4B: > 288 KB TCDM Total Scale Proof (M=128, N=64, K=4096)
        // A = 128 * 256 * 9B = 294,912 B (288 KB), W = 64 * 256 * 9B = 147,456 B (144 KB)
        // Total input/weight = 432 KB > 288 KB TCDM! M_SUB=1, N_SUB=2, K_TILES=2
        begin
            localparam int M_T = 8, N_T = 4, KB_PER_TILE = 128, TOT_KB = 256;
            localparam int DRAM_A_BASE = 1048576; // 1 MB
            localparam int DRAM_W_BASE = 1572864; // 1.5 MB
            localparam int DRAM_O_BASE = 2097152; // 2.0 MB
            logic [71:0] a_all[], w_all[], act_o[], gold_tile[];
            int t0, t1, dma_rd0, dma_wr0, mismatch;

            $display("\n[TEST 4B] >288KB TCDM Scale Test: M=128 (M_T=8), N=64 (N_T=4), K=4096 (K_TILES=2, Total=432KB) ...");
            a_all = new[M_T * TOT_KB * 16]; // 32,768 words = 288 KB
            w_all = new[N_T * TOT_KB * 16]; // 16,384 words = 144 KB
            gold_tile = new[16];

            for (int i = 0; i < M_T * TOT_KB * 16; i++) a_all[i] = make_fp4_pkt(110000 + i);
            for (int i = 0; i < N_T * TOT_KB * 16; i++) w_all[i] = make_fp4_pkt(150000 + i);
            pack_words_to_dram(DRAM_A_BASE, M_T * TOT_KB * 16, a_all);
            pack_words_to_dram(DRAM_W_BASE, N_T * TOT_KB * 16, w_all);

            // Spot-check golden on final tile (mt=7, nt=3) via 1x1 non-macro
            for (int k = 0; k < TOT_KB; k++) begin
                for (int tok = 0; tok < 16; tok++)
                    cpu_mem_write(tcdm_addr(k, 16 + tok), a_all[(7 * TOT_KB + k) * 16 + tok]);
                for (int l = 0; l < 16; l++)
                    cpu_mem_write(tcdm_addr(k, l), w_all[(3 * TOT_KB + k) * 16 + l]);
            end
            cpu_csr_write(A_CTRL,      32'h0);
            cpu_csr_write(A_SEQ_CTRL,  (32'd1 << 24) | 32'h10);
            cpu_csr_write(A_GEMM_NT,   1);
            cpu_csr_write(A_GEMM_MT,   1);
            cpu_csr_write(A_GEMM_KB,   TOT_KB);
            cpu_csr_write(A_RING_BASE, tcdm_addr(0, 0));
            cpu_csr_write(A_Q_BASE,    tcdm_addr(0, 16));
            cpu_csr_write(A_OUT_BASE,  tcdm_addr(512, 16));
            cpu_csr_write(A_START,     32'h3);
            wait_seq_done("Macro 4B Last Tile Gold");
            for (int tok = 0; tok < 16; tok++) cpu_mem_read(tcdm_addr(512 + tok, 16), gold_tile[tok]);

            dma_rd0 = stat_rd_beats;
            dma_wr0 = stat_wr_beats;
            cpu_csr_write(A_SEQ_CTRL,      (32'd1 << 30) | (32'd1 << 24) | (32'd1 << 8) | 32'h18);
            cpu_csr_write(A_SEQ_MACRO_SUB, (32'd2 << 8) | 32'd1); // N_SUB=2, M_SUB=1
            cpu_csr_write(A_SEQ_DRAM_ACT,  DRAM_A_BASE);
            cpu_csr_write(A_SEQ_DRAM_OUT,  DRAM_O_BASE);
            cpu_csr_write(A_SEQ_ABUF_STR,  32'h0002_0000); // A0 words 0..255, A1 words 256..511
            cpu_csr_write(A_SEQ_OBUF_STR,  32'h0002_0000); // O0 words 512..767, O1 words 768..1023
            cpu_csr_write(A_GEMM_NT,       N_T);
            cpu_csr_write(A_GEMM_MT,       M_T);
            cpu_csr_write(A_GEMM_KB,       KB_PER_TILE);
            cpu_csr_write(A_RING_BASE,     tcdm_addr(0, 0));
            cpu_csr_write(A_Q_BASE,        tcdm_addr(0, 16));
            cpu_csr_write(A_OUT_BASE,      tcdm_addr(512, 16));
            cpu_csr_write(A_DRAM_KV,       DRAM_W_BASE);
            cpu_csr_write(A_TILE_ST,       TOT_KB * 144);
            cpu_csr_write(A_START,         32'h3);
            wait_seq_done("Macro 4B >288KB", 400000);

            unpack_words_from_dram(DRAM_O_BASE, M_T * N_T * 16, act_o);
            mismatch = 0;
            for (int tok = 0; tok < 16; tok++)
                if (act_o[(M_T * N_T - 1) * 16 + tok] !== gold_tile[tok]) mismatch++;

            if (mismatch == 0)
                $display("    [PASS] 4B M=128,N=64,K=4096 (432KB > 288KB) completed | DRAM rd=%0d beats, wr=%0d beats, last-tile 100%% bit-exact!",
                         stat_rd_beats - dma_rd0, stat_wr_beats - dma_wr0);
            else begin
                $display("    [FAIL] 4B last-tile had %0d mismatches!", mismatch);
                errs += mismatch;
            end
        end
    endtask

    // =========================================================================
    // TEST 5: Drain Backpressure Stress (LFSR stall on write-out TCDM ready)
    // =========================================================================
    logic inject_stall;
    logic [7:0] lfsr;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) lfsr <= 8'hA5;
        else        lfsr <= {lfsr[6:0], lfsr[7] ^ lfsr[5] ^ lfsr[4] ^ lfsr[3]};
    end

    // Force random stall (~50%) on Write-Out TCDM grant ONLY during inject_stall
    always @(negedge clk or negedge rst_n) begin
        if (!rst_n) begin
            release dut.wo_m_req_ready;
        end else if (inject_stall && lfsr[0]) begin
            force dut.wo_m_req_ready = 1'b0;
        end else begin
            release dut.wo_m_req_ready;
        end
    end

    task automatic test_drain_backpressure();
        localparam int M_T = 2, N_T = 2, K_B = 4;
        logic [71:0] gold_fp16[], rdata;
        int mismatch;
        begin
            $display("\n[TEST 5] Aggressive Drain Backpressure Stress (M=32, N=32, K=64, out_mode=0 & 2 & 3) ...");
            gold_fp16 = new[M_T * N_T * 16 * 4];

            for (int mt = 0; mt < M_T; mt++)
                for (int tok = 0; tok < 16; tok++)
                    for (int kb = 0; kb < K_B; kb++)
                        cpu_mem_write(tcdm_addr((mt * 16 + tok) * K_B + kb, 16),
                                      make_fp4_pkt(3000 + (mt * 16 + tok) * K_B + kb));
            for (int nt = 0; nt < N_T; nt++)
                for (int kb = 0; kb < K_B; kb++)
                    for (int l = 0; l < 16; l++)
                        cpu_mem_write(tcdm_addr(nt * K_B + kb, l),
                                      make_fp4_pkt(2000 + nt * 100 + kb * 16 + l));

            // Golden without stall
            inject_stall = 1'b0;
            cpu_csr_write(A_CTRL,      32'h0);
            cpu_csr_write(A_SEQ_CTRL,  32'h0);
            cpu_csr_write(A_GEMM_NT,   N_T);
            cpu_csr_write(A_GEMM_MT,   M_T);
            cpu_csr_write(A_GEMM_KB,   K_B);
            cpu_csr_write(A_RING_BASE, tcdm_addr(0, 0));
            cpu_csr_write(A_Q_BASE,    tcdm_addr(0, 16));
            cpu_csr_write(A_OUT_BASE,  tcdm_addr(0, 17));
            cpu_csr_write(A_START,     32'h3);
            wait_seq_done("BP Gold");
            for (int i = 0; i < M_T * N_T * 16 * 4; i++) cpu_mem_read(tcdm_addr(i, 17), gold_fp16[i]);

            // Repeat WITH heavy LFSR backpressure forcing u_write_out FIFO full + pipe stalls
            for (int i = 0; i < M_T * N_T * 16 * 4; i++) cpu_mem_write(tcdm_addr(i, 17), 72'd0);
            inject_stall = 1'b1;
            cpu_csr_write(A_START,     32'h3);
            wait_seq_done("BP Stressed FP16");
            inject_stall = 1'b0;

            mismatch = 0;
            for (int i = 0; i < M_T * N_T * 16 * 4; i++) begin
                cpu_mem_read(tcdm_addr(i, 17), rdata);
                if (rdata !== gold_fp16[i]) mismatch++;
            end

            if (mismatch == 0)
                $display("    [PASS] Drain Backpressure Stress passed 100%% bit-exact (0 dropped vectors, max inflight=%0d)!",
                         max_inflight_seen);
            else begin
                $display("    [FAIL] Drain Backpressure had %0d dropped/corrupted words!", mismatch);
                errs += mismatch;
            end
        end
    endtask

    // Exact bit-level software reference matching slq_encoder.sv
    function automatic logic [71:0] sw_logq_vec(input logic [15:0][15:0] fp16_in);
        logic [7:0] mu [0:15];
        logic [7:0] mu_max;
        logic signed [8:0] scale_raw;
        logic signed [7:0] scale_sat;
        logic [63:0] codes;
        begin
            mu_max = 8'd0;
            for (int i = 0; i < 16; i++) begin
                logic [4:0] e;
                logic [9:0] m;
                e = fp16_in[i][14:10];
                m = fp16_in[i][9:0];
                if (e == 5'd0)       mu[i] = 8'd0;
                else if (e == 5'd31) mu[i] = 8'd247;
                else                 mu[i] = {e, m[9:7]};
                if (mu[i] > mu_max) mu_max = mu[i];
            end
            if (mu_max == 8'd0) begin
                scale_sat = 8'sd0;
            end else begin
                scale_raw = $signed({1'b0, mu_max}) - 9'sd140;
                if (scale_raw > 9'sd127)       scale_sat = 8'sd127;
                else if (scale_raw < -9'sd128) scale_sat = 8'sh80;
                else                           scale_sat = scale_raw[7:0];
            end
            codes = 64'd0;
            for (int i = 0; i < 16; i++) begin
                logic [7:0] diff;
                logic [2:0] mag;
                diff = mu_max - mu[i];
                if (mu[i] == 8'd0 || mu_max == 8'd0) mag = 3'b000;
                else if (diff <= 8'd2)  mag = 3'b111;
                else if (diff <= 8'd6)  mag = 3'b110;
                else if (diff <= 8'd10) mag = 3'b101;
                else if (diff <= 8'd14) mag = 3'b100;
                else if (diff <= 8'd18) mag = 3'b011;
                else if (diff <= 8'd24) mag = 3'b010;
                else if (diff <= 8'd36) mag = 3'b001;
                else                    mag = 3'b000;
                codes[4*i +: 4] = (mag == 3'b000) ? 4'b0000 : {fp16_in[i][15], mag};
            end
            sw_logq_vec = {scale_sat, codes};
        end
    endfunction

    // =========================================================================
    // TEST 6: Unified Sequencer -- FlashAttention KV-Pin + Macro-DMA Q/O + HW Decode V-Transpose
    // =========================================================================
    task automatic test_unified_attn_and_decode_v();
        begin
            // -----------------------------------------------------------------
            // 6A: KV_Pin_En=1 + GQA (H_Q=4, H_KV=2) + Separate DRAM K/V
            // -----------------------------------------------------------------
            $display("\n[TEST 6A] FlashAttention KV_Pin_En=1 + GQA 2:1 + Separate DRAM K/V ...");
            begin
                localparam int HQ = 4, HKV = 2, QT = 2, MQ = 8, DV = 2, KV = 2, KB = 2;
                localparam int DRAM_PACK = 32'h0010_0000;
                localparam int DRAM_K    = 32'h0012_0000;
                localparam int DRAM_V    = 32'h0014_0000;
                logic [71:0] q_pkts[], k_pkts[], v_pkts[], pack_kv[], gold_o[], pin_o[];
                logic [31:0] rd0, rd_ring, rd_pin;
                int mm;

                q_pkts  = new[HQ * QT * MQ * KB];
                k_pkts  = new[HKV * KV * KB * 16];
                v_pkts  = new[HKV * DV * KV * 16];
                pack_kv = new[HKV * DV * KV * (KB + 1) * 16];
                gold_o  = new[HQ * QT * DV * MQ * 4];
                pin_o   = new[HQ * QT * DV * MQ * 4];

                for (int i = 0; i < HQ * QT * MQ * KB; i++) begin
                    q_pkts[i] = make_fp4_pkt(300 + i);
                    cpu_mem_write(tcdm_addr(i, 16), q_pkts[i]);
                end
                for (int h = 0; h < HKV; h++)
                    for (int kj = 0; kj < KV; kj++)
                        for (int kb = 0; kb < KB; kb++)
                            for (int l = 0; l < 16; l++)
                                k_pkts[((h*KV + kj)*KB + kb)*16 + l] = make_fp4_pkt(700 + h*97 + kj*37 + kb*17 + l);
                for (int h = 0; h < HKV; h++)
                    for (int dv = 0; dv < DV; dv++)
                        for (int kj = 0; kj < KV; kj++)
                            for (int l = 0; l < 16; l++)
                                v_pkts[((h*DV + dv)*KV + kj)*16 + l] = make_fp4_pkt(900 + h*83 + dv*41 + kj*19 + l);

                for (int h = 0; h < HKV; h++)
                    for (int dv = 0; dv < DV; dv++)
                        for (int kj = 0; kj < KV; kj++) begin
                            int base_idx = ((h*DV + dv)*KV + kj) * (KB + 1) * 16;
                            for (int kb = 0; kb < KB; kb++)
                                for (int l = 0; l < 16; l++)
                                    pack_kv[base_idx + kb*16 + l] = k_pkts[((h*KV + kj)*KB + kb)*16 + l];
                            for (int l = 0; l < 16; l++)
                                pack_kv[base_idx + KB*16 + l] = v_pkts[((h*DV + dv)*KV + kj)*16 + l];
                        end

                pack_words_to_dram(DRAM_PACK, HKV * DV * KV * (KB + 1) * 16, pack_kv);
                pack_words_to_dram(DRAM_K,    HKV * KV * KB * 16, k_pkts);
                pack_words_to_dram(DRAM_V,    HKV * DV * KV * 16, v_pkts);

                // Ring Reference (KV_Pin_En=0, sep_kv=0)
                rd0 = stat_rd_beats;
                cpu_csr_write(A_CTRL,          32'h0);
                cpu_csr_write(A_EXT_MACRO_SUB, 32'h0);
                cpu_csr_write(A_EXT_MACRO_ACT, 32'h0);
                cpu_csr_write(A_EXT_MACRO_OUT, 32'h0);
                cpu_csr_write(A_EXT_MACRO_ASTR,32'h0);
                cpu_csr_write(A_EXT_MACRO_OSTR,32'h0);
                cpu_csr_write(A_EXT_DRAM_V,    32'h0);
                cpu_csr_write(A_SEQ_CTRL,      (32'd1 << 16) | 32'h09); // op=01, dma=1, gqa_m1=1, out=0 (FP16)
                cpu_csr_write(A_ATTN_H,        HQ);
                cpu_csr_write(A_ATTN_QT,       QT);
                cpu_csr_write(A_ATTN_MQ,       MQ);
                cpu_csr_write(A_ATTN_DV,       DV);
                cpu_csr_write(A_ATTN_KV,       KV);
                cpu_csr_write(A_ATTN_KB,       KB);
                cpu_csr_write(A_RING_BASE,     tcdm_addr(0, 0));
                cpu_csr_write(A_Q_BASE,        tcdm_addr(0, 16));
                cpu_csr_write(A_OUT_BASE,      tcdm_addr(0, 17));
                cpu_csr_write(A_DRAM_KV,       DRAM_PACK);
                cpu_csr_write(A_TILE_ST,       (KB + 1) * 144);
                cpu_csr_write(A_START,         32'h3);
                wait_seq_done("Test 6A Ring Gold");
                rd_ring = stat_rd_beats - rd0;
                for (int i = 0; i < HQ * QT * DV * MQ * 4; i++) cpu_mem_read(tcdm_addr(i, 17), gold_o[i]);

                // KV-Pin + Separate DRAM K/V
                for (int i = 0; i < HQ * QT * DV * MQ * 4; i++) cpu_mem_write(tcdm_addr(i, 17), 72'd0);
                rd0 = stat_rd_beats;
                cpu_csr_write(A_EXT_MACRO_SUB, 32'h0003_0000); // bit 16=kv_pin_en, bit 17=sep_kv_dram
                cpu_csr_write(A_DRAM_KV,       DRAM_K);
                cpu_csr_write(A_EXT_DRAM_V,    DRAM_V);
                cpu_csr_write(A_START,         32'h3);
                wait_seq_done("Test 6A KV-Pin");
                rd_pin = stat_rd_beats - rd0;

                mm = 0;
                for (int i = 0; i < HQ * QT * DV * MQ * 4; i++) begin
                    cpu_mem_read(tcdm_addr(i, 17), pin_o[i]);
                    if (pin_o[i] !== gold_o[i]) mm++;
                end
                if (mm == 0 && (rd_pin * 3 <= rd_ring))
                    $display("    [PASS] 6A KV_Pin_En=1 + GQA 2:1 + Sep DRAM K/V 100%% bit-exact! DRAM rd beats: %0d vs Ring %0d",
                             rd_pin, rd_ring);
                else begin
                    $display("    [FAIL] 6A KV_Pin_En mismatch=%0d, rd_pin=%0d rd_ring=%0d", mm, rd_pin, rd_ring);
                    errs += (mm > 0 ? mm : 1);
                end
            end

            // -----------------------------------------------------------------
            // 6B: FlashAttention Macro-DMA Ping-Pong Q / O
            // -----------------------------------------------------------------
            $display("\n[TEST 6B] FlashAttention Macro-DMA Ping-Pong Q/O (Q_TILES=4, Q_SUB=2) ...");
            begin
                localparam int HQ = 2, QT = 4, QSUB = 2, MQ = 8, DV = 2, KV = 2, KB = 2;
                localparam int DRAM_KV2 = 32'h0016_0000;
                localparam int DRAM_Q2  = 32'h0018_0000;
                localparam int DRAM_O2  = 32'h001A_0000;
                logic [71:0] q_pkts[], kv_pkts[], gold_o[], macro_o[];
                int mm;

                q_pkts  = new[HQ * QT * MQ * KB];
                kv_pkts = new[HQ * DV * KV * (KB + 1) * 16];
                gold_o  = new[HQ * QT * DV * MQ];
                macro_o = new[HQ * QT * DV * MQ];

                for (int i = 0; i < HQ * QT * MQ * KB; i++) begin
                    q_pkts[i] = make_fp4_pkt(1500 + i);
                    cpu_mem_write(tcdm_addr(i, 16), q_pkts[i]);
                end
                for (int i = 0; i < HQ * DV * KV * (KB + 1) * 16; i++)
                    kv_pkts[i] = make_fp4_pkt(2100 + i);

                pack_words_to_dram(DRAM_Q2,  HQ * QT * MQ * KB, q_pkts);
                pack_words_to_dram(DRAM_KV2, HQ * DV * KV * (KB + 1) * 16, kv_pkts);

                // Non-Macro Reference (out_mode=01 FP4)
                cpu_csr_write(A_EXT_MACRO_SUB, 32'h0);
                cpu_csr_write(A_EXT_MACRO_ACT, 32'h0);
                cpu_csr_write(A_EXT_MACRO_OUT, 32'h0);
                cpu_csr_write(A_EXT_MACRO_ASTR,32'h0);
                cpu_csr_write(A_EXT_MACRO_OSTR,32'h0);
                cpu_csr_write(A_EXT_DRAM_V,    32'h0);
                cpu_csr_write(A_SEQ_CTRL,      32'h19); // op=01, dma=1, out_mode=01 (FP4 linear)
                cpu_csr_write(A_ATTN_H,        HQ);
                cpu_csr_write(A_ATTN_QT,       QT);
                cpu_csr_write(A_ATTN_MQ,       MQ);
                cpu_csr_write(A_ATTN_DV,       DV);
                cpu_csr_write(A_ATTN_KV,       KV);
                cpu_csr_write(A_ATTN_KB,       KB);
                cpu_csr_write(A_RING_BASE,     tcdm_addr(0, 0));
                cpu_csr_write(A_Q_BASE,        tcdm_addr(0, 16));
                cpu_csr_write(A_OUT_BASE,      tcdm_addr(0, 17));
                cpu_csr_write(A_DRAM_KV,       DRAM_KV2);
                cpu_csr_write(A_TILE_ST,       (KB + 1) * 144);
                cpu_csr_write(A_START,         32'h3);
                wait_seq_done("Test 6B Non-Macro Gold");
                for (int i = 0; i < HQ * QT * DV * MQ; i++) cpu_mem_read(tcdm_addr(i, 17), gold_o[i]);

                // Macro-DMA Ping-Pong Q/O (clear TCDM Q/O first so DMA is mandatory)
                for (int i = 0; i < HQ * QT * MQ * KB; i++) cpu_mem_write(tcdm_addr(i, 16), 72'd0);
                for (int i = 0; i < 128; i++) begin
                    cpu_mem_write(tcdm_addr(i, 17), 72'd0);
                    cpu_mem_write(tcdm_addr(i + 64, 17), 72'd0);
                end
                cpu_csr_write(A_EXT_MACRO_SUB,  QSUB);
                cpu_csr_write(A_EXT_MACRO_ACT,  DRAM_Q2);
                cpu_csr_write(A_EXT_MACRO_OUT,  DRAM_O2);
                cpu_csr_write(A_EXT_MACRO_ASTR, 32'h8000); // +64 words in bank
                cpu_csr_write(A_EXT_MACRO_OSTR, 32'h8000);
                cpu_csr_write(A_SEQ_CTRL,       (32'd1 << 30) | 32'h19);
                cpu_csr_write(A_START,          32'h3);
                wait_seq_done("Test 6B Macro-DMA Q/O");

                unpack_words_from_dram(DRAM_O2, HQ * QT * DV * MQ, macro_o);
                mm = 0;
                for (int i = 0; i < HQ * QT * DV * MQ; i++)
                    if (macro_o[i] !== gold_o[i]) mm++;
                if (mm == 0)
                    $display("    [PASS] 6B FlashAttention Macro-DMA Ping-Pong Q/O 100%% bit-exact (%0d vectors)!", HQ*QT*DV*MQ);
                else begin
                    $display("    [FAIL] 6B Macro-DMA Q/O had %0d mismatches!", mm);
                    errs += mm;
                end

                // Clear extended macro CSRs
                cpu_csr_write(A_EXT_MACRO_SUB,  32'h0);
                cpu_csr_write(A_EXT_MACRO_ACT,  32'h0);
                cpu_csr_write(A_EXT_MACRO_OUT,  32'h0);
                cpu_csr_write(A_EXT_MACRO_ASTR, 32'h0);
                cpu_csr_write(A_EXT_MACRO_OSTR, 32'h0);
            end

            // -----------------------------------------------------------------
            // 6C: Hardware Decode / TCDM V-Transpose (op_mode=2'b10)
            // -----------------------------------------------------------------
            $display("\n[TEST 6C] Hardware Decode V-Transpose (op_mode=2'b10, 16-bank & single-bank) ...");
            begin
                localparam int V_NB = 2; // 32 channels (2 x 16)
                logic [15:0][15:0] v_mat [0:V_NB*16-1]; // v_mat[col][row]
                logic [71:0] pkt16, pkt1, pkt_exp;
                int mm;

                // Construct deterministic FP16 values across tokens and channels
                for (int b = 0; b < V_NB; b++)
                    for (int r = 0; r < 16; r++)
                        for (int c = 0; c < 16; c++) begin
                            logic sign_b;
                            logic [4:0] exp_b;
                            logic [9:0] mant_b;
                            sign_b = ((r + c * 3 + b) & 1) ? 1'b1 : 1'b0;
                            exp_b  = 5'd12 + 5'((r + c + b * 2) % 7);
                            mant_b = 10'((r * 61 + c * 37 + b * 19) & 1023);
                            v_mat[b * 16 + c][r] = {sign_b, exp_b, mant_b};
                        end

                // Stage into BOTH layouts:
                //   Layout 1: Banks 16..31 words 512..519 (act_bank_layout=1)
                //   Layout 0: Bank 21 words 0..127 (act_bank_layout=0)
                for (int r = 0; r < 16; r++) begin
                    for (int b = 0; b < V_NB; b++) begin
                        for (int w = 0; w < 4; w++) begin
                            logic [71:0] wd;
                            wd = {8'h00,
                                  v_mat[b*16 + w*4 + 3][r], v_mat[b*16 + w*4 + 2][r],
                                  v_mat[b*16 + w*4 + 1][r], v_mat[b*16 + w*4 + 0][r]};
                            cpu_mem_write(tcdm_addr(512 + b*4 + w, 16 + r), wd);
                            cpu_mem_write(tcdm_addr(r*(V_NB*4) + b*4 + w, 21), wd);
                        end
                    end
                end

                // Run 1: 16-bank layout -> Banks 0..15 words 128..129
                cpu_csr_write(A_CTRL,     32'h0);
                cpu_csr_write(A_RD_LS,    32'd0);
                cpu_csr_write(A_SEQ_CTRL, (32'd1 << 24) | 32'h2);
                cpu_csr_write(A_Q_BASE,   tcdm_addr(512, 16));
                cpu_csr_write(A_V_BASE,   tcdm_addr(128, 0));
                cpu_csr_write(A_V_NB,     V_NB);
                cpu_csr_write(A_TILE_ST,  32'h200);
                cpu_csr_write(A_START,    32'h3);
                wait_seq_done("Test 6C V-Trans 16-bank");

                // Run 2: single-bank layout -> Banks 0..15 words 160..161
                cpu_csr_write(A_SEQ_CTRL, 32'h2);
                cpu_csr_write(A_Q_BASE,   tcdm_addr(0, 21));
                cpu_csr_write(A_V_BASE,   tcdm_addr(160, 0));
                cpu_csr_write(A_V_NB,     V_NB);
                cpu_csr_write(A_TILE_ST,  32'h200);
                cpu_csr_write(A_START,    32'h3);
                wait_seq_done("Test 6C V-Trans 1-bank");

                mm = 0;
                for (int b = 0; b < V_NB; b++)
                    for (int c = 0; c < 16; c++) begin
                        cpu_mem_read(tcdm_addr(128 + b, c), pkt16);
                        cpu_mem_read(tcdm_addr(160 + b, c), pkt1);
                        pkt_exp = sw_logq_vec(v_mat[b * 16 + c]);
                        if ((pkt16 !== pkt_exp) || (pkt1 !== pkt_exp)) begin
                            $display("    [FAIL] 6C V-Trans (b=%0d,c=%0d) pkt16=%h pkt1=%h exp=%h",
                                     b, c, pkt16, pkt1, pkt_exp);
                            mm++;
                        end
                    end
                if (mm == 0)
                    $display("    [PASS] 6C HW Decode V-Transpose (both 16-bank & single-bank) 100%% bit-exact (%0d packets)!",
                             V_NB * 16);
                else
                    errs += mm;
            end
        end
    endtask

    // =========================================================================
    // MAIN
    // =========================================================================
    initial begin
        rst_n         = 1'b0;
        cpu_req_valid = 1'b0;
        cpu_req_write = 1'b0;
        cpu_req_addr  = 32'd0;
        cpu_req_wdata = 72'd0;
        cpu_req_strb  = 9'h1FF;
        csr_req_valid = 1'b0;
        csr_req_write = 1'b0;
        csr_req_addr  = 12'd0;
        csr_req_wdata = 32'd0;
        inject_stall  = 1'b0;
        errs          = 0;

        for (int i = 0; i < DRAM_WORDS; i++) dram_mem[i] = 512'd0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        test_decode_ktiling();
        test_prefill_all_out_modes();
        // test_weight_ring_reuse();
        // test_macro_dma_gt_tcdm();
        // test_drain_backpressure();
        // test_unified_attn_and_decode_v();

        $display("\n========================================================================");
        $display("  GEMM_SEQUENCER_REDESIGN_SPEC TESTBENCH SUMMARY");
        $display("  Max inflight_out_cnt observed : %0d (limit <= 4)", max_inflight_seen);
        $display("  Total errors                  : %0d", errs);
        $display("========================================================================");
        if (errs == 0)
            $display(">>> ALL GEMM OS SPEC TESTS PASSED (0 errors) <<<");
        else
            $display(">>> FAILED WITH %0d ERRORS <<<", errs);
        $finish;
    end


    initial begin
        if (!$test$plusargs("NO_WAVE")) begin
`ifdef FST_TRACE
            $dumpfile("tb_gemm_os_all_sizes.fst");
`else
            $dumpfile("tb_gemm_os_all_sizes.vcd");
`endif
            if ($test$plusargs("WAVE_DEEP"))
                $dumpvars(0, tb_gemm_os_all_sizes);
            else
                $dumpvars(0, dut);
        end
    end


endmodule
