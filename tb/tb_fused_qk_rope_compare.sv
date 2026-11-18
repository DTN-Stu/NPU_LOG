// =============================================================================
// Testbench: tb_fused_qk_rope_compare.sv
// Description: Verifies Offline W_Q Permutation + Direct Fused Path:
//                mac_array -> fused_fifo -> u_sfu (RoPE) -> u_log_quantizer -> FP4
//              vs. Baseline 3-stage path:
//                GEMM (FP16 TCDM) -> CPU repack/SFU (RoPE) -> Requant (FP4)
//              Compares BOTH cycle count AND numerical equivalence (SQNR, CosSim).
// =============================================================================
`timescale 1ns/1ps

module tb_fused_qk_rope_compare;

    localparam int M_TOKENS = 32;
    localparam int D_MODEL  = 128;
    localparam int N_HEADS  = 4;
    localparam int D_K      = 32;
    localparam int K_PKT    = D_MODEL / 16; // 8
    localparam int N_TILES  = D_MODEL / 16; // 8
    localparam int NUM_VEC  = M_TOKENS * N_TILES; // 256

    localparam int ACT_BANK      = 17;
    localparam int WT_BASE_WORD  = 0;
    localparam int WT_PERM_WORD  = 128;
    localparam int FP16_OUT_BANK = 18;
    localparam int ROPE_COS_BANK = 22;
    localparam int BASE_FP4_BANK = 24;
    localparam int FUSED_FP4_BANK= 25;
    localparam int ROPE_B_BANK   = 26;

    // CSR map
    localparam logic [11:0] A_CTRL       = 12'd0;
    localparam logic [11:0] A_WO_MODE    = 12'd12;
    localparam logic [11:0] A_WO_BASE    = 12'd13;
    localparam logic [11:0] A_WO_NV      = 12'd14;
    localparam logic [11:0] A_WO_LS      = 12'd15;
    localparam logic [11:0] A_RD_BASE    = 12'd27;
    localparam logic [11:0] A_RD_NV      = 12'd28;
    localparam logic [11:0] A_RD_LS      = 12'd29;
    localparam logic [11:0] A_SEQ_CTRL   = 12'd31;
    localparam logic [11:0] A_GEMM_NT    = 12'd38;
    localparam logic [11:0] A_GEMM_MT    = 12'd39;
    localparam logic [11:0] A_GEMM_KB    = 12'd40;
    localparam logic [11:0] A_RING_BASE  = 12'd41;
    localparam logic [11:0] A_Q_BASE     = 12'd42;
    localparam logic [11:0] A_OUT_BASE   = 12'd43;
    localparam logic [11:0] A_TILE_ST    = 12'd45;
    localparam logic [11:0] A_SFU_CTRL   = 12'd46;
    localparam logic [11:0] A_SFU_INV_N  = 12'd47;
    localparam logic [11:0] A_SFU_EPS    = 12'd48;
    localparam logic [11:0] A_SFU_B_BASE = 12'd49;
    localparam logic [11:0] A_SFU_B_NV   = 12'd50;
    localparam logic [11:0] A_SFU_B_LS   = 12'd51;
    localparam logic [11:0] A_START      = 12'd53;

    logic clk, rst_n;
    logic        csr_req_valid, csr_req_ready, csr_req_write;
    logic [11:0] csr_req_addr;
    logic [31:0] csr_req_data;
    logic        csr_rsp_valid, csr_rsp_ready;
    logic [31:0] csr_rsp_data;

    logic        cpu_mem_req_valid, cpu_mem_req_ready, cpu_mem_req_write;
    logic [31:0] cpu_mem_req_addr;
    logic [71:0] cpu_mem_req_wdata;
    logic [8:0]  cpu_mem_req_strb;
    logic        cpu_mem_rsp_valid;
    logic [71:0] cpu_mem_rsp_rdata;

    logic        dma_mem_req_valid, dma_mem_req_ready, dma_mem_req_write;
    logic [31:0] dma_mem_req_addr;
    logic [71:0] dma_mem_req_wdata;
    logic [8:0]  dma_mem_req_strb;
    logic        dma_mem_rsp_valid;
    logic [71:0] dma_mem_rsp_rdata;

    logic        dma_desc_valid, dma_desc_ready;
    logic [1:0]  dma_desc_dir;
    logic [31:0] dma_desc_dram_addr, dma_desc_tcdm_addr, dma_desc_size;
    logic        dma_desc_done;
    logic        subsystem_busy, subsystem_done_irq;

    snax_llm_top_clean #(
        .NUM_BANKS     (32),
        .BANK_DEPTH    (1024),
        .NUM_LANES     (16),
        .ACC_DEPTH     (16),
        .SFU_ALU_LANES (8)
    ) dut (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .cpu_csr_req_valid_i    (csr_req_valid),
        .cpu_csr_req_ready_o    (csr_req_ready),
        .cpu_csr_req_write_i    (csr_req_write),
        .cpu_csr_req_addr_i     (csr_req_addr),
        .cpu_csr_req_data_i     (csr_req_data),
        .cpu_csr_rsp_valid_o    (csr_rsp_valid),
        .cpu_csr_rsp_data_o     (csr_rsp_data),
        .cpu_csr_rsp_ready_i    (csr_rsp_ready),
        .cpu_mem_req_valid_i    (cpu_mem_req_valid),
        .cpu_mem_req_ready_o    (cpu_mem_req_ready),
        .cpu_mem_req_addr_i     (cpu_mem_req_addr),
        .cpu_mem_req_write_i    (cpu_mem_req_write),
        .cpu_mem_req_wdata_i    (cpu_mem_req_wdata),
        .cpu_mem_req_strb_i     (cpu_mem_req_strb),
        .cpu_mem_rsp_valid_o    (cpu_mem_rsp_valid),
        .cpu_mem_rsp_rdata_o    (cpu_mem_rsp_rdata),
        .dma_mem_req_valid_i    (dma_mem_req_valid),
        .dma_mem_req_ready_o    (dma_mem_req_ready),
        .dma_mem_req_addr_i     (dma_mem_req_addr),
        .dma_mem_req_write_i    (dma_mem_req_write),
        .dma_mem_req_wdata_i    (dma_mem_req_wdata),
        .dma_mem_req_strb_i     (dma_mem_req_strb),
        .dma_mem_rsp_valid_o    (dma_mem_rsp_valid),
        .dma_mem_rsp_rdata_o    (dma_mem_rsp_rdata),
        .dma_desc_valid_o       (dma_desc_valid),
        .dma_desc_ready_i       (dma_desc_ready),
        .dma_desc_dir_o         (dma_desc_dir),
        .dma_desc_dram_addr_o   (dma_desc_dram_addr),
        .dma_desc_tcdm_addr_o   (dma_desc_tcdm_addr),
        .dma_desc_size_o        (dma_desc_size),
        .dma_desc_done_i        (dma_desc_done),
        .subsystem_busy_o       (subsystem_busy),
        .subsystem_done_irq_o   (subsystem_done_irq)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    // Optional FST waveform dump (for Verilator --trace-fst / Icarus -fst / +dump / +fst)
    initial begin
        // if ($test$plusargs("dump") || $test$plusargs("DUMP") || $test$plusargs("fst")) begin
            $dumpfile("tb_fused_qk_rope_compare.fst");
            $dumpvars(0, tb_fused_qk_rope_compare);
        // end
    end

    int cyc_total;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) cyc_total <= 0;
        else        cyc_total <= cyc_total + 1;
    end

    function automatic logic [31:0] tcdm_addr(input int word, input int bank);
        return (word << 9) | (bank << 4);
    endfunction

    function automatic real fp16_to_real(input logic [15:0] h);
        logic s; int e, m; real v;
        begin
            s = h[15]; e = h[14:10]; m = h[9:0];
            if (e == 0) v = (m / 1024.0) * (2.0 ** -14);
            else        v = (1.0 + m / 1024.0) * (2.0 ** (e - 15));
            return s ? -v : v;
        end
    endfunction

    function automatic logic [15:0] real_to_fp16(input real r);
        logic s; real ar; int e, m;
        begin
            if (r == 0.0) return 16'h0000;
            s = (r < 0.0); ar = s ? -r : r;
            e = 15;
            while (ar >= 2.0 && e < 30) begin ar /= 2.0; e++; end
            while (ar <  1.0 && e > 1)  begin ar *= 2.0; e--; end
            m = int'((ar - 1.0) * 1024.0 + 0.5);
            if (m >= 1024) begin m = 0; e++; end
            return {s, e[4:0], m[9:0]};
        end
    endfunction

    function automatic real e2m1_mag(input logic [2:0] c);
        case (c)
            3'd0: return 0.0; 3'd1: return 0.5; 3'd2: return 1.0; 3'd3: return 1.5;
            3'd4: return 2.0; 3'd5: return 3.0; 3'd6: return 4.0; default: return 6.0;
        endcase
    endfunction

    function automatic real decode_pkt_elem(input logic [71:0] pkt, input int idx);
        logic [3:0] code;
        logic signed [7:0] sc;
        real mag;
        begin
            code = pkt[idx*4 +: 4];
            sc   = signed'(pkt[71:64]);
            mag  = e2m1_mag(code[2:0]);
            if (code[3]) mag = -mag;
            return mag * (2.0 ** (real'(sc) / 8.0));
        end
    endfunction

    // CSR / TCDM helpers
    task automatic cpu_csr_write(input logic [11:0] addr, input logic [31:0] data);
        begin
            @(negedge clk);
            csr_req_valid = 1'b1; csr_req_write = 1'b1;
            csr_req_addr  = addr; csr_req_data  = data;
            #1;
            while (!csr_req_ready) begin @(negedge clk); #1; end
            @(posedge clk); @(negedge clk);
            csr_req_valid = 1'b0; csr_req_write = 1'b0;
        end
    endtask

    task automatic cpu_mem_write(input logic [31:0] a, input logic [71:0] d);
        begin
            @(negedge clk);
            cpu_mem_req_valid = 1'b1; cpu_mem_req_write = 1'b1;
            cpu_mem_req_addr  = a; cpu_mem_req_wdata = d; cpu_mem_req_strb = 9'h1FF;
            #1;
            while (!cpu_mem_req_ready) begin @(negedge clk); #1; end
            @(posedge clk); @(negedge clk);
            cpu_mem_req_valid = 1'b0; cpu_mem_req_write = 1'b0; cpu_mem_req_strb = 9'h000;
        end
    endtask

    task automatic cpu_mem_read(input logic [31:0] a, output logic [71:0] d);
        begin
            @(negedge clk);
            cpu_mem_req_valid = 1'b1; cpu_mem_req_write = 1'b0;
            cpu_mem_req_addr  = a; cpu_mem_req_wdata = '0; cpu_mem_req_strb = 9'h000;
            #1;
            while (!cpu_mem_req_ready) begin @(negedge clk); #1; end
            @(posedge clk); @(negedge clk);
            cpu_mem_req_valid = 1'b0;
            d = cpu_mem_rsp_rdata;
        end
    endtask

    // Deterministic synthetic 72-bit FP4 packets (realistic [-16..+8] Log8 scale)
    function automatic logic [71:0] gen_fp4_pkt(input int seed);
        logic [71:0] p;
        int s, r;
        begin
            s = (seed * 1103515245 + 12345) & 32'h7FFF_FFFF;
            p[71:64] = 8'(-8 + (s % 16));
            for (int e = 0; e < 16; e++) begin
                s = (s * 1103515245 + 12345) & 32'h7FFF_FFFF;
                r = (s >> 16) & 7;
                p[e*4 +: 4] = {(s >> 20) & 1, r[2:0]};
            end
            return p;
        end
    endfunction

    // Build W_Q (original) and W_Q' (offline column-permuted per head: [j, j+D_K/2])
    logic [71:0] act_pkts [0:NUM_VEC-1];
    logic [71:0] wq_orig [0:N_TILES-1][0:K_PKT-1][0:15];
    logic [71:0] wq_perm [0:N_TILES-1][0:K_PKT-1][0:15];
    logic [15:0] cos_tbl [0:M_TOKENS-1][0:D_K/2-1];
    logic [15:0] sin_tbl [0:M_TOKENS-1][0:D_K/2-1];

    initial begin
        rst_n = 1'b0;
        csr_req_valid = 0; csr_req_write = 0; csr_req_addr = 0; csr_req_data = 0; csr_rsp_ready = 1;
        cpu_mem_req_valid = 0; cpu_mem_req_write = 0; cpu_mem_req_addr = 0; cpu_mem_req_wdata = 0; cpu_mem_req_strb = 0;
        dma_mem_req_valid = 0; dma_mem_req_write = 0; dma_mem_req_addr = 0; dma_mem_req_wdata = 0; dma_mem_req_strb = 0;
        dma_desc_ready = 1; dma_desc_done = 0;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (5) @(posedge clk);

        // 1. Initialize X, W_Q, W_Q' (column-permuted!), and RoPE cos/sin
        for (int t = 0; t < M_TOKENS; t++) begin
            for (int k = 0; k < K_PKT; k++) begin
                act_pkts[t*K_PKT + k] = gen_fp4_pkt(100 + t*K_PKT + k);
                cpu_mem_write(tcdm_addr(t*K_PKT + k, ACT_BANK), act_pkts[t*K_PKT + k]);
            end
            for (int j = 0; j < D_K/2; j++) begin
                real ang;
                ang = real'(t) * (0.1 + 0.03 * real'(j));
                cos_tbl[t][j] = real_to_fp16($cos(ang));
                sin_tbl[t][j] = real_to_fp16($sin(ang));
            end
        end

        for (int nt = 0; nt < N_TILES; nt++) begin
            for (int k = 0; k < K_PKT; k++) begin
                for (int l = 0; l < 16; l++) begin
                    wq_orig[nt][k][l] = gen_fp4_pkt(5000 + nt*128 + k*16 + l);
                    cpu_mem_write(tcdm_addr(WT_BASE_WORD + nt*K_PKT + k, l), wq_orig[nt][k][l]);
                end
            end
        end

        // Offline column permutation of W_Q per head (D_K = 32 = 2 tiles of 16):
        // Output channel idx in head h (0..31):
        //   even 2*j   -> original channel j
        //   odd  2*j+1 -> original channel j + 16
        for (int hh = 0; hh < N_HEADS; hh++) begin
            for (int j = 0; j < 16; j++) begin
                int src_tile_a, src_lane_a, src_tile_b, src_lane_b;
                int dst_tile_0, dst_lane_0, dst_tile_1, dst_lane_1;
                int ch_even, ch_odd;
                src_tile_a = hh*2 + 0; src_lane_a = j;
                src_tile_b = hh*2 + 1; src_lane_b = j;
                ch_even = 2*j;
                ch_odd  = 2*j + 1;
                dst_tile_0 = hh*2 + (ch_even / 16); dst_lane_0 = ch_even % 16;
                dst_tile_1 = hh*2 + (ch_odd  / 16); dst_lane_1 = ch_odd  % 16;
                for (int k = 0; k < K_PKT; k++) begin
                    wq_perm[dst_tile_0][k][dst_lane_0] = wq_orig[src_tile_a][k][src_lane_a];
                    wq_perm[dst_tile_1][k][dst_lane_1] = wq_orig[src_tile_b][k][src_lane_b];
                end
            end
        end
        for (int nt = 0; nt < N_TILES; nt++) begin
            for (int k = 0; k < K_PKT; k++) begin
                for (int l = 0; l < 16; l++) begin
                    cpu_mem_write(tcdm_addr(WT_PERM_WORD + nt*K_PKT + k, l), wq_perm[nt][k][l]);
                end
            end
        end

        // Write interleaved cos/sin for Method B (in tile-major order matching GEMM output: nt outer, t inner!)
        for (int nt = 0; nt < N_TILES; nt++) begin
            int half_idx;
            half_idx = (nt % 2) * 8; // 0 for even tile (j=0..7), 8 for odd tile (j=8..15)
            for (int t = 0; t < M_TOKENS; t++) begin
                int vec_idx;
                vec_idx = nt * M_TOKENS + t;
                for (int w = 0; w < 4; w++) begin
                    logic [71:0] pk;
                    int j0, j1;
                    j0 = half_idx + w*2;
                    j1 = half_idx + w*2 + 1;
                    pk = {8'd0, sin_tbl[t][j1], cos_tbl[t][j1], sin_tbl[t][j0], cos_tbl[t][j0]};
                    cpu_mem_write(tcdm_addr(vec_idx*4 + w, ROPE_B_BANK), pk);
                end
            end
        end

        // =====================================================================
        // METHOD A (Baseline): GEMM -> FP16 TCDM -> CPU Repack/SFU -> Requant
        // =====================================================================
        begin
            int t0, cyc_gemm, cyc_rope, cyc_req;
            int g;
            t0 = cyc_total;
            // 1. GEMM Q -> FP16_OUT_BANK
            cpu_csr_write(A_CTRL,     32'd0);
            cpu_csr_write(A_SEQ_CTRL, 32'h0000_0000); // OP_GEMM, out_mode=00 (FP16)
            cpu_csr_write(A_GEMM_NT,  N_TILES);
            cpu_csr_write(A_GEMM_MT,  M_TOKENS / 16);
            cpu_csr_write(A_GEMM_KB,  K_PKT);
            cpu_csr_write(A_Q_BASE,   tcdm_addr(0, ACT_BANK));
            cpu_csr_write(A_RING_BASE, tcdm_addr(WT_BASE_WORD, 0));
            cpu_csr_write(A_TILE_ST,  K_PKT << 9);
            cpu_csr_write(A_OUT_BASE, tcdm_addr(0, FP16_OUT_BANK));
            cpu_csr_write(A_START,    32'h0000_0003);
            repeat (4) @(posedge clk);
            while (dut.seq_busy || dut.wo_busy) @(posedge clk);
            cyc_gemm = cyc_total - t0;

            // 2. CPU repack + SFU RoPE per vector (exact same mechanism as tb_snax_llm_layer_mid)
            t0 = cyc_total;
            for (int t = 0; t < M_TOKENS; t++) begin
                for (int hh = 0; hh < N_HEADS; hh++) begin
                    logic [3:0][71:0] vwa, vwb;
                    logic [31:0][15:0] headv;
                    for (int w = 0; w < 4; w++) begin
                        cpu_mem_read(tcdm_addr(((hh*2+0)*M_TOKENS + t)*4 + w, FP16_OUT_BANK), vwa[w]);
                        cpu_mem_read(tcdm_addr(((hh*2+1)*M_TOKENS + t)*4 + w, FP16_OUT_BANK), vwb[w]);
                        for (int el = 0; el < 4; el++) begin
                            headv[4*w + el]      = vwa[w][16*el +: 16];
                            headv[16 + 4*w + el] = vwb[w][16*el +: 16];
                        end
                    end
                    for (int v = 0; v < 2; v++) begin
                        for (int w = 0; w < 4; w++) begin
                            logic [71:0] pka, pkb;
                            int j0, j1;
                            j0 = v*8 + w*2; j1 = v*8 + w*2 + 1;
                            pka = {8'd0, headv[16+j1], headv[j1], headv[16+j0], headv[j0]};
                            pkb = {8'd0, sin_tbl[t][j1], cos_tbl[t][j1], sin_tbl[t][j0], cos_tbl[t][j0]};
                            cpu_mem_write(tcdm_addr(w, 20), pka);
                            cpu_mem_write(tcdm_addr(w, 21), pkb);
                        end
                        cpu_csr_write(A_CTRL,       32'd1);
                        cpu_csr_write(A_SFU_CTRL,   (32'd1 << 5) | (32'd3 << 2) | 32'h3);
                        cpu_csr_write(A_RD_BASE,    tcdm_addr(0, 20));
                        cpu_csr_write(A_RD_NV,      32'd1);
                        cpu_csr_write(A_RD_LS,      32'h200);
                        cpu_csr_write(A_SFU_B_BASE, tcdm_addr(0, 21));
                        cpu_csr_write(A_SFU_B_NV,   32'd1);
                        cpu_csr_write(A_SFU_B_LS,   32'h200);
                        cpu_csr_write(A_WO_MODE,    32'd1);
                        cpu_csr_write(A_WO_BASE,    tcdm_addr(((hh*2+v)*M_TOKENS + t)*4, 19));
                        cpu_csr_write(A_WO_NV,      32'd1);
                        cpu_csr_write(A_WO_LS,      32'h200);
                        cpu_csr_write(A_START,      32'h0000_01D1);
                        repeat (4) @(posedge clk);
                        while (dut.sfu_busy || dut.wo_busy || dut.u_read_fp16.busy_o || dut.u_read_fp16_b.busy_o) @(posedge clk);
                        cpu_csr_write(A_SFU_CTRL,   32'd0);
                    end
                end
            end
            cyc_rope = cyc_total - t0;

            // 3. Requant FP16 -> FP4 (BASE_FP4_BANK)
            t0 = cyc_total;
            cpu_csr_write(A_CTRL,     32'd0);
            cpu_csr_write(A_SEQ_CTRL, 32'h0000_0013); // OP_READBACK, out_mode=01 (FP4)
            cpu_csr_write(A_RD_BASE,  tcdm_addr(0, 19));
            cpu_csr_write(A_RD_NV,    NUM_VEC);
            cpu_csr_write(A_RD_LS,    32'h200);
            cpu_csr_write(A_WO_BASE,  tcdm_addr(0, BASE_FP4_BANK));
            cpu_csr_write(A_START,    32'h0000_0003);
            repeat (4) @(posedge clk);
            while (dut.seq_busy || dut.wo_busy || dut.rd_busy) @(posedge clk);
            cyc_req = cyc_total - t0;

            $display("\n[METHOD A - Baseline] GEMM=%0d, CPU+SFU_RoPE=%0d, Requant=%0d  ==> TOTAL = %0d cycles",
                     cyc_gemm, cyc_rope, cyc_req, cyc_gemm + cyc_rope + cyc_req);
        end

        // =====================================================================
        // METHOD B (Fused): W_Q' (Offline Perm) + Core -> fused_fifo -> SFU(RoPE) -> logq -> FP4
        // =====================================================================
        begin
            int t0, cyc_fused;
            t0 = cyc_total;
            cpu_csr_write(A_CTRL,       32'd0);
            cpu_csr_write(A_SEQ_CTRL,   32'h0000_0010); // OP_GEMM (00), out_mode=01 (FP4 linear)
            cpu_csr_write(A_GEMM_NT,    N_TILES);
            cpu_csr_write(A_GEMM_MT,    M_TOKENS / 16);
            cpu_csr_write(A_GEMM_KB,    K_PKT);
            cpu_csr_write(A_Q_BASE,     tcdm_addr(0, ACT_BANK));
            cpu_csr_write(A_RING_BASE,  tcdm_addr(WT_PERM_WORD, 0));
            cpu_csr_write(A_TILE_ST,    K_PKT << 9);
            cpu_csr_write(A_OUT_BASE,   tcdm_addr(0, FUSED_FP4_BANK));
            // Enable SFU RoPE on direct fused path (in_a_src=1, out_dst=0 -> logq -> FP4)
            cpu_csr_write(A_SFU_CTRL,   32'h0020_0000 | (32'(NUM_VEC) << 5) | (32'd3 << 2) | 32'h1); // out_dst=0, en=1
            cpu_csr_write(A_SFU_B_BASE, tcdm_addr(0, ROPE_B_BANK));
            cpu_csr_write(A_SFU_B_NV,   NUM_VEC);
            cpu_csr_write(A_SFU_B_LS,   32'h200);
            // Launch u_seq (bit 1) + u_sfu (bit 7) + u_read_fp16_b (bit 8) simultaneously!
            cpu_csr_write(A_START,      32'h0000_0183);
            repeat (8) @(posedge clk);
            while (dut.seq_busy || dut.sfu_busy || dut.wo_busy || dut.rd_b_busy) @(posedge clk);
            cpu_csr_write(A_SFU_CTRL,   32'd0);
            cyc_fused = cyc_total - t0;

            $display("[METHOD B - Fused W'] Core->FIFO->SFU(RoPE)->LogQ->FP4        ==> TOTAL = %0d cycles\n",
                     cyc_fused);
        end

        // =====================================================================
        // Compare Numerical Equivalence of Final FP4 Packets (BASE_FP4_BANK vs FUSED_FP4_BANK)
        // =====================================================================
        begin
            int exact_pkt;
            real sum_a2, sum_err2, dot_ab, sum_b2, sqnr, cos_sim;
            exact_pkt = 0; sum_a2 = 0.0; sum_err2 = 0.0; dot_ab = 0.0; sum_b2 = 0.0;
            for (int v = 0; v < NUM_VEC; v++) begin
                logic [71:0] pa, pb;
                cpu_mem_read(tcdm_addr(v, BASE_FP4_BANK),  pa);
                cpu_mem_read(tcdm_addr(v, FUSED_FP4_BANK), pb);
                if (pa === pb) exact_pkt++;
                for (int e = 0; e < 16; e++) begin
                    real va, vb;
                    va = decode_pkt_elem(pa, e);
                    vb = decode_pkt_elem(pb, e);
                    sum_a2   += va * va;
                    sum_b2   += vb * vb;
                    sum_err2 += (va - vb) * (va - vb);
                    dot_ab   += va * vb;
                end
            end
            sqnr = (sum_err2 > 0.0) ? (10.0 * $log10(sum_a2 / sum_err2)) : 999.0;
            cos_sim = dot_ab / ($sqrt(sum_a2) * $sqrt(sum_b2));
            $display("[ACCURACY CHECK] Exact 72b FP4 Packet Match : %0d / %0d", exact_pkt, NUM_VEC);
            $display("[ACCURACY CHECK] Method B vs Method A       : SQNR = %0.2f dB, CosineSim = %0.6f\n",
                     sqnr, cos_sim);
        end

        $finish;
    end

endmodule
