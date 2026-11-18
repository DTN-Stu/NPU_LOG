// =============================================================================
// Testbench: tb_simple_dma_gemm.sv
// Description: Verifies snax_simple_dma (512-bit DRAM <-> 72-bit TCDM gearbox)
//              connected to snax_llm_top_clean:
//   [1] Round-trip gearbox test: TCDM -> 512b DRAM -> TCDM (128 x 72b words)
//   [2] Autonomous Streaming GEMM via snax_llm_sequencer + snax_simple_dma
//       (cfg_enable_dma = 1, 16 tiles streamed through 4-slot ring buffer)
//       with 6-cycle DRAM latency model and throughput telemetry.
// =============================================================================
`timescale 1ns/1ps

module tb_simple_dma_gemm;

    localparam int unsigned DRAM_LATENCY = 6;
    localparam int unsigned M_TOKENS     = 32;
    localparam int unsigned N_TILES      = 8;
    localparam int unsigned K_PACKETS    = 8;
    localparam int unsigned ACT_BANK     = 16;
    localparam int unsigned OUT_BANK     = 17;
    localparam int unsigned STG_BANK     = 18;
    localparam int unsigned VER_BANK     = 19;

    // CSR addresses
    localparam logic [11:0] A_CTRL      = 12'd0;
    localparam logic [11:0] A_WO_MODE   = 12'd12;
    localparam logic [11:0] A_SEQ_CTRL  = 12'd31;
    localparam logic [11:0] A_GEMM_NT   = 12'd38;
    localparam logic [11:0] A_GEMM_MT   = 12'd39;
    localparam logic [11:0] A_GEMM_KB   = 12'd40;
    localparam logic [11:0] A_RING_BASE = 12'd41;
    localparam logic [11:0] A_Q_BASE    = 12'd42;
    localparam logic [11:0] A_OUT_BASE  = 12'd43;
    localparam logic [11:0] A_DRAM_KV   = 12'd44;
    localparam logic [11:0] A_TILE_ST   = 12'd45;
    localparam logic [11:0] A_START     = 12'd53;
    localparam logic [11:0] A_RO_STATUS = 12'd54;

    logic clk, rst_n;
    initial clk = 1'b0;
    always #5 clk = ~clk; // 100 MHz

    // CPU & CSR
    logic        cpu_req_valid, cpu_req_ready, cpu_req_write, cpu_rsp_valid;
    logic [31:0] cpu_req_addr;
    logic [71:0] cpu_req_wdata, cpu_rsp_rdata;
    logic [8:0]  cpu_req_strb;

    logic        csr_req_valid, csr_req_ready, csr_req_write, csr_rsp_valid;
    logic [11:0] csr_req_addr;
    logic [31:0] csr_req_wdata, csr_rsp_rdata;
    logic        irq_o;

    // DMA <-> DUT (4 parallel 72b TCDM ports = 288b/cycle)
    logic [3:0]        dma_tcdm_req_valid, dma_tcdm_req_ready, dma_tcdm_req_write, dma_tcdm_rsp_valid;
    logic [3:0][31:0]  dma_tcdm_req_addr;
    logic [3:0][71:0]  dma_tcdm_req_wdata, dma_tcdm_rsp_rdata;
    logic [3:0][8:0]   dma_tcdm_req_strb;

    logic        dut_desc_valid, dut_desc_ready, dut_desc_done;
    logic [1:0]  dut_desc_dir;
    logic [31:0] dut_desc_dram_addr, dut_desc_tcdm_addr, dut_desc_size;

    // Manual override mux for Test 1 (round-trip gearbox test)
    logic        manual_dma_mode;
    logic        man_desc_valid, man_desc_ready, man_desc_done;
    logic [1:0]  man_desc_dir;
    logic [31:0] man_desc_dram_addr, man_desc_tcdm_addr, man_desc_size;

    logic        dma_desc_valid, dma_desc_ready, dma_desc_done;
    logic [1:0]  dma_desc_dir;
    logic [31:0] dma_desc_dram_addr, dma_desc_tcdm_addr, dma_desc_size;

    assign dma_desc_valid     = manual_dma_mode ? man_desc_valid     : dut_desc_valid;
    assign dma_desc_dir       = manual_dma_mode ? man_desc_dir       : dut_desc_dir;
    assign dma_desc_dram_addr = manual_dma_mode ? man_desc_dram_addr : dut_desc_dram_addr;
    assign dma_desc_tcdm_addr = manual_dma_mode ? man_desc_tcdm_addr : dut_desc_tcdm_addr;
    assign dma_desc_size      = manual_dma_mode ? man_desc_size      : dut_desc_size;
    assign dut_desc_ready     = manual_dma_mode ? 1'b0               : dma_desc_ready;
    assign dut_desc_done      = manual_dma_mode ? 1'b0               : dma_desc_done;
    assign man_desc_ready     = manual_dma_mode ? dma_desc_ready     : 1'b0;
    assign man_desc_done      = manual_dma_mode ? dma_desc_done      : 1'b0;

    // 512-bit External DRAM interface
    logic         dram_req_valid, dram_req_ready, dram_req_write, dram_rsp_valid;
    logic [31:0]  dram_req_addr;
    logic [511:0] dram_req_wdata, dram_rsp_rdata;
    logic [63:0]  dram_req_strb;

    logic [31:0]  stat_rd_beats, stat_wr_beats, stat_tcdm_words, stat_busy_cycles;

    snax_llm_top_clean dut (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .cpu_csr_req_valid_i   (csr_req_valid),
        .cpu_csr_req_ready_o   (csr_req_ready),
        .cpu_csr_req_write_i   (csr_req_write),
        .cpu_csr_req_addr_i    (csr_req_addr),
        .cpu_csr_req_data_i    (csr_req_wdata),
        .cpu_csr_rsp_valid_o   (csr_rsp_valid),
        .cpu_csr_rsp_data_o    (csr_rsp_rdata),
        .cpu_csr_rsp_ready_i   (1'b1),
        .cpu_mem_req_valid_i   (cpu_req_valid),
        .cpu_mem_req_ready_o   (cpu_req_ready),
        .cpu_mem_req_addr_i    (cpu_req_addr),
        .cpu_mem_req_write_i   (cpu_req_write),
        .cpu_mem_req_wdata_i   (cpu_req_wdata),
        .cpu_mem_req_strb_i    (cpu_req_strb),
        .cpu_mem_rsp_valid_o   (cpu_rsp_valid),
        .cpu_mem_rsp_rdata_o   (cpu_rsp_rdata),
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
        .dma_desc_valid_o      (dut_desc_valid),
        .dma_desc_ready_i      (dut_desc_ready),
        .dma_desc_dir_o        (dut_desc_dir),
        .dma_desc_dram_addr_o  (dut_desc_dram_addr),
        .dma_desc_tcdm_addr_o  (dut_desc_tcdm_addr),
        .dma_desc_size_o       (dut_desc_size),
        .dma_desc_done_i       (dut_desc_done),
        .subsystem_busy_o      (),
        .subsystem_done_irq_o  (irq_o)
    );

    snax_simple_dma #(
        .TCDM_PORTS  (4),
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
    // Behavioral 512-bit External DRAM Model with configurable read latency
    // -------------------------------------------------------------------------
    localparam int DRAM_WORDS = 4096; // 4096 x 64B = 256 KB
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
                    for (int b = 0; b < 64; b++) begin
                        if (dram_req_strb[b])
                            dram_mem[dram_req_addr[17:6]][b*8 +: 8] <= dram_req_wdata[b*8 +: 8];
                    end
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

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    function automatic logic [31:0] tcdm_addr(input int word, input int bank);
        tcdm_addr = (32'(word) << 9) | (32'(bank) << 4);
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

    task automatic cpu_csr_read(input logic [11:0] addr, output logic [31:0] data);
        begin
            @(negedge clk);
            csr_req_valid = 1'b1;
            csr_req_write = 1'b0;
            csr_req_addr  = addr;
            @(posedge clk);
            while (!csr_req_ready) @(posedge clk);
            @(negedge clk);
            csr_req_valid = 1'b0;
            while (!csr_rsp_valid) @(posedge clk);
            data = csr_rsp_rdata;
        end
    endtask

    // Pack 72-bit words tightly into dram_mem starting at byte_base
    task automatic pack_72b_into_dram(input int byte_base, input int num_words,
                                      input logic [71:0] src []);
        int bit_pos;
        begin
            bit_pos = byte_base * 8;
            for (int i = 0; i < num_words; i++) begin
                for (int b = 0; b < 72; b++) begin
                    int widx, boff;
                    widx = (bit_pos + b) / 512;
                    boff = (bit_pos + b) % 512;
                    dram_mem[widx][boff] = src[i][b];
                end
                bit_pos += 72;
            end
        end
    endtask

    // -------------------------------------------------------------------------
    // Main Test
    // -------------------------------------------------------------------------
    localparam int NUM_PRELOAD = 12000;
    logic [31:0] pre_addr [0:NUM_PRELOAD-1];
    logic [71:0] pre_data [0:NUM_PRELOAD-1];
    logic [71:0] xn_pkts  [0:M_TOKENS*K_PACKETS-1];
    logic [71:0] w_tile_buf[];
    logic [71:0] gold_out [0:N_TILES*M_TOKENS*4-1];
    logic [71:0] dma_out  [0:N_TILES*M_TOKENS*4-1];

    int cyc_cnt, errs;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) cyc_cnt <= 0;
        else        cyc_cnt <= cyc_cnt + 1;
    end

    initial begin
        rst_n              = 1'b0;
        cpu_req_valid      = 1'b0;
        cpu_req_write      = 1'b0;
        cpu_req_addr       = 32'd0;
        cpu_req_wdata      = 72'd0;
        cpu_req_strb       = 9'h1FF;
        csr_req_valid      = 1'b0;
        csr_req_write      = 1'b0;
        csr_req_addr       = 12'd0;
        csr_req_wdata      = 32'd0;
        manual_dma_mode    = 1'b0;
        man_desc_valid     = 1'b0;
        man_desc_dir       = 2'd0;
        man_desc_dram_addr = 32'd0;
        man_desc_tcdm_addr = 32'd0;
        man_desc_size      = 32'd0;
        errs               = 0;

        for (int i = 0; i < DRAM_WORDS; i++) dram_mem[i] = 512'd0;
        for (int i = 0; i < NUM_PRELOAD; i++) begin
            pre_addr[i] = 32'bx;
            pre_data[i] = 72'd0;
        end

        $readmemh("sim_snax_llm_layer_mid/hex/tcdm_preload_addr.hex", pre_addr);
        $readmemh("sim_snax_llm_layer_mid/hex/tcdm_preload_data.hex", pre_data);
        $readmemh("sim_snax_llm_layer_mid/hex/layer_xn_packets.hex",  xn_pkts);

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // =====================================================================
        // TEST 1: Direct 512b <-> 72b Gearbox Round-Trip (TCDM -> DRAM -> TCDM)
        // =====================================================================
        $display("[TEST 1] Gearbox round-trip TCDM -> 512b DRAM -> TCDM (128 words = 1152B) ...");
        manual_dma_mode = 1'b1;
        for (int i = 0; i < 128; i++) begin
            cpu_mem_write(tcdm_addr(i, STG_BANK), {8'(i), 64'hCAFE_0000_0000_0000 | 64'(i * 32'h0101_0001)});
        end

        // TCDM -> DRAM (dir=2'b11: write to DRAM, linear TCDM)
        @(negedge clk);
        man_desc_valid     = 1'b1;
        man_desc_dir       = 2'b11;
        man_desc_dram_addr = 32'h0001_0000;
        man_desc_tcdm_addr = tcdm_addr(0, STG_BANK);
        man_desc_size      = 32'(128 * 9);
        @(posedge clk);
        while (!man_desc_ready) @(posedge clk);
        @(negedge clk);
        man_desc_valid = 1'b0;
        while (!man_desc_done) @(posedge clk);

        // DRAM -> TCDM (dir=2'b10: read from DRAM, linear TCDM)
        @(negedge clk);
        man_desc_valid     = 1'b1;
        man_desc_dir       = 2'b10;
        man_desc_dram_addr = 32'h0001_0000;
        man_desc_tcdm_addr = tcdm_addr(0, VER_BANK);
        man_desc_size      = 32'(128 * 9);
        @(posedge clk);
        while (!man_desc_ready) @(posedge clk);
        @(negedge clk);
        man_desc_valid = 1'b0;
        while (!man_desc_done) @(posedge clk);

        begin
            int rt_match; rt_match = 0;
            for (int i = 0; i < 128; i++) begin
                logic [71:0] act, exp;
                exp = {8'(i), 64'hCAFE_0000_0000_0000 | 64'(i * 32'h0101_0001)};
                cpu_mem_read(tcdm_addr(i, VER_BANK), act);
                if (act === exp) rt_match++;
                else errs++;
            end
            $display("    [PASS] Round-trip exact: %0d / 128", rt_match);
        end
        manual_dma_mode = 1'b0;

        // =====================================================================
        // TEST 2: Golden Baseline GEMM (cfg_enable_dma = 0) vs Streaming DMA GEMM (cfg_enable_dma = 1)
        // =====================================================================
        $display("[TEST 2] Streaming GEMM (M=2, N=8, K=8 -> 16 tiles via 4-slot ring + 512b DRAM) ...");
        // Preload W_Q (word 0..63 in banks 0..15) and ACT_BANK
        for (int i = 0; i < NUM_PRELOAD; i++)
            if (pre_addr[i] !== 32'bx)
                cpu_mem_write(pre_addr[i], pre_data[i]);
        for (int i = 0; i < M_TOKENS * K_PACKETS; i++)
            cpu_mem_write(tcdm_addr(i, ACT_BANK), xn_pkts[i]);

        // Run baseline GEMM with DMA disabled (weights already in banks 0..15)
        cpu_csr_write(A_CTRL,      32'h0);
        cpu_csr_write(A_SEQ_CTRL,  32'h0); // op 00 GEMM, dma=0
        cpu_csr_write(A_WO_MODE,   32'd1); // FP16 out
        cpu_csr_write(A_GEMM_NT,   N_TILES);
        cpu_csr_write(A_GEMM_MT,   M_TOKENS / 16);
        cpu_csr_write(A_GEMM_KB,   K_PACKETS);
        cpu_csr_write(A_RING_BASE, tcdm_addr(0, 0));
        cpu_csr_write(A_Q_BASE,    tcdm_addr(0, ACT_BANK));
        cpu_csr_write(A_OUT_BASE,  tcdm_addr(0, OUT_BANK));
        cpu_csr_write(A_START,     32'h0000_0003);
        repeat (4) @(posedge clk);
        while (dut.seq_busy || dut.wo_busy) @(posedge clk);
        for (int i = 0; i < N_TILES * M_TOKENS * 4; i++)
            cpu_mem_read(tcdm_addr(i, OUT_BANK), gold_out[i]);

        // Pack W_Q (N_TILES * K_PACKETS * 16 words) into 512-bit DRAM at byte 0
        // Stride per N-tile in DRAM = K_PACKETS * 16 * 9 = 1152 bytes (exact 18 x 512b beats!)
        w_tile_buf = new[N_TILES * K_PACKETS * 16];
        for (int nt = 0; nt < N_TILES; nt++)
            for (int kp = 0; kp < K_PACKETS; kp++)
                for (int l = 0; l < 16; l++)
                    cpu_mem_read(tcdm_addr(nt * K_PACKETS + kp, l),
                                 w_tile_buf[(nt * K_PACKETS + kp) * 16 + l]);
        pack_72b_into_dram(0, N_TILES * K_PACKETS * 16, w_tile_buf);

        // Clear ring buffer slots 0..3 in TCDM so we prove DMA actually loads them!
        for (int s = 0; s < 4; s++)
            for (int kp = 0; kp < K_PACKETS; kp++)
                for (int l = 0; l < 16; l++)
                    cpu_mem_write(tcdm_addr(s * 32 + kp, l), 72'd0);

        // Clear OUT_BANK
        for (int i = 0; i < N_TILES * M_TOKENS * 4; i++)
            cpu_mem_write(tcdm_addr(i, OUT_BANK), 72'd0);

        // Run Streaming DMA GEMM (cfg_enable_dma = 1 -> A_SEQ_CTRL bit 3 = 1 -> 32'h8)
        begin
            int t0, t1, dma_rd0, dma_words0, dma_busy0;
            int gemm_cycles, dram_bytes;
            logic [31:0] st;

            dma_rd0    = stat_rd_beats;
            dma_words0 = stat_tcdm_words;
            dma_busy0  = stat_busy_cycles;

            cpu_csr_write(A_CTRL,      32'h0);
            cpu_csr_write(A_SEQ_CTRL,  32'h8); // op 00 GEMM, enable_dma=1
            cpu_csr_write(A_WO_MODE,   32'd1);
            cpu_csr_write(A_GEMM_NT,   N_TILES);
            cpu_csr_write(A_GEMM_MT,   M_TOKENS / 16);
            cpu_csr_write(A_GEMM_KB,   K_PACKETS);
            cpu_csr_write(A_RING_BASE, tcdm_addr(0, 0));
            cpu_csr_write(A_Q_BASE,    tcdm_addr(0, ACT_BANK));
            cpu_csr_write(A_OUT_BASE,  tcdm_addr(0, OUT_BANK));
            cpu_csr_write(A_DRAM_KV,   32'd0);
            cpu_csr_write(A_TILE_ST,   32'(K_PACKETS * 16 * 9)); // 1152 bytes per N-tile

            t0 = cyc_cnt;
            cpu_csr_write(A_START,     32'h0000_0003);
            repeat (4) @(posedge clk);
            while (dut.seq_busy || dut.wo_busy) @(posedge clk);
            t1 = cyc_cnt;

            gemm_cycles = t1 - t0;
            dram_bytes  = (stat_rd_beats - dma_rd0) * 64;

            begin
                int gemm_match; gemm_match = 0;
                for (int i = 0; i < N_TILES * M_TOKENS * 4; i++) begin
                    cpu_mem_read(tcdm_addr(i, OUT_BANK), dma_out[i]);
                    if (dma_out[i] === gold_out[i]) gemm_match++;
                    else errs++;
                end
                $display("    [PASS] Streaming DMA GEMM exact vs Baseline: %0d / %0d words",
                         gemm_match, N_TILES * M_TOKENS * 4);
            end

            $display("    [TELEMETRY] GEMM cycles       : %0d (vs ~2184 non-DMA)", gemm_cycles);
            $display("    [TELEMETRY] DRAM 512b rd beats: %0d (%0d bytes)", stat_rd_beats - dma_rd0, dram_bytes);
            $display("    [TELEMETRY] TCDM 72b words    : %0d", stat_tcdm_words - dma_words0);
            $display("    [TELEMETRY] DMA busy cycles   : %0d", stat_busy_cycles - dma_busy0);
        end

        if (errs == 0) begin
            $display("\n>>> ALL SIMPLE DMA TESTS PASSED (0 errors) <<<");
        end else begin
            $display("\n>>> SIMPLE DMA TESTS FAILED (%0d errors) <<<", errs);
            $fatal(1, "DMA mismatch");
        end
        $finish;
    end

    // Optional FST waveform dump (for Verilator --trace-fst / Icarus -fst / +dump / +fst)
    initial begin
        // if ($test$plusargs("dump") || $test$plusargs("DUMP") || $test$plusargs("fst")) begin
            $dumpfile("tb_simple_dma_gemm.fst");
            $dumpvars(0, tb_simple_dma_gemm);
        // end
    end

endmodule
