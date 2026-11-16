// =============================================================================
// tb_snax_gemm_k1024.sv
//
// Single-tile Dense GEMM on snax_llm_top_clean:
//   M = 16 tokens, N = 16 channels, K = 1024 (K_BLOCKS = 64)
//
// This is the largest K that fits the current one-bank activation layout:
//   A: bank 16, word = token*64 + kb   (16 x 64 = 1024 words)
//   W: banks 0..15, word = kb          (64 words/bank)
//   C: bank 17, word = 4*token + wd    (64 words)
//
// Purpose: measure the compute vs drain balance at a realistic K, instead of
// the K=16 minimal layer test (which is output-bound).
//
// Data: ../tb/generate_snax_gemm_k1024_data.py -> sim_snax_gemm_k1024/hex
// =============================================================================
`timescale 1ns/1ps

module tb_snax_gemm_k1024;

    localparam int M_TOKENS   = 16;
    localparam int K_BLOCKS   = 64;
    localparam int NUM_VEC    = 16;
    localparam int NUM_WORD   = NUM_VEC * 4;        // 64
    localparam int NUM_FP16   = NUM_WORD * 4;       // 256
    localparam int NUM_PRELOAD= 2048;
    localparam int CLK_PERIOD = 10;

    localparam logic [31:0] Q_BASE   = 32'h0000_0100;   // bank 16, word 0
    localparam logic [31:0] OUT_BASE = 32'h0000_0110;   // bank 17, word 0
    localparam logic [31:0] RING     = 32'h0000_0000;   // bank 0,  word 0

    localparam logic [11:0] A_CTRL      = 12'd0;
    localparam logic [11:0] A_SEQ_CTRL  = 12'd31;
    localparam logic [11:0] A_GEMM_NT   = 12'd38;
    localparam logic [11:0] A_GEMM_MT   = 12'd39;
    localparam logic [11:0] A_GEMM_KB   = 12'd40;
    localparam logic [11:0] A_RING_BASE = 12'd41;
    localparam logic [11:0] A_Q_BASE    = 12'd42;
    localparam logic [11:0] A_OUT_BASE  = 12'd43;
    localparam logic [11:0] A_START     = 12'd47;
    localparam logic [11:0] A_RO_ID     = 12'd55;
    localparam logic [31:0] CSR_ID_VALUE= 32'h534E_4146;

    localparam int JOB_TIMEOUT    = 500000;
    localparam int HANDSHAKE_GUARD= 200000;

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
        #(20_000_000);
        $display("\n!!! GLOBAL TIMEOUT !!!\n");
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
            $dumpfile("tb_snax_gemm_k1024.fst");
`else
            $dumpfile("tb_snax_gemm_k1024.vcd");
`endif
            if ($test$plusargs("WAVE_DEEP"))
                $dumpvars(0, tb_snax_gemm_k1024);
            else
                $dumpvars(0, dut);
        end
    end

    logic [31:0] pre_addr [0:NUM_PRELOAD-1];
    logic [71:0] pre_data [0:NUM_PRELOAD-1];
    logic [71:0] exp_word [0:NUM_WORD-1];
    logic [71:0] out_word [0:NUM_WORD-1];

    int checks_pass, checks_fail;
    int stage_match, stage_mismatch;
    int i, w, e, guard;
    int job_cycles, cycle_cnt;
    int cyc_total, cyc_compute, cyc_drain, cyc_wo, cyc_launch, cyc_run, cyc_wait;
    bit seen_busy, seen_done;
    logic [31:0] rd32;
    logic [71:0] rw;
    logic [15:0] a16, e16;

    function automatic logic [31:0] tcdm_addr(input int word, input int bank);
        tcdm_addr = (32'(word) << 9) | (32'(bank) << 4);
    endfunction

    task automatic check(input bit cond, input string name);
        if (cond) checks_pass = checks_pass + 1;
        else begin
            checks_fail = checks_fail + 1;
            $display("    [FAIL] %s", name);
        end
    endtask

    task automatic fatal(input string msg);
        $display("\n!!! FATAL: %s !!!\n", msg);
        $finish;
    endtask

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

    task automatic dma_read(input logic [31:0] a, output logic [71:0] d);
        begin
            guard = 0;
            @(negedge clk);
            dma_mem_req_valid = 1'b1; dma_mem_req_write = 1'b0;
            dma_mem_req_addr = a; dma_mem_req_wdata = '0; dma_mem_req_strb = 9'h000;
            #1;
            while (!dma_mem_req_ready && guard < HANDSHAKE_GUARD) begin
                @(negedge clk); #1; guard = guard + 1;
            end
            if (!dma_mem_req_ready) fatal("DMA read ready timeout");
            @(posedge clk);
            @(negedge clk);
            dma_mem_req_valid = 1'b0;
            d = dma_mem_rsp_rdata;
        end
    endtask

    // Cycle-mix counters for the GEMM job.
    always @(posedge clk) begin
        if (rst_n) begin
            cyc_total = cyc_total + 1;
            if (|dut.lane_valid) cyc_compute = cyc_compute + 1;
            if (dut.tile_drain_req) cyc_drain = cyc_drain + 1;
            if (dut.wo_busy) cyc_wo = cyc_wo + 1;
            if (dut.u_seq.state_q == 5'd11) cyc_launch = cyc_launch + 1;
            if (dut.u_seq.state_q == 5'd12) cyc_run = cyc_run + 1;
            if (dut.u_seq.state_q == 5'd19) cyc_wait = cyc_wait + 1;
        end
    end

    initial begin
        csr_req_valid = 1'b0; csr_req_write = 1'b0; csr_req_addr = '0; csr_req_data = '0;
        csr_rsp_ready = 1'b1;
        cpu_mem_req_valid = 1'b0; cpu_mem_req_write = 1'b0; cpu_mem_req_addr = '0;
        cpu_mem_req_wdata = '0; cpu_mem_req_strb = '0;
        dma_mem_req_valid = 1'b0; dma_mem_req_write = 1'b0; dma_mem_req_addr = '0;
        dma_mem_req_wdata = '0; dma_mem_req_strb = '0;
        dma_desc_ready = 1'b1; dma_desc_done = 1'b0;

        checks_pass = 0; checks_fail = 0;
        cyc_total = 0; cyc_compute = 0; cyc_drain = 0; cyc_wo = 0;
        cyc_launch = 0; cyc_run = 0; cyc_wait = 0;

        $readmemh("sim_snax_gemm_k1024/hex/tcdm_preload_addr.hex", pre_addr);
        $readmemh("sim_snax_gemm_k1024/hex/tcdm_preload_data.hex", pre_data);
        $readmemh("sim_snax_gemm_k1024/hex/expected_out_words.hex", exp_word);

        wait (rst_n == 1'b1);
        repeat (5) @(posedge clk);

        $display("\n========================================================================");
        $display("  SNAX GEMM M=%0d N=%0d K=%0d (K_BLOCKS=%0d)", M_TOKENS, 16, K_BLOCKS*16, K_BLOCKS);
        $display("========================================================================\n");

        $display("[1] DMA preload (%0d words) ...", NUM_PRELOAD);
        for (i = 0; i < NUM_PRELOAD; i++)
            if (pre_addr[i] !== 32'bx)
                dma_write(pre_addr[i], pre_data[i]);

        cpu_csr_read(A_RO_ID, rd32);
        check(rd32 == CSR_ID_VALUE, "CSR ID (SNAF)");

        $display("[2] Program GEMM descriptor and start ...");
        cpu_csr_write(A_CTRL,      32'h0);
        cpu_csr_write(A_SEQ_CTRL,  32'h0);           // op 00 GEMM, FP16 out
        cpu_csr_write(A_GEMM_NT,   32'd1);
        cpu_csr_write(A_GEMM_MT,   32'd1);
        cpu_csr_write(A_GEMM_KB,   K_BLOCKS);
        cpu_csr_write(A_RING_BASE, RING);
        cpu_csr_write(A_Q_BASE,    Q_BASE);
        cpu_csr_write(A_OUT_BASE,  OUT_BASE);
        cpu_csr_write(A_START,     32'h0000_0003);

        seen_busy = 1'b0; seen_done = 1'b0; cycle_cnt = 0;
        while (!seen_done && cycle_cnt < JOB_TIMEOUT) begin
            @(posedge clk);
            cycle_cnt = cycle_cnt + 1;
            if (busy) seen_busy = 1'b1;
            if (seen_busy && done_irq && !busy) seen_done = 1'b1;
        end
        job_cycles = cycle_cnt;
        check(seen_done, "GEMM job finished");
        $display("    Completed in %0d cycles (busy=%0b done=%0b)", cycle_cnt, busy, done_irq);

        $display("[3] Read back %0d words from bank 17 ...", NUM_WORD);
        for (i = 0; i < NUM_WORD; i++)
            dma_read(tcdm_addr(i, 17), out_word[i]);

        stage_match = 0; stage_mismatch = 0;
        for (i = 0; i < NUM_WORD; i++) begin
            for (e = 0; e < 4; e++) begin
                a16 = out_word[i][16*e +: 16];
                e16 = exp_word[i][16*e +: 16];
                if (a16 === e16) stage_match++;
                else begin
                    stage_mismatch++;
                    if (stage_mismatch <= 8)
                        $display("    mismatch word=%0d lane=%0d act=0x%04X exp=0x%04X",
                                 i, e, a16, e16);
                end
            end
        end
        check(stage_mismatch == 0, "GEMM output bit-exact");

        $display("\n========================================================================");
        $display("  GEMM K=%0d RESULT", K_BLOCKS*16);
        $display("========================================================================");
        $display("  FP16 match        : %0d / %0d", stage_match, NUM_FP16);
        $display("  Job cycles        : %0d", job_cycles);
        $display("  Cycle mix         : compute=%0d drain_req=%0d wo_busy=%0d launch=%0d run=%0d wait=%0d",
                 cyc_compute, cyc_drain, cyc_wo, cyc_launch, cyc_run, cyc_wait);
        $display("  Drain share       : %0d.%02d%% of job (word writes 64 / (K+64))",
                 (cyc_wait * 100) / job_cycles,
                 ((cyc_wait * 10000) / job_cycles) % 100);
        $display("  Checks            : %0d passed, %0d failed", checks_pass, checks_fail);
        $display("========================================================================\n");

        if (checks_fail == 0 && stage_mismatch == 0)
            $display(">>> GEMM K=%0d PASS: bit-exact <<<\n", K_BLOCKS*16);
        else
            $display(">>> GEMM K=%0d FAIL <<<\n", K_BLOCKS*16);

        $finish;
    end

endmodule
