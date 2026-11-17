// =============================================================================
// tb_snax_sfu_system.sv
// System-level smoke TB: drives snax_llm_sfu_opt through snax_llm_top_clean
// TCDM/read-stream/CSR paths.  Verifies Residual, RoPE, SwiGLU, RMSNorm.
// =============================================================================
`timescale 1ns/1ps

module tb_snax_sfu_system;
    localparam int D_MODEL = 16;
    localparam int N_VEC   = 1;

    logic clk = 1'b0;
    logic rst_n;
    always #5 clk = ~clk;

    // CPU CSR
    logic        cpu_csr_req_valid, cpu_csr_req_ready;
    logic        cpu_csr_req_write;
    logic [11:0] cpu_csr_req_addr;
    logic [31:0] cpu_csr_req_data;
    logic        cpu_csr_rsp_valid, cpu_csr_rsp_ready;
    logic [31:0] cpu_csr_rsp_data;

    // CPU TCDM
    logic        cpu_mem_req_valid, cpu_mem_req_ready;
    logic [31:0] cpu_mem_req_addr;
    logic        cpu_mem_req_write;
    logic [71:0] cpu_mem_req_wdata;
    logic [8:0]  cpu_mem_req_strb;
    logic        cpu_mem_rsp_valid;
    logic [71:0] cpu_mem_rsp_rdata;

    // DMA tied off
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
    logic        subsystem_busy, subsystem_done_irq;

    // CSR map (must match snax_flash_csr_snax.sv)
    localparam logic [11:0] A_CTRL      = 12'd0;
    localparam logic [11:0] A_WO_MODE   = 12'd12;
    localparam logic [11:0] A_WO_BASE   = 12'd13;
    localparam logic [11:0] A_WO_NV     = 12'd14;
    localparam logic [11:0] A_WO_LS     = 12'd15;
    localparam logic [11:0] A_RD_BASE   = 12'd27;
    localparam logic [11:0] A_RD_NV     = 12'd28;
    localparam logic [11:0] A_RD_LS     = 12'd29;
    localparam logic [11:0] A_SEQ_CTRL  = 12'd31;
    localparam logic [11:0] A_GEMM_NT   = 12'd38;
    localparam logic [11:0] A_GEMM_MT   = 12'd39;
    localparam logic [11:0] A_GEMM_KB   = 12'd40;
    localparam logic [11:0] A_RING_BASE = 12'd41;
    localparam logic [11:0] A_Q_BASE    = 12'd42;
    localparam logic [11:0] A_OUT_BASE  = 12'd43;
    localparam logic [11:0] A_SFU_CTRL  = 12'd46;
    localparam logic [11:0] A_SFU_INV_N = 12'd47;
    localparam logic [11:0] A_SFU_EPS   = 12'd48;
    localparam logic [11:0] A_SFU_B_BASE= 12'd49;
    localparam logic [11:0] A_SFU_B_NV  = 12'd50;
    localparam logic [11:0] A_SFU_B_LS  = 12'd51;
    localparam logic [11:0] A_START     = 12'd53;
    localparam logic [11:0] A_RO_ID     = 12'd61;
    localparam logic [31:0] CSR_ID_VALUE= 32'h534E_4146;

    localparam int A_BANK = 1;
    localparam int B_BANK = 2;
    localparam int O_BANK = 3;

    snax_llm_top_clean dut (
        .clk                        (clk),
        .rst_n                      (rst_n),
        .cpu_csr_req_valid_i        (cpu_csr_req_valid),
        .cpu_csr_req_ready_o        (cpu_csr_req_ready),
        .cpu_csr_req_write_i        (cpu_csr_req_write),
        .cpu_csr_req_addr_i         (cpu_csr_req_addr),
        .cpu_csr_req_data_i         (cpu_csr_req_data),
        .cpu_csr_rsp_valid_o        (cpu_csr_rsp_valid),
        .cpu_csr_rsp_data_o         (cpu_csr_rsp_data),
        .cpu_csr_rsp_ready_i        (cpu_csr_rsp_ready),
        .cpu_mem_req_valid_i        (cpu_mem_req_valid),
        .cpu_mem_req_ready_o        (cpu_mem_req_ready),
        .cpu_mem_req_addr_i         (cpu_mem_req_addr),
        .cpu_mem_req_write_i        (cpu_mem_req_write),
        .cpu_mem_req_wdata_i        (cpu_mem_req_wdata),
        .cpu_mem_req_strb_i         (cpu_mem_req_strb),
        .cpu_mem_rsp_valid_o        (cpu_mem_rsp_valid),
        .cpu_mem_rsp_rdata_o        (cpu_mem_rsp_rdata),
        .dma_mem_req_valid_i        (dma_mem_req_valid),
        .dma_mem_req_ready_o        (dma_mem_req_ready),
        .dma_mem_req_addr_i         (dma_mem_req_addr),
        .dma_mem_req_write_i        (dma_mem_req_write),
        .dma_mem_req_wdata_i        (dma_mem_req_wdata),
        .dma_mem_req_strb_i         (dma_mem_req_strb),
        .dma_mem_rsp_valid_o        (dma_mem_rsp_valid),
        .dma_mem_rsp_rdata_o        (dma_mem_rsp_rdata),
        .dma_desc_valid_o           (dma_desc_valid),
        .dma_desc_ready_i           (dma_desc_ready),
        .dma_desc_dir_o             (dma_desc_dir),
        .dma_desc_dram_addr_o       (dma_desc_dram_addr),
        .dma_desc_tcdm_addr_o       (dma_desc_tcdm_addr),
        .dma_desc_size_o            (dma_desc_size),
        .dma_desc_done_i            (dma_desc_done),
        .subsystem_busy_o           (subsystem_busy),
        .subsystem_done_irq_o       (subsystem_done_irq)
    );

    function automatic logic [31:0] tcdm_addr(input int word, input int bank);
        tcdm_addr = (32'(word) << 9) | (32'(bank) << 4);
    endfunction

    function automatic real fp16_to_real(input logic [15:0] h);
        int s, e, m;
        real v;
        s = h[15]; e = h[14:10]; m = h[9:0];
        if (e == 0) v = (m == 0) ? 0.0 : ($itor(m) / 1024.0) * (2.0 ** -14);
        else if (e == 31) v = 65504.0;
        else v = (1.0 + $itor(m) / 1024.0) * (2.0 ** (e - 15));
        return s ? -v : v;
    endfunction

    function automatic logic [15:0] real_to_fp16(input real r);
        logic [63:0] d;
        logic s;
        int e;
        logic [9:0] m;
        if (r == 0.0) return 16'h0000;
        d = $realtobits(r);
        s = d[63];
        e = d[62:52] - 1023 + 15;
        m = d[51:42];
        if (e <= 0) return {s, 15'd0};
        if (e >= 31) return {s, 5'h1E, 10'h3FF};
        return {s, e[4:0], m};
    endfunction

    function automatic real abs_r(input real x);
        return (x < 0.0) ? -x : x;
    endfunction

    function automatic logic [71:0] pack4(input logic [15:0] a, b, c, d);
        pack4 = {8'h00, d, c, b, a};
    endfunction

    task automatic cpu_mem_write(input logic [31:0] a, input logic [71:0] d);
        int guard;
        begin
            @(negedge clk);
            cpu_mem_req_valid = 1'b1;
            cpu_mem_req_write = 1'b1;
            cpu_mem_req_addr  = a;
            cpu_mem_req_wdata = d;
            cpu_mem_req_strb  = 9'h1FF;
            guard = 0;
            while (!cpu_mem_req_ready) begin
                @(posedge clk); guard++;
                if (guard > 1000) $fatal(1, "cpu_mem_write ready timeout");
            end
            @(posedge clk);
            cpu_mem_req_valid = 1'b0;
            cpu_mem_req_write = 1'b0;
            cpu_mem_req_strb  = 9'h000;
        end
    endtask

    task automatic cpu_mem_read(input logic [31:0] a, output logic [71:0] d);
        int guard;
        begin
            @(negedge clk);
            cpu_mem_req_valid = 1'b1;
            cpu_mem_req_write = 1'b0;
            cpu_mem_req_addr  = a;
            cpu_mem_req_wdata = '0;
            cpu_mem_req_strb  = 9'h000;
            guard = 0;
            while (!cpu_mem_req_ready) begin
                @(posedge clk); guard++;
                if (guard > 1000) $fatal(1, "cpu_mem_read ready timeout");
            end
            @(posedge clk);
            cpu_mem_req_valid = 1'b0;
            guard = 0;
            while (!cpu_mem_rsp_valid) begin
                @(posedge clk); guard++;
                if (guard > 1000) $fatal(1, "cpu_mem_read rsp timeout");
            end
            d = cpu_mem_rsp_rdata;
        end
    endtask

    task automatic csr_write(input logic [11:0] a, input logic [31:0] d);
        int guard;
        begin
            @(negedge clk);
            cpu_csr_req_valid = 1'b1;
            cpu_csr_req_write = 1'b1;
            cpu_csr_req_addr  = a;
            cpu_csr_req_data  = d;
            guard = 0;
            while (!cpu_csr_req_ready) begin
                @(posedge clk); guard++;
                if (guard > 1000) $fatal(1, "csr_write ready timeout");
            end
            @(posedge clk);
            cpu_csr_req_valid = 1'b0;
        end
    endtask

    task automatic csr_read(input logic [11:0] a, output logic [31:0] d);
        int guard;
        begin
            @(negedge clk);
            cpu_csr_req_valid = 1'b1;
            cpu_csr_req_write = 1'b0;
            cpu_csr_req_addr  = a;
            cpu_csr_req_data  = '0;
            guard = 0;
            while (!cpu_csr_req_ready) begin
                @(posedge clk); guard++;
                if (guard > 1000) $fatal(1, "csr_read ready timeout");
            end
            @(posedge clk);
            cpu_csr_req_valid = 1'b0;
            guard = 0;
            while (!cpu_csr_rsp_valid) begin
                @(posedge clk); guard++;
                if (guard > 1000) $fatal(1, "csr_read rsp timeout");
            end
            d = cpu_csr_rsp_data;
        end
    endtask

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

    task automatic wait_idle(input string tag);
        int guard;
        begin
            // Let the CSR start pulse propagate into the streamers/SFU/write-out
            // before checking the idle condition.
            repeat (8) @(posedge clk);
            guard = 0;
            while (dut.sfu_busy || dut.wo_busy || dut.u_read_fp16.busy_o ||
                   dut.u_read_fp16_b.busy_o) begin
                @(posedge clk); guard++;
                if (guard > 20000) $fatal(1, "%s timeout", tag);
            end
        end
    endtask

    task automatic run_sfu(input logic [2:0] op, input bit rms,
                           input int a_word, input int b_word, input int o_word);
        logic [31:0] start_nonrms, start_rms1, start_rms2;
        begin
            start_nonrms = 32'h0000_01D1; // commit, wo, rdA, sfu, rdB
            start_rms1   = 32'h0000_00C1; // commit, rdA, sfu
            start_rms2   = 32'h0000_0151; // commit, wo, rdA, rdB
            csr_write(A_CTRL,       32'h0000_0001);
            csr_write(A_SFU_CTRL,   (32'd1 << 5) | (32'(op) << 2) | 32'h2 | 32'h1);
            csr_write(A_SFU_INV_N,  32'h3D800000); // 1/16
            csr_write(A_SFU_EPS,    32'h3727C5AC); // 1e-5
            csr_write(A_RD_BASE,    tcdm_addr(a_word, A_BANK));
            csr_write(A_RD_NV,      N_VEC);
            csr_write(A_RD_LS,      32'h0000_0200);
            csr_write(A_SFU_B_BASE, tcdm_addr(b_word, B_BANK));
            csr_write(A_SFU_B_NV,   N_VEC);
            csr_write(A_SFU_B_LS,   32'h0000_0200);
            csr_write(A_WO_MODE,    32'd1); // FP16 output
            csr_write(A_WO_BASE,    tcdm_addr(o_word, O_BANK));
            csr_write(A_WO_NV,      N_VEC);
            csr_write(A_WO_LS,      32'h0000_0200);

            if (rms) begin
                csr_write(A_START, start_rms1);
                // Read A is the only streamer active in pass 1.
                repeat (4) @(posedge clk);
                while (dut.u_read_fp16.busy_o) @(posedge clk);
                csr_write(A_START, start_rms2);
            end else begin
                csr_write(A_START, start_nonrms);
            end
            wait_idle("SFU system job");
        end
    endtask

    task automatic check_lane(input string tag, input int lane, input real got,
                               input real exp, input real tol);
        real err;
        begin
            err = abs_r(got - exp);
            if (err > tol) begin
                $display("[FAIL] %s lane %0d: got=%f exp=%f err=%f tol=%f",
                         tag, lane, got, exp, err, tol);
                $fatal(1, "SFU system check failed");
            end
        end
    endtask

    real a_vals [0:15];
    real b_vals [0:15];
    real o_vals [0:15];
    real sum_sq, inv_rms;
    logic [31:0] rd32;
    int errors;

    initial begin
        rst_n = 1'b0;
        cpu_csr_req_valid = 0; cpu_csr_req_write = 0; cpu_csr_req_addr = '0; cpu_csr_req_data = '0;
        cpu_csr_rsp_ready = 1;
        cpu_mem_req_valid = 0; cpu_mem_req_write = 0; cpu_mem_req_addr = '0;
        cpu_mem_req_wdata = '0; cpu_mem_req_strb = '0;
        dma_mem_req_valid = 0; dma_mem_req_addr = '0; dma_mem_req_write = 0;
        dma_mem_req_wdata = '0; dma_mem_req_strb = '0;
        dma_desc_ready = 1; dma_desc_done = 0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (5) @(posedge clk);

        csr_read(A_RO_ID, rd32);
        if (rd32 !== CSR_ID_VALUE) $fatal(1, "CSR ID mismatch: %08x", rd32);

        errors = 0;

        // Residual
        for (int i = 0; i < 16; i++) begin
            a_vals[i] = 0.5 * (i + 1);
            b_vals[i] = -0.25 * i;
        end
        write_vec(0, A_BANK, a_vals);
        write_vec(0, B_BANK, b_vals);
        run_sfu(3'd1, 1'b0, 0, 0, 0);
        read_vec(0, O_BANK, o_vals);
        for (int i = 0; i < 16; i++)
            check_lane("RESIDUAL", i, o_vals[i], a_vals[i] + b_vals[i], 0.05);
        $display("[PASS] System RESIDUAL");

        // RoPE
        for (int k = 0; k < 8; k++) begin
            real th, xe, xo;
            th = 0.2 * (k + 1);
            xe = 1.0 + 0.1 * k;
            xo = -0.5 + 0.1 * k;
            a_vals[2*k] = xe; a_vals[2*k+1] = xo;
            b_vals[2*k] = $cos(th); b_vals[2*k+1] = $sin(th);
        end
        write_vec(0, A_BANK, a_vals);
        write_vec(0, B_BANK, b_vals);
        run_sfu(3'd3, 1'b0, 0, 0, 0);
        read_vec(0, O_BANK, o_vals);
        for (int k = 0; k < 8; k++) begin
            real th, xe, xo;
            th = 0.2 * (k + 1);
            xe = 1.0 + 0.1 * k;
            xo = -0.5 + 0.1 * k;
            check_lane("ROPE_EVEN", 2*k, o_vals[2*k], xe*$cos(th) - xo*$sin(th), 0.10);
            check_lane("ROPE_ODD", 2*k+1, o_vals[2*k+1], xe*$sin(th) + xo*$cos(th), 0.10);
        end
        $display("[PASS] System ROPE");

        // SwiGLU
        for (int i = 0; i < 16; i++) begin
            real g, u;
            g = -3.0 + 0.4 * i;
            u = 0.5 + 0.1 * i;
            a_vals[i] = g; b_vals[i] = u;
        end
        write_vec(0, A_BANK, a_vals);
        write_vec(0, B_BANK, b_vals);
        run_sfu(3'd4, 1'b0, 0, 0, 0);
        read_vec(0, O_BANK, o_vals);
        for (int i = 0; i < 16; i++) begin
            real g, u, silu;
            g = -3.0 + 0.4 * i;
            u = 0.5 + 0.1 * i;
            silu = g / (1.0 + $exp(-g));
            check_lane("SWIGLU", i, o_vals[i], silu * u, 0.15);
        end
        $display("[PASS] System SWIGLU");

        // RMSNorm (one 16-element vector, gamma=1)
        sum_sq = 0.0;
        for (int i = 0; i < 16; i++) begin
            a_vals[i] = 0.25 * ((i % 7) - 3);
            b_vals[i] = 1.0;
            sum_sq += a_vals[i] * a_vals[i];
        end
        inv_rms = 1.0 / $sqrt(sum_sq / 16.0 + 1e-5);
        write_vec(0, A_BANK, a_vals);
        write_vec(0, B_BANK, b_vals);
        run_sfu(3'd2, 1'b1, 0, 0, 0);
        read_vec(0, O_BANK, o_vals);
        for (int i = 0; i < 16; i++)
            check_lane("RMSNORM", i, o_vals[i], a_vals[i] * inv_rms, 0.10);
        $display("[PASS] System RMSNORM");

        // ---------------------------------------------------------------------
        // Fused Core -> SFU (sfu_in_a_src = 1):
        //   16x16 GEMM (A=1.0, W=1.0 -> 16.0) -> SFU RESIDUAL (+ B) -> TCDM
        // ---------------------------------------------------------------------
        begin
            logic [71:0] pkt_ones;
            int guard;
            // Log8 scale = 0, 16x E2M1 = 1.0 (4'b0010) -> 64'h2222_2222_2222_2222
            pkt_ones = {8'h00, 64'h2222_2222_2222_2222};
            for (int t = 0; t < 16; t++) begin
                // Activation packet for token t in bank 20 (stride 0x200)
                cpu_mem_write(tcdm_addr(t, 20), pkt_ones);
                // Weight packet for lane t in bank t (banks 0..15)
                cpu_mem_write(tcdm_addr(0, t), pkt_ones);
                // Residual B vector t in bank 22
                for (int i = 0; i < 16; i++) b_vals[i] = 0.25 * (t + 1);
                write_vec(t * 4, 22, b_vals);
            end

            csr_write(A_CTRL,       32'h0000_0000); // Sequencer mode
            csr_write(A_SEQ_CTRL,   32'h0000_0000); // op=00 GEMM, out_mode=0 (FP16)
            csr_write(A_GEMM_NT,    32'd1);
            csr_write(A_GEMM_MT,    32'd1);
            csr_write(A_GEMM_KB,    32'd1);
            csr_write(A_RING_BASE,  tcdm_addr(0, 0));
            csr_write(A_Q_BASE,     tcdm_addr(0, 20));
            csr_write(A_OUT_BASE,   tcdm_addr(0, 23));

            // SFU: sfu_en=1, sfu_out_dst=1 (FP16), sfu_op=1 (RESIDUAL),
            //      sfu_num_vecs=16, sfu_in_a_src=1 (direct from u_core)
            csr_write(A_SFU_CTRL,   (32'd1 << 21) | (32'd16 << 5) | (32'd1 << 2) | 32'h2 | 32'h1);
            csr_write(A_SFU_B_BASE, tcdm_addr(0, 22));
            csr_write(A_SFU_B_NV,   32'd16);
            csr_write(A_SFU_B_LS,   32'h0000_0200);

            // Start: commit(0) | seq(1) | sfu(7) | rd_b(8) = 0x183
            csr_write(A_START,      32'h0000_0183);

            // Wait with counters (start counting immediately)
            begin
                int cv_cnt, sfu_out_cnt, wo_req_cnt, drain_cnt;
                cv_cnt = 0; sfu_out_cnt = 0; wo_req_cnt = 0; drain_cnt = 0;
                guard = 0;
                while (guard < 12 || dut.seq_busy || dut.sfu_busy || dut.wo_busy || dut.u_read_fp16_b.busy_o) begin
                    @(posedge clk); guard++;
                    if (dut.tile_drain_req) drain_cnt++;
                    if (dut.core_fp16_valid && dut.core_fp16_ready) cv_cnt++;
                    if (dut.sfu_out_valid && dut.sfu_out_ready) sfu_out_cnt++;
                    if (dut.wo_m_req_valid && dut.wo_m_req_ready) wo_req_cnt++;
                    if (guard > 30000) begin
                        $display("TIMEOUT: drain=%0d cv=%0d sfu_out=%0d wo_req=%0d state=%0d act_b=%b wt_b=%b act_v=%b wt_v=%h wt_req=%h act_r=%b wt_r=%h wt_st=%0d wt_rep=%0d wt_skid=%h",
                                 drain_cnt, cv_cnt, sfu_out_cnt, wo_req_cnt, dut.u_seq.state_q,
                                 dut.act_busy, dut.wt_busy, dut.act_stream_valid, dut.wt_stream_valid,
                                 dut.wt_m_req_valid, dut.act_stream_ready, dut.wt_stream_ready,
                                 dut.u_wt_streamer.state_q, dut.u_wt_streamer.rep_cnt_q, dut.u_wt_streamer.skid_valid);
                        $fatal(1, "Fused Core->SFU timeout");
                    end
                end
            end

            for (int t = 0; t < 16; t++) begin
                read_vec(t * 4, 23, o_vals);
                for (int i = 0; i < 16; i++)
                    check_lane("FUSED_CORE_SFU_RESIDUAL", i, o_vals[i], 16.0 + 0.25 * (t + 1), 0.10);
            end
            $display("[PASS] System FUSED Core->SFU RESIDUAL (16x16 GEMM + Residual, sfu_in_a_src=1)");
        end

        $display("\n=== ALL SFU SYSTEM TESTS PASSED (0 errors) ===");
        $finish;
    end

    initial begin
        #(10 * 1000 * 1000);
        $fatal(1, "watchdog timeout");
    end

endmodule
