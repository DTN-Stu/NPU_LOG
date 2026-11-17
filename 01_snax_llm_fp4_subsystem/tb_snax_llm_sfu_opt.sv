// =============================================================================
// Module: tb_snax_llm_sfu_opt.sv
// Description: Self-checking testbench for parameterized snax_llm_sfu_opt.
//              Supports ALU_LANES = 16, 8, or 4 (default 8, override via +define+ALU_LANES=4).
// =============================================================================
`timescale 1ns/1ps

`ifndef ALU_LANES
`define ALU_LANES 8
`endif

module tb_snax_llm_sfu_opt;
    localparam int NUM_LANES = 16;
    localparam int ALU_L     = `ALU_LANES;

    logic clk = 1'b0;
    logic rst_n;
    always #5 clk = ~clk;

    logic                       start_i;
    logic [2:0]                 cfg_op_i;
    logic [15:0]                cfg_num_vecs_i;
    logic [31:0]                cfg_inv_n_fp32_i;
    logic [31:0]                cfg_eps_fp32_i;
    logic                       busy_o, done_o;

    logic [NUM_LANES*16-1:0]    in_a_fp16_i, in_b_fp16_i, out_fp16_o;
    logic                       in_a_valid_i, in_a_ready_o;
    logic                       in_b_valid_i, in_b_ready_o;
    logic                       out_valid_o,  out_ready_i;

    snax_llm_sfu_opt #(
        .NUM_LANES (NUM_LANES),
        .ALU_LANES (ALU_L)
    ) dut (
        .*
    );

    function automatic real fp16_to_real(input logic [15:0] h);
        int s, e, m;
        real val;
        s = h[15];
        e = h[14:10];
        m = h[9:0];
        if (e == 0)       val = (m == 0) ? 0.0 : ($itor(m) / 1024.0) * (2.0 ** -14);
        else if (e == 31) val = 65504.0;
        else              val = (1.0 + $itor(m) / 1024.0) * (2.0 ** (e - 15));
        return s ? -val : val;
    endfunction

    function automatic logic [15:0] real_to_fp16(input real r);
        logic [63:0] d_bits;
        logic        s;
        int          e;
        logic [9:0]  m;
        if (r == 0.0) return 16'h0000;
        d_bits = $realtobits(r);
        s = d_bits[63];
        e = d_bits[62:52] - 1023 + 15;
        m = d_bits[51:42];
        if (e <= 0)  return {s, 15'd0};
        if (e >= 31) return {s, 5'h1E, 10'h3FF};
        return {s, e[4:0], m};
    endfunction

    function automatic real abs_r(input real x);
        return (x < 0.0) ? -x : x;
    endfunction

    int err_cnt = 0;

    task automatic check_lane(input string tag, input int lane, input logic [15:0] got, input real exp_val, input real tol);
        real g_val, err;
        g_val = fp16_to_real(got);
        err   = abs_r(g_val - exp_val);
        if (err > tol) begin
            $display("[FAIL] (ALU_LANES=%0d) %s lane %0d: got=%0f (0x%04x), expected=%0f, err=%0f > tol=%0f",
                     ALU_L, tag, lane, g_val, got, exp_val, err, tol);
            err_cnt++;
        end
    endtask

    initial begin
        $display("=== Testing snax_llm_sfu_opt with ALU_LANES = %0d ===", ALU_L);
        rst_n            = 1'b0;
        start_i          = 1'b0;
        cfg_op_i         = 3'd0;
        cfg_num_vecs_i   = 16'd4;
        cfg_inv_n_fp32_i = 32'h3C800000; // 1/64
        cfg_eps_fp32_i   = 32'h3727C5AC; // 1e-5
        in_a_fp16_i      = '0;
        in_b_fp16_i      = '0;
        in_a_valid_i     = 1'b0;
        in_b_valid_i     = 1'b0;
        out_ready_i      = 1'b1;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        // ---------------------------------------------------------------------
        // Test 1: RESIDUAL_ADD
        // ---------------------------------------------------------------------
        @(negedge clk);
        cfg_op_i     = 3'd1;
        in_a_valid_i = 1'b1;
        in_b_valid_i = 1'b1;
        for (int i = 0; i < 16; i++) begin
            in_a_fp16_i[i*16 +: 16] = real_to_fp16(0.5 * (i + 1));
            in_b_fp16_i[i*16 +: 16] = real_to_fp16(-0.25 * i);
        end
        @(posedge clk); #1;
        in_a_valid_i = 1'b0;
        in_b_valid_i = 1'b0;
        while (!out_valid_o) begin @(posedge clk); #1; end
        for (int i = 0; i < 16; i++) begin
            check_lane("RESIDUAL", i, out_fp16_o[i*16 +: 16], 0.5*(i+1) - 0.25*i, 0.02);
        end
        $display("[PASS] Test 1: RESIDUAL_ADD");

        // ---------------------------------------------------------------------
        // Test 2: ROPE
        // ---------------------------------------------------------------------
        begin
            real th, xe, xo;
            @(negedge clk);
            cfg_op_i     = 3'd3;
            in_a_valid_i = 1'b1;
            in_b_valid_i = 1'b1;
            for (int k = 0; k < 8; k++) begin
                th = 0.2 * (k + 1);
                in_a_fp16_i[(2*k)*16 +: 16]   = real_to_fp16(1.0 + 0.1*k);
                in_a_fp16_i[(2*k+1)*16 +: 16] = real_to_fp16(-0.5 + 0.1*k);
                in_b_fp16_i[(2*k)*16 +: 16]   = real_to_fp16($cos(th));
                in_b_fp16_i[(2*k+1)*16 +: 16] = real_to_fp16($sin(th));
            end
            @(posedge clk); #1;
            in_a_valid_i = 1'b0;
            in_b_valid_i = 1'b0;
            while (!out_valid_o) begin @(posedge clk); #1; end
            for (int k = 0; k < 8; k++) begin
                th = 0.2 * (k + 1);
                xe = fp16_to_real(real_to_fp16(1.0 + 0.1*k));
                xo = fp16_to_real(real_to_fp16(-0.5 + 0.1*k));
                check_lane("ROPE_EVEN", 2*k,   out_fp16_o[(2*k)*16 +: 16],   xe*$cos(th) - xo*$sin(th), 0.02);
                check_lane("ROPE_ODD",  2*k+1, out_fp16_o[(2*k+1)*16 +: 16], xe*$sin(th) + xo*$cos(th), 0.02);
            end
            $display("[PASS] Test 2: ROPE");
        end

        // ---------------------------------------------------------------------
        // Test 3: SWIGLU
        // ---------------------------------------------------------------------
        begin
            real g, u, silu;
            @(negedge clk);
            cfg_op_i     = 3'd4;
            in_a_valid_i = 1'b1;
            in_b_valid_i = 1'b1;
            for (int i = 0; i < 16; i++) begin
                g = -3.0 + 0.4 * i;
                u = 0.5 + 0.1 * i;
                in_a_fp16_i[i*16 +: 16] = real_to_fp16(g);
                in_b_fp16_i[i*16 +: 16] = real_to_fp16(u);
            end
            @(posedge clk); #1;
            in_a_valid_i = 1'b0;
            in_b_valid_i = 1'b0;
            while (!out_valid_o) begin @(posedge clk); #1; end
            for (int i = 0; i < 16; i++) begin
                g = fp16_to_real(real_to_fp16(-3.0 + 0.4 * i));
                u = fp16_to_real(real_to_fp16(0.5 + 0.1 * i));
                silu = g / (1.0 + $exp(-g));
                check_lane("SWIGLU", i, out_fp16_o[i*16 +: 16], silu * u, 0.10);
            end
            $display("[PASS] Test 3: SWIGLU");
        end

        // ---------------------------------------------------------------------
        // Test 4: 2-Pass Streaming RMSNORM
        // ---------------------------------------------------------------------
        begin
            real x_vals[64];
            real sum_sq;
            real inv_rms;
            sum_sq = 0.0;
            for (int n = 0; n < 64; n++) begin
                x_vals[n] = fp16_to_real(real_to_fp16(0.25 * ((n % 7) - 3)));
                sum_sq += x_vals[n] * x_vals[n];
            end
            inv_rms = 1.0 / $sqrt(sum_sq / 64.0 + 1e-5);

            @(negedge clk);
            cfg_op_i = 3'd2;
            start_i  = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_i  = 1'b0;

            // Pass 1: stream 4 vectors of x (wait for in_a_ready_o)
            for (int v = 0; v < 4; v++) begin
                while (!in_a_ready_o) @(negedge clk);
                in_a_valid_i = 1'b1;
                for (int i = 0; i < 16; i++) begin
                    in_a_fp16_i[i*16 +: 16] = real_to_fp16(x_vals[v*16 + i]);
                end
                @(posedge clk);
                @(negedge clk);
                in_a_valid_i = 1'b0;
            end

            // Wait NUM_CHUNKS cycles for remaining Pass 1 chunks + ST_RMS_RSQRT -> ST_RMS_PASS2
            repeat (16 / ALU_L) @(posedge clk);

            // Pass 2: stream x + gamma
            for (int v = 0; v < 4; v++) begin
                @(negedge clk);
                in_a_valid_i = 1'b1;
                in_b_valid_i = 1'b1;
                for (int i = 0; i < 16; i++) begin
                    in_a_fp16_i[i*16 +: 16] = real_to_fp16(x_vals[v*16 + i]);
                    in_b_fp16_i[i*16 +: 16] = real_to_fp16(1.0);
                end
                @(posedge clk); #1;
                in_a_valid_i = 1'b0;
                in_b_valid_i = 1'b0;
                while (!out_valid_o) begin @(posedge clk); #1; end
                for (int i = 0; i < 16; i++) begin
                    check_lane("RMSNORM", v*16 + i, out_fp16_o[i*16 +: 16], x_vals[v*16 + i] * inv_rms, 0.03);
                end
            end
            $display("[PASS] Test 4: RMSNORM (2-Pass Zero-Buffer)");
        end

        if (err_cnt == 0) begin
            $display("\n=== ALL SFU_OPT (ALU_LANES=%0d) TESTS PASSED (0 errors) ===", ALU_L);
        end else begin
            $display("\n=== SFU_OPT (ALU_LANES=%0d) TESTS FAILED (%0d errors) ===", ALU_L, err_cnt);
        end
        $finish;
    end

endmodule
