// =============================================================================
// Module: tb_ablation_saif.sv
// Description: 500 MHz (2ns clock) SAIF power & activity testbench for Configs #1..#5.
//   Select DUT:
//     +define+DUT_CFG1 .. +define+DUT_CFG5 (default CFG5)
//   Select Stimulus Mode:
//     +define+STIM_LLM_HEX  : Load real quantized 72-bit LLM layer traces
//                             (stim_act.hex, stim_wt.hex) via $readmemh
//     (default)             : LLM-like sparse/Gaussian-quantized distribution
//                             (high probability of 0.0/0.5/1.0, rare outliers)
//     +define+STIM_UNIFORM  : Uniform-random bits (worst-case toggle, MiX style)
// =============================================================================
`timescale 1ns/10ps

module tb_ablation_saif;
    localparam int NUM_LANES = 16;
    localparam int ACC_DEPTH = 16;
    localparam int PKT_W     = 72;
    localparam int WARMUP    = 64;
    localparam int CAPTURE   = 4000;
    localparam int TRACE_LEN = 4096;

    logic clk = 1'b0;
    logic reset_n;

    logic [NUM_LANES-1:0]              lane_valid_in;
    logic [$clog2(ACC_DEPTH)-1:0]      lane_acc_idx;
    logic [PKT_W-1:0]                    act_broadcast;
    logic [NUM_LANES-1:0][PKT_W-1:0]   wt_in;
    logic                                tile_drain_req;
    logic [$clog2(ACC_DEPTH)-1:0]      tile_drain_acc_idx;
    logic                                tile_clear_on_drain;

`ifdef DUT_CFG1
    logic [NUM_LANES-1:0][31:0] out_vec;
    logic                       out_valid;
    cfg1_mxfp4_g16_fp32_ff dut (
        .clk                 (clk),
        .reset_n             (reset_n),
        .lane_valid_in       (lane_valid_in),
        .lane_acc_idx        (lane_acc_idx),
        .act_broadcast       (act_broadcast),
        .wt_in               (wt_in),
        .tile_drain_req      (tile_drain_req),
        .tile_drain_acc_idx  (tile_drain_acc_idx),
        .tile_clear_on_drain (tile_clear_on_drain),
        .out_fp32_vec        (out_vec),
        .out_valid           (out_valid)
    );
`elsif DUT_CFG2
    logic [NUM_LANES-1:0][31:0] out_vec;
    logic                       out_valid;
    cfg2_nvfp4_fp32_ff dut (
        .clk                 (clk),
        .reset_n             (reset_n),
        .lane_valid_in       (lane_valid_in),
        .lane_acc_idx        (lane_acc_idx),
        .act_broadcast       (act_broadcast),
        .wt_in               (wt_in),
        .tile_drain_req      (tile_drain_req),
        .tile_drain_acc_idx  (tile_drain_acc_idx),
        .tile_clear_on_drain (tile_clear_on_drain),
        .out_fp32_vec        (out_vec),
        .out_valid           (out_valid)
    );
`elsif DUT_CFG3
    logic [NUM_LANES-1:0][15:0] out_vec;
    logic                       out_valid;
    cfg3_nvfp4_int32_lazy_ff dut (
        .clk                 (clk),
        .reset_n             (reset_n),
        .lane_valid_in       (lane_valid_in),
        .lane_acc_idx        (lane_acc_idx),
        .act_broadcast       (act_broadcast),
        .wt_in               (wt_in),
        .tile_drain_req      (tile_drain_req),
        .tile_drain_acc_idx  (tile_drain_acc_idx),
        .tile_clear_on_drain (tile_clear_on_drain),
        .out_fp16_vec        (out_vec),
        .out_valid           (out_valid)
    );
`elsif DUT_CFG4
    logic [NUM_LANES-1:0][15:0] out_vec;
    logic                       out_valid;
    cfg4_log8_int32_lazy_ff dut (
        .clk                 (clk),
        .reset_n             (reset_n),
        .lane_valid_in       (lane_valid_in),
        .lane_acc_idx        (lane_acc_idx),
        .act_broadcast       (act_broadcast),
        .wt_in               (wt_in),
        .tile_drain_req      (tile_drain_req),
        .tile_drain_acc_idx  (tile_drain_acc_idx),
        .tile_clear_on_drain (tile_clear_on_drain),
        .out_fp16_vec        (out_vec),
        .out_valid           (out_valid)
    );
`elsif DUT_CFG5
    logic [NUM_LANES-1:0][15:0] out_vec;
    logic                       out_valid;
    cfg5_log8_int32_lazy_latch dut (
        .clk                 (clk),
        .reset_n             (reset_n),
        .lane_valid_in       (lane_valid_in),
        .lane_acc_idx        (lane_acc_idx),
        .act_broadcast       (act_broadcast),
        .wt_in               (wt_in),
        .tile_drain_req      (tile_drain_req),
        .tile_drain_acc_idx  (tile_drain_acc_idx),
        .tile_clear_on_drain (tile_clear_on_drain),
        .out_fp16_vec        (out_vec),
        .out_valid           (out_valid)
    );
`else
    logic [NUM_LANES-1:0][31:0] out_vec;
    logic                       out_valid;
    cfg6_log8_fp32_ff dut (
        .clk                 (clk),
        .reset_n             (reset_n),
        .lane_valid_in       (lane_valid_in),
        .lane_acc_idx        (lane_acc_idx),
        .act_broadcast       (act_broadcast),
        .wt_in               (wt_in),
        .tile_drain_req      (tile_drain_req),
        .tile_drain_acc_idx  (tile_drain_acc_idx),
        .tile_clear_on_drain (tile_clear_on_drain),
        .out_fp32_vec        (out_vec),
        .out_valid           (out_valid)
    );
`endif

    // Optional real LLM trace memories (72-bit packets)
    logic [PKT_W-1:0] act_trace_mem [TRACE_LEN];
    logic [PKT_W-1:0] wt_trace_mem  [TRACE_LEN * NUM_LANES];

    // 2 ns clock -> 500 MHz
    always #1 clk = ~clk;

    // Generate LLM-like E2M1 element (bell-shaped: mostly 0, 0.5, 1.0, 1.5; rare 4.0, 6.0)
    function automatic logic [3:0] rand_llm_fp4();
        int r;
        logic s;
        logic [2:0] mag;
        r = $urandom_range(0, 99);
        s = $urandom_range(0, 1);
        if      (r < 25) mag = 3'b000; // 0.0 (25%)
        else if (r < 55) mag = 3'b001; // 0.5 (30%)
        else if (r < 78) mag = 3'b010; // 1.0 (23%)
        else if (r < 90) mag = 3'b011; // 1.5 (12%)
        else if (r < 96) mag = 3'b100; // 2.0 (6%)
        else if (r < 98) mag = 3'b101; // 3.0 (2%)
        else if (r < 99) mag = 3'b110; // 4.0 (1%)
        else             mag = 3'b111; // 6.0 (1%)
        return (mag == 3'b000) ? 4'b0000 : {s, mag};
    endfunction

    function automatic logic [PKT_W-1:0] rand_llm_pkt();
        logic [PKT_W-1:0] pkt;
        // Scale centered in realistic LLM exponent range matching each format's encoding
`ifdef DUT_CFG1
        // MXFP4 (E8M0, bias=127): 2^-4 .. 2^4
        pkt[71:64] = 8'($urandom_range(123, 131));
`elsif DUT_CFG2
        // NVFP4 (E4M3, bias=7): normal FP8 scales ~ 0.625 .. 7.5
        pkt[71:64] = 8'($urandom_range(50, 75));
`elsif DUT_CFG3
        pkt[71:64] = 8'($urandom_range(50, 75));
`else
        // Log8 (signed S4.3, 2^(e/8)): e in [-24, +8] -> 2^-3 .. 2^1 (prevents artificial Q16.16 saturation)
        pkt[71:64] = 8'(signed'($urandom_range(0, 32)) - 24);
`endif
        for (int e = 0; e < 16; e++) begin
            pkt[e*4 +: 4] = rand_llm_fp4();
        end
        return pkt;
    endfunction

    task automatic drive_stimulus(input int cyc);
        int idx;
        idx = cyc % TRACE_LEN;
        lane_valid_in = 16'hFFFF;
        lane_acc_idx  = cyc[3:0];

`ifdef STIM_LLM_HEX
        act_broadcast = act_trace_mem[idx];
        for (int l = 0; l < NUM_LANES; l++) begin
            wt_in[l] = wt_trace_mem[idx * NUM_LANES + l];
        end
`elsif STIM_UNIFORM
        begin
            logic [95:0] rtmp;
            rtmp = {$random, $random, $random};
            act_broadcast = rtmp[PKT_W-1:0];
            for (int l = 0; l < NUM_LANES; l++) begin
                rtmp = {$random, $random, $random};
                wt_in[l] = rtmp[PKT_W-1:0];
            end
        end
`else
        act_broadcast = rand_llm_pkt();
        for (int l = 0; l < NUM_LANES; l++) begin
            wt_in[l] = rand_llm_pkt();
        end
`endif

        tile_drain_req      = (cyc % 16 == 15);
        tile_drain_acc_idx  = (cyc >> 4) & 4'hF;
        tile_clear_on_drain = 1'b1;
    endtask

    initial begin
`ifdef STIM_LLM_HEX
        $readmemh("stim_act.hex", act_trace_mem);
        $readmemh("stim_wt.hex",  wt_trace_mem);
`endif
        reset_n             = 1'b0;
        lane_valid_in       = '0;
        lane_acc_idx        = '0;
        act_broadcast       = '0;
        wt_in               = '0;
        tile_drain_req      = 1'b0;
        tile_drain_acc_idx  = '0;
        tile_clear_on_drain = 1'b0;

        repeat (4) @(posedge clk);
        reset_n = 1'b1;

        for (int c = 0; c < WARMUP; c++) begin
            @(negedge clk);
            drive_stimulus(c);
        end

`ifdef DUMP_SAIF
        $set_toggle_region(tb_ablation_saif.dut);
        $toggle_start;
`endif
        for (int c = 0; c < CAPTURE; c++) begin
            @(negedge clk);
            drive_stimulus(c);
        end
`ifdef DUMP_SAIF
        $toggle_stop;
        $toggle_report("ablation_dut.saif", 1.0e-9, "tb_ablation_saif.dut");
`endif
        $display("[tb_ablation_saif] Completed %0d capture cycles cleanly.", CAPTURE);
        $finish;
    end

endmodule
