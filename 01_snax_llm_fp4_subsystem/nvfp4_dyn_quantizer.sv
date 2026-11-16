// =============================================================================
// Module: nvfp4_dyn_quantizer.sv
// Description: Baseline Dynamic FP16 -> NVFP4 (16 x E2M1 + FP8 E4M3 Scale)
//              Quantizer for IEEE TCAD/TVLSI Hardware Synthesis Comparison
//              against the proposed Division-Free SLQ Encoder (slq_encoder.sv).
//
// Mathematical & Datapath Operations Required by NVFP4 (B = 16):
//   Stage 1 (Max-Abs Reduction & E4M3 Block Scale Generation):
//     1. 16-input 15-bit magnitude comparator tree -> x_max = max_i |x_i|
//     2. Compute S_fp16 = x_max / 6.0 = x_max * (2/3) * 2^(-2):
//        - Requires an 11b x 11b mantissa multiplier (sig_max * 0x555)
//        - Normalizes and rounds-to-nearest-even into FP8 E4M3:
//          S_e4m3 = 2^(E_s - 7) * (1 + M_s / 8)
//   Stage 2 (Reciprocal Lookup + 16x Parallel FP Multiply-Shift-Quantize):
//     1. 8-entry Reciprocal Mantissa LUT: R_m = round(2^7 / (1 + M_s/8)) (8-bit Q1.7)
//     2. Across all 16 lanes in parallel:
//        - Exponent subtractor: dE_i = (E_i - 15) - (E_s - 7)
//        - 16 x Parallel Mantissa Multipliers (11b x 8b -> 19b):
//            P_i = (1.M_i) * R_m
//        - 16 x Parallel Barrel Shifters:
//            Shifts P_i by dE_i to form fixed-point magnitude y_i = |x_i| / S_e4m3
//        - 16 x 7-Threshold E2M1 Midpoint Comparators:
//            Compares y_i against {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}
// =============================================================================
`timescale 1ns/1ps

module nvfp4_dyn_quantizer #(
    parameter int BLOCK_SIZE   = 16,
    parameter int PACKET_WIDTH = 72
)(
    input  logic                          clk,
    input  logic                          reset_n,

    input  logic [BLOCK_SIZE-1:0][15:0]   fp16_in,
    input  logic                          in_valid,
    output logic                         in_ready,

    output logic [PACKET_WIDTH-1:0]      packed_vec,
    output logic                         out_valid,
    input  logic                          out_ready
);

    wire pipe_advance = !out_valid || out_ready;
    assign in_ready   = pipe_advance;

    // =========================================================================
    // STAGE 1: 15-bit Max-Abs Tree + (x_max / 6.0) E4M3 Scale Generator
    // =========================================================================
    logic [BLOCK_SIZE-1:0]       sign_s1;
    logic [BLOCK_SIZE-1:0][14:0] abs_s1;

    genvar gi;
    generate
        for (gi = 0; gi < BLOCK_SIZE; gi++) begin : GEN_ABS
            assign sign_s1[gi] = fp16_in[gi][15];
            // Flush NaN/Inf (E=31) to max normal FP16 magnitude
            assign abs_s1[gi]  = (fp16_in[gi][14:10] == 5'd31) ? 15'h7BFF
                                                                : fp16_in[gi][14:0];
        end
    endgenerate

    function automatic logic [14:0] max15(input logic [14:0] a, input logic [14:0] b);
        return (a > b) ? a : b;
    endfunction

    logic [7:0][14:0] tree_l1;
    logic [3:0][14:0] tree_l2;
    logic [1:0][14:0] tree_l3;
    logic [14:0]      max_abs;

    int k;
    always_comb begin
        for (k = 0; k < 8; k++) tree_l1[k] = max15(abs_s1[2*k], abs_s1[2*k+1]);
        for (k = 0; k < 4; k++) tree_l2[k] = max15(tree_l1[2*k], tree_l1[2*k+1]);
        for (k = 0; k < 2; k++) tree_l3[k] = max15(tree_l2[2*k], tree_l2[2*k+1]);
        max_abs = max15(tree_l3[0], tree_l3[1]);
    end

    // Compute S_e4m3 = round_e4m3(max_abs / 6.0):
    //   1/6 = (2/3) * 2^(-3).
    //   Multiply 11-bit significand {1'b1, M_max} (Q1.10) by 2/3 in Q0.11 (11'd1365)
    //   -> 22-bit product in Q1.21.
    wire [4:0]  e_max     = max_abs[14:10];
    wire [9:0]  m_max     = max_abs[9:0];
    wire        max_zero  = (e_max == 5'd0);
    wire [10:0] sig_max   = max_zero ? 11'd0 : {1'b1, m_max};

    // [STAGE-1 MULTIPLIER]: 11b x 11b constant multiplier for /6.0
    localparam logic [10:0] INV3_Q11 = 11'd1365; // round(2/3 * 2048) = 1365
    wire [21:0] div6_prod = sig_max * INV3_Q11;  // Q1.21; MSB is at bit 21 or bit 20

    wire        norm_msb  = div6_prod[21];       // 1 if product in [1.0, 4/3), 0 if in [2/3, 1.0)
    wire [2:0]  mant_raw  = norm_msb ? div6_prod[20:18] : div6_prod[19:17];
    wire        round_bit = norm_msb ? div6_prod[17]    : div6_prod[16];
    wire        sticky    = norm_msb ? (|div6_prod[16:0]) : (|div6_prod[15:0]);

    wire        rnd_up    = round_bit & (sticky | mant_raw[0]);
    wire [3:0]  mant_rnd  = {1'b0, mant_raw} + {3'b000, rnd_up};

    // Exponent conversion from FP16 (bias 15) divided by 6 (shift -2 or -3) to E4M3 (bias 7):
    //   exp_e4m3 = (e_max - 15) - (norm_msb ? 2 : 3) + mant_rnd[3] + 7
    wire signed [6:0] exp_s_raw = $signed({2'b00, e_max}) - (norm_msb ? 7'sd10 : 7'sd11)
                                  + $signed({6'd0, mant_rnd[3]});

    logic [7:0] scale_e4m3_s1;
    always_comb begin
        if (max_zero || exp_s_raw <= 7'sd0) begin
            scale_e4m3_s1 = 8'h00;
        end else if (exp_s_raw >= 7'sd15) begin
            scale_e4m3_s1 = {1'b0, 4'd15, 3'b110}; // Max normal E4M3 (448.0)
        end else begin
            scale_e4m3_s1 = {1'b0, exp_s_raw[3:0], mant_rnd[2:0]};
        end
    end

    // Pipeline Register Stage 1 -> Stage 2
    logic                       valid_q1;
    logic [BLOCK_SIZE-1:0]      sign_q1;
    logic [BLOCK_SIZE-1:0][14:0] abs_q1;
    logic [7:0]                 scale_e4m3_q1;

    int s1_i;
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            valid_q1      <= 1'b0;
            sign_q1       <= '0;
            scale_e4m3_q1 <= 8'd0;
            for (s1_i = 0; s1_i < BLOCK_SIZE; s1_i++) abs_q1[s1_i] <= 15'd0;
        end else if (pipe_advance) begin
            valid_q1 <= in_valid;
            if (in_valid) begin
                sign_q1       <= sign_s1;
                scale_e4m3_q1 <= scale_e4m3_s1;
                for (s1_i = 0; s1_i < BLOCK_SIZE; s1_i++) abs_q1[s1_i] <= abs_s1[s1_i];
            end
        end
    end

    // =========================================================================
    // STAGE 2: Reciprocal LUT + 16x Parallel Multipliers & Shifters -> E2M1
    // =========================================================================
    wire [3:0] es_q1     = scale_e4m3_q1[6:3];
    wire [2:0] ms_q1     = scale_e4m3_q1[2:0];
    wire       s_is_zero = (scale_e4m3_q1[6:0] == 7'd0);

    // 8-entry LUT for reciprocal of E4M3 significand: R_m = round(128 / (1 + ms/8)) in Q1.7
    logic [7:0] recip_mant_q7;
    always_comb begin
        unique case (ms_q1)
            3'd0: recip_mant_q7 = 8'd128; // 1.0 / 1.000 = 1.0000 * 128
            3'd1: recip_mant_q7 = 8'd114; // 1.0 / 1.125 = 0.8889 * 128
            3'd2: recip_mant_q7 = 8'd102; // 1.0 / 1.250 = 0.8000 * 128
            3'd3: recip_mant_q7 = 8'd93;  // 1.0 / 1.375 = 0.7273 * 128
            3'd4: recip_mant_q7 = 8'd85;  // 1.0 / 1.500 = 0.6667 * 128
            3'd5: recip_mant_q7 = 8'd79;  // 1.0 / 1.625 = 0.6154 * 128
            3'd6: recip_mant_q7 = 8'd73;  // 1.0 / 1.750 = 0.5714 * 128
            3'd7: recip_mant_q7 = 8'd68;  // 1.0 / 1.875 = 0.5333 * 128
        endcase
    end

    // E2M1 arithmetic midpoints in Q4.8 fixed-point:
    //   0.25 -> 64, 0.75 -> 192, 1.25 -> 320, 1.75 -> 448,
    //   2.50 -> 640, 3.50 -> 896, 5.00 -> 1280
    function automatic logic [2:0] e2m1_mag_from_q4_8(input logic [11:0] y_q8);
        if      (y_q8 >= 12'd1280) return 3'b111; // 6.0
        else if (y_q8 >= 12'd896)  return 3'b110; // 4.0
        else if (y_q8 >= 12'd640)  return 3'b101; // 3.0
        else if (y_q8 >= 12'd448)  return 3'b100; // 2.0
        else if (y_q8 >= 12'd320)  return 3'b011; // 1.5
        else if (y_q8 >= 12'd192)  return 3'b010; // 1.0
        else if (y_q8 >= 12'd64)   return 3'b001; // 0.5
        else                       return 3'b000; // 0.0
    endfunction

    logic [BLOCK_SIZE-1:0][3:0] fp4_code_s2;

    genvar lane;
    generate
        for (lane = 0; lane < BLOCK_SIZE; lane++) begin : GEN_NVFP4_ENC_LANE
            wire [4:0]  ei       = abs_q1[lane][14:10];
            wire [9:0]  mi       = abs_q1[lane][9:0];
            wire        xi_zero  = (ei == 5'd0) || s_is_zero;
            wire [10:0] sig_i_q10 = xi_zero ? 11'd0 : {1'b1, mi}; // Q1.10

            // [HARDWARE OVERHEAD #1 PER LANE]: 11-bit x 8-bit Mantissa Multiplier
            //   sig_i_q10 (Q1.10) * recip_mant_q7 (Q1.7) -> prod_q17 (19-bit Q2.17)
            wire [18:0] prod_q17 = sig_i_q10 * recip_mant_q7;

            // Unbiased exponent difference: (ei - 15) - (es_q1 - 7) = ei - es_q1 - 8
            wire signed [6:0] exp_diff = $signed({2'b00, ei}) - $signed({3'b000, es_q1}) - 7'sd8;

            // [HARDWARE OVERHEAD #2 PER LANE]: Barrel Shifter to align Q2.17 to Q4.8
            //   At exp_diff == 0, Q2.17 >> 9 gives Q4.8.
            //   So net right shift = 9 - exp_diff.
            wire signed [6:0] rshift_signed = 7'sd9 - exp_diff;
            logic [11:0] y_q4_8;

            always_comb begin
                if (xi_zero || (rshift_signed > 7'sd18)) begin
                    y_q4_8 = 12'd0;
                end else if (rshift_signed < 7'sd0) begin
                    y_q4_8 = 12'hFFF; // Saturate to max E2M1 (6.0)
                end else begin
                    logic [18:0] shifted_tmp;
                    shifted_tmp = prod_q17 >> rshift_signed[4:0];
                    y_q4_8 = (|shifted_tmp[18:12]) ? 12'hFFF : shifted_tmp[11:0];
                end
            end

            wire [2:0] mag_lane = e2m1_mag_from_q4_8(y_q4_8);
            assign fp4_code_s2[lane] = (mag_lane == 3'b000) ? 4'b0000
                                                            : {sign_q1[lane], mag_lane};
        end
    endgenerate

    // Output Packet Register (matches slq_encoder.sv cycle-for-cycle)
    int out_i;
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            out_valid  <= 1'b0;
            packed_vec <= '0;
        end else if (pipe_advance) begin
            out_valid <= valid_q1;
            if (valid_q1) begin
                for (out_i = 0; out_i < BLOCK_SIZE; out_i++) begin
                    packed_vec[4*out_i +: 4] <= fp4_code_s2[out_i];
                end
                packed_vec[PACKET_WIDTH-1 -: 8] <= scale_e4m3_q1;
            end
        end
    end

endmodule
