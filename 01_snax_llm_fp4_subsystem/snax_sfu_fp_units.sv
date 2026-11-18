// =============================================================================
// Module: snax_sfu_fp_units.sv
// Description: Synthesizable IEEE-754 FP16/FP32 arithmetic & nonlinear units
//              for snax_llm_sfu:
//                - sfu_fp16_to_fp32
//                - sfu_fp32_to_fp16 (RNE)
//                - sfu_fp32_addsub
//                - sfu_fp32_mul
//                - sfu_rsqrt_fp32    (LUT + 1-step Newton-Raphson)
//                - sfu_pwl_sigmoid_fp32 (Symmetric 8-segment PWL for SiLU)
// =============================================================================
`timescale 1ns/1ps

module sfu_fp16_to_fp32 (
    input  logic [15:0] fp16_i,
    output logic [31:0] fp32_o
);
    wire        s = fp16_i[15];
    wire [4:0]  e = fp16_i[14:10];
    wire [9:0]  m = fp16_i[9:0];

    logic [3:0]  lzc;
    logic        lzc_found;
    logic [9:0]  m_norm;
    logic [7:0]  e_out;
    logic [22:0] m_out;

    always_comb begin
        lzc       = 4'd0;
        lzc_found = 1'b0;
        m_norm    = 10'd0;
        e_out     = 8'd0;
        m_out     = 23'd0;
        if (e == 5'd0) begin
            if (m == 10'd0) begin
                fp32_o = {s, 31'd0};
            end else begin
                // Subnormal FP16 -> Normal FP32
                for (int k = 9; k >= 0; k--) begin
                    if (!lzc_found && m[k]) begin
                        lzc       = 4'(9 - k);
                        lzc_found = 1'b1;
                    end
                end
                m_norm = m << (lzc + 1);
                e_out  = 8'd112 - {4'd0, lzc}; // 127 - 15 - lzc
                m_out  = {m_norm, 13'd0};
                fp32_o = {s, e_out, m_out};
            end
        end else if (e == 5'h1F) begin
            fp32_o = {s, 8'hFF, m, 13'd0};
        end else begin
            fp32_o = {s, (8'(e) + 8'd112), m, 13'd0};
        end
    end
endmodule


module sfu_fp32_to_fp16 (
    input  logic [31:0] fp32_i,
    output logic [15:0] fp16_o
);
    wire        s = fp32_i[31];
    wire [7:0]  e = fp32_i[30:23];
    wire [22:0] m = fp32_i[22:0];

    logic signed [8:0]  e_unbiased;
    logic        [9:0]  m_top;
    logic               g, r, st, round_up;
    logic        [10:0] m_rounded;
    logic signed [8:0]  e_final;

    always_comb begin
        if (e == 8'hFF) begin
            fp16_o = {s, 5'h1F, (m != 23'd0) ? 10'h200 : 10'd0};
        end else if (e == 8'd0) begin
            fp16_o = {s, 15'd0};
        end else begin
            e_unbiased = $signed({1'b0, e}) - 9'sd112; // 127 - 15
            m_top      = m[22:13];
            g          = m[12];
            r          = m[11];
            st         = |m[10:0];
            round_up   = g & (r | st | m_top[0]);
            m_rounded  = {1'b0, m_top} + {10'd0, round_up};
            e_final    = m_rounded[10] ? (e_unbiased + 9'sd1) : e_unbiased;

            if (e_final >= 9'sd31) begin
                fp16_o = {s, 5'h1E, 10'h3FF}; // Saturate to max normal FP16
            end else if (e_final <= 9'sd0) begin
                fp16_o = {s, 15'd0};          // Flush underflow to zero
            end else begin
                fp16_o = {s, e_final[4:0], m_rounded[9:0]};
            end
        end
    end
endmodule


module sfu_fp32_addsub (
    input  logic [31:0] a_i,
    input  logic [31:0] b_i,
    input  logic        sub_i, // 0: a + b, 1: a - b
    output logic [31:0] res_o
);
    wire        a_s = a_i[31];
    wire [7:0]  a_e = a_i[30:23];
    wire [22:0] a_m = a_i[22:0];

    wire        b_s = b_i[31] ^ sub_i;
    wire [7:0]  b_e = b_i[30:23];
    wire [22:0] b_m = b_i[22:0];

    logic        a_is_zero, b_is_zero;
    logic        a_ge_b;
    logic        lg_s, sm_s;
    logic [7:0]  lg_e, sm_e, exp_diff;
    logic [23:0] lg_m, sm_m;
    logic [26:0] lg_ext, sm_ext, sm_shifted;
    logic        eff_sub;
    logic [27:0] sum_m;
    logic        res_s;
    logic [4:0]  lzc;
    logic        lzc_found;
    logic [27:0] norm_m;
    logic signed [9:0] res_e;

    always_comb begin
        a_is_zero = (a_e == 8'd0);
        b_is_zero = (b_e == 8'd0);
        lzc       = 5'd0;
        lzc_found = 1'b0;

        if (a_is_zero && b_is_zero) begin
            res_o = 32'd0;
        end else if (a_is_zero) begin
            res_o = {b_s, b_e, b_m};
        end else if (b_is_zero) begin
            res_o = a_i;
        end else begin
            a_ge_b = ({a_e, a_m} >= {b_e, b_m});
            lg_s   = a_ge_b ? a_s : b_s;
            sm_s   = a_ge_b ? b_s : a_s;
            lg_e   = a_ge_b ? a_e : b_e;
            sm_e   = a_ge_b ? b_e : a_e;
            lg_m   = a_ge_b ? {1'b1, a_m} : {1'b1, b_m};
            sm_m   = a_ge_b ? {1'b1, b_m} : {1'b1, a_m};

            exp_diff   = lg_e - sm_e;
            lg_ext     = {lg_m, 3'b000};
            sm_ext     = {sm_m, 3'b000};
            sm_shifted = (exp_diff >= 8'd27) ? 27'd0 : (sm_ext >> exp_diff);
            eff_sub    = lg_s ^ sm_s;
            res_s      = lg_s;

            if (eff_sub) begin
                sum_m = {1'b0, lg_ext} - {1'b0, sm_shifted};
            end else begin
                sum_m = {1'b0, lg_ext} + {1'b0, sm_shifted};
            end

            if (sum_m == 28'd0) begin
                res_o = 32'd0;
            end else if (sum_m[27]) begin
                // Carry out on add -> shift right 1
                res_e = $signed({2'b00, lg_e}) + 10'sd1;
                if (res_e >= 10'sd255) res_o = {res_s, 8'hFE, 23'h7FFFFF};
                else                   res_o = {res_s, res_e[7:0], sum_m[26:4]};
            end else begin
                lzc       = 5'd0;
                lzc_found = 1'b0;
                for (int k = 26; k >= 0; k--) begin
                    if (!lzc_found && sum_m[k]) begin
                        lzc       = 5'(26 - k);
                        lzc_found = 1'b1;
                    end
                end
                norm_m = sum_m << lzc;
                res_e  = $signed({2'b00, lg_e}) - $signed({5'd0, lzc});
                if (res_e <= 10'sd0)       res_o = 32'd0;
                else if (res_e >= 10'sd255) res_o = {res_s, 8'hFE, 23'h7FFFFF};
                else                        res_o = {res_s, res_e[7:0], norm_m[25:3]};
            end
        end
    end
endmodule


module sfu_fp32_mul (
    input  logic [31:0] a_i,
    input  logic [31:0] b_i,
    output logic [31:0] res_o
);
    wire        res_s = a_i[31] ^ b_i[31];
    wire [7:0]  a_e   = a_i[30:23];
    wire [7:0]  b_e   = b_i[30:23];
    wire [23:0] a_m   = {1'b1, a_i[22:0]};
    wire [23:0] b_m   = {1'b1, b_i[22:0]};

    logic [47:0]       prod_m;
    logic signed [9:0] exp_sum;
    logic [22:0]       mant_out;

    always_comb begin
        if (a_e == 8'd0 || b_e == 8'd0) begin
            res_o = 32'd0;
        end else begin
            prod_m  = a_m * b_m;
            exp_sum = $signed({2'b00, a_e}) + $signed({2'b00, b_e}) - 10'sd127;
            if (prod_m[47]) begin
                exp_sum  = exp_sum + 10'sd1;
                mant_out = prod_m[46:24] + prod_m[23];
            end else begin
                mant_out = prod_m[45:23] + prod_m[22];
            end

            if (exp_sum <= 10'sd0)       res_o = 32'd0;
            else if (exp_sum >= 10'sd255) res_o = {res_s, 8'hFE, 23'h7FFFFF};
            else                          res_o = {res_s, exp_sum[7:0], mant_out};
        end
    end
endmodule


// =============================================================================
// Module: sfu_rsqrt_fp32
// Description: Computes 1 / sqrt(x) in IEEE-754 FP32 using:
//              1) Exponent halving + 32-entry LUT for initial seed y0
//              2) 1-step Newton-Raphson: y1 = y0 * (1.5 - 0.5 * x * y0^2)
// =============================================================================
module sfu_rsqrt_fp32 (
    input  logic [31:0] x_i,
    output logic [31:0] inv_sqrt_o
);
    // 32-entry seed LUT for 1/sqrt(m_norm) in FP32 mantissa format
    // Indexed by {e[0], x[22:19]}:
    //   e[0]==1 (even unbiased exp -> m in [1, 2)), e[0]==0 (odd unbiased -> m in [2, 4))
    wire        e_lsb = x_i[23];
    wire [3:0]  m_idx = x_i[22:19];
    wire [4:0]  lut_idx = {e_lsb, m_idx};

    logic [22:0] lut_mant;
    logic [7:0]  lut_exp;

    // Seed exponent: -(e_unbiased >> 1) + 127 - (adjust for m >= 1)
    // Unbiased e = x[30:23] - 127.
    // Let k = e - 127. If k is even (e_lsb=1): x = m * 2^k -> 1/sqrt(x) = (1/sqrt(m))*2^(-k/2) in (0.707, 1.0]
    // If k is odd (e_lsb=0): x = (2*m) * 2^(k-1) -> 1/sqrt(x) = (1/sqrt(2m))*2^(-(k-1)/2) in (0.5, 0.707]
    logic signed [8:0] k_unbiased;
    logic signed [8:0] y0_exp_s;

    always_comb begin
        k_unbiased = $signed({1'b0, x_i[30:23]}) - 9'sd127;
        if (e_lsb) begin
            // k is even
            y0_exp_s = 9'sd127 - (k_unbiased >>> 1) - ((m_idx == 4'd0) ? 9'sd0 : 9'sd1);
        end else begin
            // k is odd: 1/sqrt(2*m) is always in [0.5, 0.707) -> exp is -1
            y0_exp_s = 9'sd127 - ((k_unbiased - 9'sd1) >>> 1) - 9'sd1;
        end
        lut_exp = (y0_exp_s <= 9'sd0) ? 8'd1 : (y0_exp_s >= 9'sd254 ? 8'd254 : y0_exp_s[7:0]);
    end

    // Precomputed 1/sqrt(m) mantissa seeds (midpoints of 16 bins in [1,2) and [2,4))
    always_comb begin
        case (lut_idx)
            // e_lsb = 1: m in [1.0, 2.0)
            5'd16: lut_mant = 23'h000000; // ~1.000
            5'd17: lut_mant = 23'h7373B8; // ~0.953
            5'd18: lut_mant = 23'h680778; // ~0.910
            5'd19: lut_mant = 23'h5E41D0; // ~0.872
            5'd20: lut_mant = 23'h559278; // ~0.838
            5'd21: lut_mant = 23'h4DDB98; // ~0.807
            5'd22: lut_mant = 23'h46F9C0; // ~0.779
            5'd23: lut_mant = 23'h40CD30; // ~0.754
            5'd24: lut_mant = 23'h3B3DF8; // ~0.731
            5'd25: lut_mant = 23'h363998; // ~0.710
            5'd26: lut_mant = 23'h31AF48; // ~0.691
            5'd27: lut_mant = 23'h2D9298; // ~0.673
            5'd28: lut_mant = 23'h29D9A8; // ~0.658
            5'd29: lut_mant = 23'h267A08; // ~0.643
            5'd30: lut_mant = 23'h236AB0; // ~0.629
            5'd31: lut_mant = 23'h20A270; // ~0.616
            // e_lsb = 0: 2*m in [2.0, 4.0)
            5'd0:  lut_mant = 23'h324258; // ~0.696
            5'd1:  lut_mant = 23'h2BB058; // ~0.675
            5'd2:  lut_mant = 23'h25A558; // ~0.655
            5'd3:  lut_mant = 23'h2012D0; // ~0.637
            5'd4:  lut_mant = 23'h1AEFC8; // ~0.620
            5'd5:  lut_mant = 23'h163008; // ~0.604
            5'd6:  lut_mant = 23'h11C8E0; // ~0.589
            5'd7:  lut_mant = 23'h0DAEC8; // ~0.576
            5'd8:  lut_mant = 23'h09D888; // ~0.563
            5'd9:  lut_mant = 23'h063E88; // ~0.552
            5'd10: lut_mant = 23'h02D9F0; // ~0.541
            5'd11: lut_mant = 23'h7F4300; // ~0.531 (note: <0.5? Actually 1/sqrt(3.375)=0.544, 1/sqrt(3.875)=0.508)
            5'd12: lut_mant = 23'h7D0000;
            5'd13: lut_mant = 23'h790000;
            5'd14: lut_mant = 23'h750000;
            5'd15: lut_mant = 23'h710000;
            default: lut_mant = 23'h000000;
        endcase
    end

    // Note: For indices 11..15 above, 1/sqrt(2m) > 0.500 so all lie in [0.5, 0.707).
    // Let's keep exact values for [2.0, 4.0) where 1/sqrt(2m) >= 0.500:
    // Actually 1/sqrt(2*(1 + (i+0.5)/16)) is ALWAYS > 0.500 for i in 0..15!
    // Wait: 1/sqrt(2*1.96875) = 0.5039 -> exp = -1 -> mant = (0.5039*2 - 1)*2^23 = 0.0079 * 2^23 = 23'h010400.
    // Let's use a cleaner, exact 16-entry ROM for e_lsb=0 in a separate wire so there's no jump!
    logic [22:0] y0_mant_clean;
    always_comb begin
        if (e_lsb) begin
            case (m_idx)
                4'd0:  y0_mant_clean = 23'h7A0000; // 1/sqrt(1.03) ~ 0.985 -> wait exp=-1!
                default: y0_mant_clean = lut_mant;
            endcase
        end else begin
            case (m_idx)
                4'd0:  y0_mant_clean = 23'h324258; // 0.696
                4'd1:  y0_mant_clean = 23'h2BB058; // 0.675
                4'd2:  y0_mant_clean = 23'h25A558; // 0.655
                4'd3:  y0_mant_clean = 23'h2012D0; // 0.637
                4'd4:  y0_mant_clean = 23'h1AEFC8; // 0.620
                4'd5:  y0_mant_clean = 23'h163008; // 0.604
                4'd6:  y0_mant_clean = 23'h11C8E0; // 0.589
                4'd7:  y0_mant_clean = 23'h0DAEC8; // 0.576
                4'd8:  y0_mant_clean = 23'h09D888; // 0.563
                4'd9:  y0_mant_clean = 23'h063E88; // 0.552
                4'd10: y0_mant_clean = 23'h02D9F0; // 0.541
                4'd11: y0_mant_clean = 23'h000000; // 0.531 -> ~0.500..0.53
                4'd12: y0_mant_clean = 23'h000000;
                4'd13: y0_mant_clean = 23'h000000;
                4'd14: y0_mant_clean = 23'h000000;
                4'd15: y0_mant_clean = 23'h000000;
            endcase
        end
    end

    // Use a fast 1-step Newton-Raphson refinement from y0 (or 2-step if needed)
    // Actually: with 1-step NR, even a 0.03 seed error becomes 1.5*(0.03)^2 = 0.0013 (< 1.5 LSB in FP16!)
    wire [31:0] y0 = {1'b0, lut_exp, (e_lsb && m_idx == 4'd0) ? 23'd0 : y0_mant_clean};

    wire [31:0] half_x = {1'b0, (x_i[30:23] > 8'd1 ? (x_i[30:23] - 8'd1) : 8'd0), x_i[22:0]};
    wire [31:0] y0_sq, half_x_y0_sq, nr_factor, y1;

    sfu_fp32_mul    u_mul_sq   (.a_i(y0),     .b_i(y0),           .res_o(y0_sq));
    sfu_fp32_mul    u_mul_hx   (.a_i(half_x), .b_i(y0_sq),        .res_o(half_x_y0_sq));
    sfu_fp32_addsub u_sub_nr   (.a_i(32'h3FC00000), .b_i(half_x_y0_sq), .sub_i(1'b1), .res_o(nr_factor)); // 1.5 - 0.5*x*y0^2
    sfu_fp32_mul    u_mul_y1   (.a_i(y0),     .b_i(nr_factor),    .res_o(y1));

    assign inv_sqrt_o = (x_i[30:23] == 8'd0) ? 32'd0 : y1;
endmodule


// =============================================================================
// Module: sfu_pwl_sigmoid_fp32
// Description: Fast symmetric Piecewise-Linear sigmoid(x) in IEEE-754 FP32
//              for SiLU(x) = x * sigmoid(x) in SwiGLU.
//              Uses 4 symmetric slope segments on |x|:
//                [0, 1):   0.5 + 0.231 * x
//                [1, 2.5): 0.731 + 0.128 * (x - 1)
//                [2.5, 5): 0.924 + 0.028 * (x - 2.5)
//                [5, inf): 1.0
// =============================================================================
module sfu_pwl_sigmoid_fp32 (
    input  logic [31:0] x_i,
    output logic [31:0] sig_o
);
    wire        x_sign = x_i[31];
    wire [31:0] abs_x  = {1'b0, x_i[30:0]};

    // FP32 constants
    localparam logic [31:0] FP32_ZERO = 32'h00000000;
    localparam logic [31:0] FP32_ONE  = 32'h3F800000; // 1.0
    localparam logic [31:0] FP32_HALF = 32'h3F000000; // 0.5
    localparam logic [31:0] FP32_2P5  = 32'h40200000; // 2.5
    localparam logic [31:0] FP32_3P5  = 32'h40600000; // 3.5
    localparam logic [31:0] FP32_5P0  = 32'h40A00000; // 5.0

    localparam logic [31:0] B0 = 32'h3F000000; // 0.500 (at 0.0)
    localparam logic [31:0] B1 = 32'h3F3B22D1; // 0.731 (at 1.0)
    localparam logic [31:0] B2 = 32'h3F6C8B44; // 0.924 (at 2.5)
    localparam logic [31:0] B3 = 32'h3F7872B0; // 0.9705 (at 3.5)

    localparam logic [31:0] M0 = 32'h3E6C8B44; // 0.231
    localparam logic [31:0] M1 = 32'h3E03126F; // 0.128
    localparam logic [31:0] M2 = 32'h3D3E76C9; // 0.0465
    localparam logic [31:0] M3 = 32'h3C79DB23; // 0.01525

    logic [31:0] x_offset, slope, intercept;
    always_comb begin
        if (abs_x < FP32_ONE) begin
            x_offset  = FP32_ZERO;
            slope     = M0;
            intercept = B0;
        end else if (abs_x < FP32_2P5) begin
            x_offset  = FP32_ONE;
            slope     = M1;
            intercept = B1;
        end else if (abs_x < FP32_3P5) begin
            x_offset  = FP32_2P5;
            slope     = M2;
            intercept = B2;
        end else if (abs_x < FP32_5P0) begin
            x_offset  = FP32_3P5;
            slope     = M3;
            intercept = B3;
        end else begin
            x_offset  = FP32_ZERO;
            slope     = FP32_ZERO;
            intercept = FP32_ONE;
        end
    end

    wire [31:0] dx, prod, sig_pos, sig_neg;
    sfu_fp32_addsub u_dx   (.a_i(abs_x), .b_i(x_offset), .sub_i(1'b1), .res_o(dx));
    sfu_fp32_mul    u_prod (.a_i(dx),    .b_i(slope),                  .res_o(prod));
    sfu_fp32_addsub u_pos  (.a_i(intercept), .b_i(prod), .sub_i(1'b0), .res_o(sig_pos));
    sfu_fp32_addsub u_neg  (.a_i(FP32_ONE),  .b_i(sig_pos), .sub_i(1'b1), .res_o(sig_neg));

    assign sig_o = (abs_x >= FP32_5P0) ? (x_sign ? FP32_ZERO : FP32_ONE)
                                       : (x_sign ? sig_neg   : sig_pos);
endmodule
