// =============================================================================
// Module: snax_sfu_fp20_units.sv
// Description: Ultra-compact Custom FP20 ({sign[19], exp[18:11], mant[10:0]})
//              arithmetic units for snax_llm_sfu_opt:
//                - 8-bit exponent (bias 127) prevents x^2 overflow in RMSNorm
//                - 11-bit mantissa (10b FP16 + 1b guard) shrinks multipliers
//                  from 24x24 (FP32) down to 12x12 (~4x area reduction)
// =============================================================================
`timescale 1ns/1ps

module sfu_fp16_to_fp20 (
    input  logic [15:0] fp16_i,
    output logic [19:0] fp20_o
);
    wire        s = fp16_i[15];
    wire [4:0]  e = fp16_i[14:10];
    wire [9:0]  m = fp16_i[9:0];

    logic [3:0] lzc;
    logic       found;
    logic [9:0] m_norm;

    always_comb begin
        lzc   = 4'd0;
        found = 1'b0;
        m_norm = 10'd0;
        if (e == 5'd0) begin
            if (m == 10'd0) begin
                fp20_o = {s, 19'd0};
            end else begin
                for (int k = 9; k >= 0; k--) begin
                    if (!found && m[k]) begin
                        lzc   = 4'(9 - k);
                        found = 1'b1;
                    end
                end
                m_norm = m << (lzc + 1);
                fp20_o = {s, (8'd112 - {4'd0, lzc}), m_norm, 1'b0};
            end
        end else if (e == 5'h1F) begin
            fp20_o = {s, 8'hFF, m, 1'b0};
        end else begin
            fp20_o = {s, (8'(e) + 8'd112), m, 1'b0};
        end
    end
endmodule


module sfu_fp20_to_fp16 (
    input  logic [19:0] fp20_i,
    output logic [15:0] fp16_o
);
    wire        s = fp20_i[19];
    wire [7:0]  e = fp20_i[18:11];
    wire [10:0] m = fp20_i[10:0];

    logic signed [8:0]  e_unbiased, e_final;
    logic [10:0]        m_rounded;
    logic               round_up;

    always_comb begin
        e_unbiased = 9'sd0;
        e_final    = 9'sd0;
        m_rounded  = 11'd0;
        round_up   = 1'b0;
        fp16_o     = 16'd0;
        if (e == 8'hFF) begin
            fp16_o = {s, 5'h1F, (m != 11'd0) ? 10'h200 : 10'd0};
        end else if (e == 8'd0) begin
            fp16_o = {s, 15'd0};
        end else begin
            e_unbiased = $signed({1'b0, e}) - 9'sd112;
            round_up   = m[0] & m[1]; // RNE with 1 guard bit
            m_rounded  = {1'b0, m[10:1]} + {10'd0, round_up};
            e_final    = m_rounded[10] ? (e_unbiased + 9'sd1) : e_unbiased;
            if (e_final >= 9'sd31)     fp16_o = {s, 5'h1E, 10'h3FF};
            else if (e_final <= 9'sd0) fp16_o = {s, 15'd0};
            else                       fp16_o = {s, e_final[4:0], m_rounded[9:0]};
        end
    end
endmodule


module sfu_fp20_mul (
    input  logic [19:0] a_i,
    input  logic [19:0] b_i,
    output logic [19:0] res_o
);
    wire        res_s = a_i[19] ^ b_i[19];
    wire [7:0]  a_e   = a_i[18:11];
    wire [7:0]  b_e   = b_i[18:11];
    wire [11:0] a_m   = {1'b1, a_i[10:0]};
    wire [11:0] b_m   = {1'b1, b_i[10:0]};

    logic [23:0]       prod_m;
    logic signed [9:0] exp_sum;
    logic [10:0]       mant_out;

    always_comb begin
        prod_m   = 24'd0;
        exp_sum  = 10'sd0;
        mant_out = 11'd0;
        res_o    = 20'd0;
        if (a_e == 8'd0 || b_e == 8'd0) begin
            res_o = 20'd0;
        end else begin
            prod_m  = a_m * b_m; // 12b x 12b
            exp_sum = $signed({2'b00, a_e}) + $signed({2'b00, b_e}) - 10'sd127;
            if (prod_m[23]) begin
                exp_sum  = exp_sum + 10'sd1;
                mant_out = prod_m[22:12] + prod_m[11];
            end else begin
                mant_out = prod_m[21:11] + prod_m[10];
            end
            if (exp_sum <= 10'sd0)       res_o = 20'd0;
            else if (exp_sum >= 10'sd255) res_o = {res_s, 8'hFE, 11'h7FF};
            else                          res_o = {res_s, exp_sum[7:0], mant_out};
        end
    end
endmodule


module sfu_fp20_addsub (
    input  logic [19:0] a_i,
    input  logic [19:0] b_i,
    input  logic        sub_i,
    output logic [19:0] res_o
);
    wire        a_s = a_i[19];
    wire [7:0]  a_e = a_i[18:11];
    wire [10:0] a_m = a_i[10:0];

    wire        b_s = b_i[19] ^ sub_i;
    wire [7:0]  b_e = b_i[18:11];
    wire [10:0] b_m = b_i[10:0];

    logic        a_ge_b, lg_s, sm_s, eff_sub, res_s, found;
    logic [7:0]  lg_e, sm_e, exp_diff;
    logic [11:0] lg_m, sm_m;
    logic [14:0] lg_ext, sm_ext, sm_shifted, norm_m;
    logic [15:0] sum_m;
    logic [3:0]  lzc;
    logic signed [9:0] res_e;

    always_comb begin
        a_ge_b     = 1'b0;
        lg_s       = 1'b0;
        sm_s       = 1'b0;
        eff_sub    = 1'b0;
        res_s      = 1'b0;
        found      = 1'b0;
        lg_e       = 8'd0;
        sm_e       = 8'd0;
        exp_diff   = 8'd0;
        lg_m       = 12'd0;
        sm_m       = 12'd0;
        lg_ext     = 15'd0;
        sm_ext     = 15'd0;
        sm_shifted = 15'd0;
        norm_m     = 15'd0;
        sum_m      = 16'd0;
        lzc        = 4'd0;
        res_e      = 10'sd0;
        res_o      = 20'd0;
        if (a_e == 8'd0 && b_e == 8'd0) begin
            res_o = 20'd0;
        end else if (a_e == 8'd0) begin
            res_o = {b_s, b_e, b_m};
        end else if (b_e == 8'd0) begin
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
            sm_shifted = (exp_diff >= 8'd15) ? 15'd0 : (sm_ext >> exp_diff);
            eff_sub    = lg_s ^ sm_s;
            res_s      = lg_s;

            sum_m = eff_sub ? ({1'b0, lg_ext} - {1'b0, sm_shifted})
                            : ({1'b0, lg_ext} + {1'b0, sm_shifted});

            if (sum_m == 16'd0) begin
                res_o = 20'd0;
            end else if (sum_m[15]) begin
                res_e = $signed({2'b00, lg_e}) + 10'sd1;
                if (res_e >= 10'sd255) res_o = {res_s, 8'hFE, 11'h7FF};
                else                   res_o = {res_s, res_e[7:0], sum_m[14:4]};
            end else begin
                for (int k = 14; k >= 0; k--) begin
                    if (!found && sum_m[k]) begin
                        lzc   = 4'(14 - k);
                        found = 1'b1;
                    end
                end
                norm_m = sum_m[14:0] << lzc;
                res_e  = $signed({2'b00, lg_e}) - $signed({6'd0, lzc});
                if (res_e <= 10'sd0)       res_o = 20'd0;
                else if (res_e >= 10'sd255) res_o = {res_s, 8'hFE, 11'h7FF};
                else                        res_o = {res_s, res_e[7:0], norm_m[13:3]};
            end
        end
    end
endmodule


module sfu_rsqrt_fp20 (
    input  logic [19:0] x_i,
    output logic [19:0] inv_sqrt_o
);
    wire       e_lsb = x_i[11];
    wire [3:0] m_idx = x_i[10:7];

    logic signed [8:0] k_unbiased, y0_exp_s;
    logic [7:0]        lut_exp;
    logic [10:0]       lut_mant;

    always_comb begin
        k_unbiased = $signed({1'b0, x_i[18:11]}) - 9'sd127;
        if (e_lsb) begin
            y0_exp_s = 9'sd127 - (k_unbiased >>> 1) - ((m_idx == 4'd0) ? 9'sd0 : 9'sd1);
        end else begin
            y0_exp_s = 9'sd127 - ((k_unbiased - 9'sd1) >>> 1) - 9'sd1;
        end
        lut_exp = (y0_exp_s <= 9'sd0) ? 8'd1 : (y0_exp_s >= 9'sd254 ? 8'd254 : y0_exp_s[7:0]);
    end

    always_comb begin
        if (e_lsb) begin
            case (m_idx)
                4'd0:  lut_mant = 11'h000;
                4'd1:  lut_mant = 11'h737;
                4'd2:  lut_mant = 11'h680;
                4'd3:  lut_mant = 11'h5E4;
                4'd4:  lut_mant = 11'h559;
                4'd5:  lut_mant = 11'h4DD;
                4'd6:  lut_mant = 11'h46F;
                4'd7:  lut_mant = 11'h40C;
                4'd8:  lut_mant = 11'h3B3;
                4'd9:  lut_mant = 11'h363;
                4'd10: lut_mant = 11'h31A;
                4'd11: lut_mant = 11'h2D9;
                4'd12: lut_mant = 11'h29D;
                4'd13: lut_mant = 11'h267;
                4'd14: lut_mant = 11'h236;
                4'd15: lut_mant = 11'h20A;
            endcase
        end else begin
            case (m_idx)
                4'd0:  lut_mant = 11'h324;
                4'd1:  lut_mant = 11'h2BB;
                4'd2:  lut_mant = 11'h25A;
                4'd3:  lut_mant = 11'h201;
                4'd4:  lut_mant = 11'h1AE;
                4'd5:  lut_mant = 11'h163;
                4'd6:  lut_mant = 11'h11C;
                4'd7:  lut_mant = 11'h0DA;
                4'd8:  lut_mant = 11'h09D;
                4'd9:  lut_mant = 11'h063;
                4'd10: lut_mant = 11'h02D;
                4'd11: lut_mant = 11'h0C0;
                4'd12: lut_mant = 11'h090;
                4'd13: lut_mant = 11'h060;
                4'd14: lut_mant = 11'h035;
                4'd15: lut_mant = 11'h010;
            endcase
        end
    end

    wire [19:0] y0     = {1'b0, lut_exp, lut_mant};
    wire [19:0] half_x = {1'b0, (x_i[18:11] > 8'd1 ? (x_i[18:11] - 8'd1) : 8'd0), x_i[10:0]};
    wire [19:0] y0_sq, half_x_y0_sq, nr_factor, y1;

    sfu_fp20_mul    u_sq  (.a_i(y0),     .b_i(y0),           .res_o(y0_sq));
    sfu_fp20_mul    u_hx  (.a_i(half_x), .b_i(y0_sq),        .res_o(half_x_y0_sq));
    sfu_fp20_addsub u_sub (.a_i(20'h3FC00), .b_i(half_x_y0_sq), .sub_i(1'b1), .res_o(nr_factor)); // 1.5
    sfu_fp20_mul    u_y1  (.a_i(y0),     .b_i(nr_factor),    .res_o(y1));

    assign inv_sqrt_o = (x_i[18:11] == 8'd0) ? 20'd0 : y1;
endmodule


// =============================================================================
// Module: sfu_sigmoid_pwl_fp20
// Description: Multiplier-less PWL sigmoid(x) using power-of-2 slopes in FP20
//              (exponent-shift slopes: 0.25, 0.125, 0.03125 -> ZERO multipliers!)
//              - [0, 1.0):   0.500 + 0.25 * x       (exp - 2)
//              - [1.0, 2.5): 0.731 + 0.125 * (x-1)  (exp - 3)
//              - [2.5, 4.5): 0.924 + 0.03125*(x-2.5)(exp - 5)
//              - [4.5, inf): 1.0
// =============================================================================
module sfu_sigmoid_pwl_fp20 (
    input  logic [19:0] x_i,
    output logic [19:0] sig_o
);
    wire        x_sign = x_i[19];
    wire [19:0] abs_x  = {1'b0, x_i[18:0]};

    localparam logic [19:0] FP20_ZERO = 20'h00000;
    localparam logic [19:0] FP20_ONE  = 20'h3F800; // 1.0
    localparam logic [19:0] FP20_2P5  = 20'h40200; // 2.5
    localparam logic [19:0] FP20_4P5  = 20'h40900; // 4.5

    localparam logic [19:0] B0 = 20'h3F000; // 0.500
    localparam logic [19:0] B1 = 20'h3F3B2; // 0.731
    localparam logic [19:0] B2 = 20'h3F6C9; // 0.924

    logic [19:0] x_off, intercept;
    logic [2:0]  rshift_exp;

    always_comb begin
        if (abs_x < FP20_ONE) begin
            x_off      = FP20_ZERO;
            intercept  = B0;
            rshift_exp = 3'd2; // * 0.25
        end else if (abs_x < FP20_2P5) begin
            x_off      = FP20_ONE;
            intercept  = B1;
            rshift_exp = 3'd3; // * 0.125
        end else if (abs_x < FP20_4P5) begin
            x_off      = FP20_2P5;
            intercept  = B2;
            rshift_exp = 3'd5; // * 0.03125
        end else begin
            x_off      = FP20_ZERO;
            intercept  = FP20_ONE;
            rshift_exp = 3'd0;
        end
    end

    wire [19:0] dx;
    sfu_fp20_addsub u_dx (.a_i(abs_x), .b_i(x_off), .sub_i(1'b1), .res_o(dx));

    wire [19:0] dx_scaled = (dx[18:11] <= {5'd0, rshift_exp} || abs_x >= FP20_4P5)
                          ? 20'd0
                          : {dx[19], (dx[18:11] - {5'd0, rshift_exp}), dx[10:0]};

    wire [19:0] sig_pos, sig_neg;
    sfu_fp20_addsub u_pos (.a_i(intercept), .b_i(dx_scaled), .sub_i(1'b0), .res_o(sig_pos));
    sfu_fp20_addsub u_neg (.a_i(FP20_ONE),  .b_i(sig_pos),   .sub_i(1'b1), .res_o(sig_neg));

    assign sig_o = (abs_x >= FP20_4P5) ? (x_sign ? FP20_ZERO : FP20_ONE)
                                       : (x_sign ? sig_neg   : sig_pos);
endmodule
