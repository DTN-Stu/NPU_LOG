// =============================================================================
// Module: fp32_lane_backend.sv
// Description: In-loop FP32 Normalize + 2-Input FP32 Adder (Baseline Backend
//              matching MiX's MXFP4_g16_PE / NVFP4_PE FP32 accumulation loop,
//              without systolic pass-through registers).
// =============================================================================
`timescale 1ns/1ps

module fp32_norm_add_2in #(
    parameter int MAG_W       = 21, // Up to 21-bit magnitude (13b dot * 8b mant)
    parameter int EXP_IN_W    = 9,  // Combined exponent width
    parameter int EXP_BIAS_OFFSET = 127
)(
    input  logic                 prod_sign_i,
    input  logic [MAG_W-1:0]     prod_mag_i,
    input  logic [EXP_IN_W-1:0]  prod_exp_i,

    input  logic [31:0]          acc_old_fp32_i,
    output logic [31:0]          acc_new_fp32_o
);

    // -------------------------------------------------------------------------
    // 1. Normalize integer/scaled product to FP32 {sign, exp[7:0], mant[22:0]}
    // -------------------------------------------------------------------------
    logic        norm_sign;
    logic [7:0]  norm_exp;
    logic [22:0] norm_mant;

    int          lead_idx;
    logic [23:0] norm_mant_full;
    logic [8:0]  norm_exp_ext;

    always_comb begin
        norm_sign      = prod_sign_i;
        lead_idx       = -1;
        norm_mant_full = 24'd0;
        norm_exp_ext   = 9'd0;
        norm_exp       = 8'd0;
        norm_mant      = 23'd0;

        for (int i = MAG_W - 1; i >= 0; i--) begin
            if ((lead_idx == -1) && prod_mag_i[i]) begin
                lead_idx = i;
            end
        end

        if (lead_idx >= 0) begin
            if (lead_idx > 23) begin
                norm_mant_full = 24'(prod_mag_i >> (lead_idx - 23));
            end else begin
                norm_mant_full = 24'(prod_mag_i) << (23 - lead_idx);
            end
            norm_exp_ext = 9'(prod_exp_i) + 9'(lead_idx) + 9'(EXP_BIAS_OFFSET);
            if (norm_exp_ext[8]) begin
                norm_exp  = 8'hFF;
                norm_mant = 23'h7FFFFF;
            end else begin
                norm_exp  = norm_exp_ext[7:0];
                norm_mant = norm_mant_full[22:0];
            end
        end else begin
            norm_sign = 1'b0;
        end
    end

    // -------------------------------------------------------------------------
    // 2. 2-Input FP32 Adder (acc_old_fp32_i + {norm_sign, norm_exp, norm_mant})
    // -------------------------------------------------------------------------
    wire        a_sign = acc_old_fp32_i[31];
    wire [7:0]  a_exp  = acc_old_fp32_i[30:23];
    wire [22:0] a_mant = acc_old_fp32_i[22:0];

    wire        b_sign = norm_sign;
    wire [7:0]  b_exp  = norm_exp;
    wire [22:0] b_mant = norm_mant;

    logic [7:0]  exp_diff;
    logic        a_exp_larger;
    logic [7:0]  max_exp;
    logic [23:0] a_mant_ext, b_mant_ext;
    logic [23:0] a_aligned, b_aligned;
    logic signed [24:0] a_signed, b_signed;
    logic signed [25:0] sum_mant;
    logic [24:0] sum_abs;
    logic        res_sign;
    logic [7:0]  res_exp;
    logic [22:0] res_mant;
    int          add_lead;
    logic [23:0] add_norm_mant;
    logic [8:0]  add_norm_exp;

    always_comb begin
        a_exp_larger = (a_exp >= b_exp);
        max_exp      = a_exp_larger ? a_exp : b_exp;
        exp_diff     = a_exp_larger ? (a_exp - b_exp) : (b_exp - a_exp);

        a_mant_ext = (a_exp != 8'd0 || a_mant != 23'd0) ? {1'b1, a_mant} : 24'd0;
        b_mant_ext = (b_exp != 8'd0 || b_mant != 23'd0) ? {1'b1, b_mant} : 24'd0;

        if (a_exp_larger) begin
            a_aligned = a_mant_ext;
            b_aligned = (exp_diff >= 8'd24) ? 24'd0 : (b_mant_ext >> exp_diff);
        end else begin
            a_aligned = (exp_diff >= 8'd24) ? 24'd0 : (a_mant_ext >> exp_diff);
            b_aligned = b_mant_ext;
        end

        a_signed = a_sign ? -$signed({1'b0, a_aligned}) : $signed({1'b0, a_aligned});
        b_signed = b_sign ? -$signed({1'b0, b_aligned}) : $signed({1'b0, b_aligned});
        sum_mant = $signed({a_signed[24], a_signed}) + $signed({b_signed[24], b_signed});

        res_sign = sum_mant[25];
        sum_abs  = res_sign ? 25'(-sum_mant) : 25'(sum_mant);

        add_lead      = -1;
        add_norm_mant = 24'd0;
        add_norm_exp  = 9'd0;
        res_exp       = 8'd0;
        res_mant      = 23'd0;

        for (int i = 24; i >= 0; i--) begin
            if ((add_lead == -1) && sum_abs[i]) begin
                add_lead = i;
            end
        end

        if (add_lead >= 0) begin
            if (add_lead > 23) begin
                add_norm_mant = 24'(sum_abs >> (add_lead - 23));
                add_norm_exp  = 9'(max_exp) + 9'(add_lead - 23);
            end else begin
                add_norm_mant = 24'(sum_abs) << (23 - add_lead);
                if (max_exp >= 8'(23 - add_lead)) begin
                    add_norm_exp = 9'(max_exp) - 9'(23 - add_lead);
                end else begin
                    add_norm_exp = 9'd0;
                end
            end

            if (add_norm_exp[8]) begin
                res_exp  = 8'hFF;
                res_mant = 23'h7FFFFF;
            end else begin
                res_exp  = add_norm_exp[7:0];
                res_mant = add_norm_mant[22:0];
            end
        end else begin
            res_sign = 1'b0;
        end

        acc_new_fp32_o = {res_sign, res_exp, res_mant};
    end

endmodule
