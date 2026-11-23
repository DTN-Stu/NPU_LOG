// =============================================================================
// Module: log8_fp4_arith_units.sv
// Description: Shared arithmetic primitives for Log8-FP4 MAC Array,
//              final_normalizer, and ablation baselines:
//                - logscale_lazy_drain_norm : 32-bit Q16.16 -> IEEE 754 FP16
//                - scale_adder              : 8-bit Log8 exponent split (q, r)
//                - logscale_frac_scaler     : 3-term CSD multiplier (m_r / 32)
//                - bidirectional_shifter    : Q7 -> Q16.16 octave barrel shifter
//                - accum_adder              : 32-bit saturating signed adder
// =============================================================================
`timescale 1ns/1ps
`include "pe_defines.svh"

// =============================================================================
// Module: logscale_lazy_drain_norm (Operand-Isolated Lazy FP16 Drain Normalizer)
// =============================================================================
module logscale_lazy_drain_norm #(
    parameter int FRAC_BITS = 16
)(
    input  logic signed [31:0] norm_in,
    output logic        [15:0] fp16_out
);

    logic        fp_sign;
    logic [31:0] abs_y;

    assign fp_sign = norm_in[31];
    assign abs_y   = fp_sign ? unsigned'(-norm_in) : unsigned'(norm_in);

    // Hierarchical 32-bit LZC (4 x 8-bit chunks)
    logic [3:0][2:0] lzc_sub;
    logic [3:0] sub_has_one;

    genvar g;
    generate
        for (g = 0; g < 4; g++) begin : gen_lzc_32
            wire [7:0] chunk = abs_y[g*8 +: 8];
            assign sub_has_one[g] = |chunk;
            always_comb begin
                casez (chunk)
                    8'b1???????: lzc_sub[g] = 3'd0;
                    8'b01??????: lzc_sub[g] = 3'd1;
                    8'b001?????: lzc_sub[g] = 3'd2;
                    8'b0001????: lzc_sub[g] = 3'd3;
                    8'b00001???: lzc_sub[g] = 3'd4;
                    8'b000001??: lzc_sub[g] = 3'd5;
                    8'b0000001?: lzc_sub[g] = 3'd6;
                    default:     lzc_sub[g] = 3'd7;
                endcase
            end
        end
    endgenerate

    logic [5:0] lzc;
    always_comb begin
        if      (sub_has_one[3]) lzc = {1'b0, 2'd0, lzc_sub[3]};
        else if (sub_has_one[2]) lzc = {1'b0, 2'd1, lzc_sub[2]};
        else if (sub_has_one[1]) lzc = {1'b0, 2'd2, lzc_sub[1]};
        else if (sub_has_one[0]) lzc = {1'b0, 2'd3, lzc_sub[0]};
        else                     lzc = 6'd32;
    end

    logic signed [7:0] raw_exp;
    assign raw_exp = 8'sd31 - $signed({2'b00, lzc}) - 8'(FRAC_BITS) + 8'sd15;

    logic [31:0] norm_shifted;
    logic [9:0]  mant_bits;
    logic        guard_bit, round_bit, sticky_bit;

    assign norm_shifted = abs_y << lzc[4:0];
    assign mant_bits    = norm_shifted[30:21];
    assign guard_bit    = norm_shifted[20];
    assign round_bit    = norm_shifted[19];
    assign sticky_bit   = |norm_shifted[18:0];

    logic        round_up;
    logic [10:0] rounded_mant;
    logic signed [7:0] final_exp;

    assign round_up     = guard_bit & (round_bit | sticky_bit | mant_bits[0]);
    assign rounded_mant = {1'b0, mant_bits} + {10'd0, round_up};
    assign final_exp    = rounded_mant[10] ? (raw_exp + 8'sd1) : raw_exp;

    always_comb begin
        if (abs_y == 32'd0 || final_exp <= 8'sd0) begin
            fp16_out = 16'h0000;
        end else if (final_exp >= 8'sd31) begin
            fp16_out = {fp_sign, 5'b11110, 10'h3FF};
        end else begin
            fp16_out = {fp_sign, final_exp[4:0], rounded_mant[9:0]};
        end
    end

endmodule


// =============================================================================
// Module: scale_adder — log-scale exponent combine (8-bit signed, eighths)
// =============================================================================
module scale_adder(
    input  logic signed [7:0]  scale_a,
    input  logic signed [7:0]  scale_w,
    output logic signed [5:0]  q,
    output logic        [2:0]  r
);

    wire signed [8:0] res = scale_a + scale_w;
    assign q = res[8:3];
    assign r = res[2:0];

endmodule


// =============================================================================
// Module: logscale_frac_scaler — P x m_r / 32 via 3-term CSD (Q2 x Q5 = Q7)
// =============================================================================
module logscale_frac_scaler (
    input  logic signed [12:0] prod_in,
    input  logic        [2:0]  r,
    output logic signed [18:0] scaled_prod
);

    logic signed [18:0] p_ext;
    assign p_ext = {{6{prod_in[12]}}, prod_in};

    logic [18:0] t0;
    assign t0 = (r[2] & r[1]) ? (p_ext <<< 6) : (p_ext <<< 5);

    logic [18:0] t1_mag, t2_mag;
    logic        sub_t1, sub_t2;

    always_comb begin
        unique case (r)
            3'd0: begin t1_mag = 19'd0;         sub_t1 = 1'b0; t2_mag = 19'd0;         sub_t2 = 1'b0; end
            3'd1: begin t1_mag = (p_ext <<< 1); sub_t1 = 1'b0; t2_mag = p_ext;         sub_t2 = 1'b0; end
            3'd2: begin t1_mag = (p_ext <<< 2); sub_t1 = 1'b0; t2_mag = (p_ext <<< 1); sub_t2 = 1'b0; end
            3'd3: begin t1_mag = (p_ext <<< 3); sub_t1 = 1'b0; t2_mag = p_ext;         sub_t2 = 1'b0; end
            3'd4: begin t1_mag = (p_ext <<< 4); sub_t1 = 1'b0; t2_mag = (p_ext <<< 1); sub_t2 = 1'b1; end
            3'd5: begin t1_mag = (p_ext <<< 4); sub_t1 = 1'b0; t2_mag = p_ext;         sub_t2 = 1'b0; end
            3'd6: begin t1_mag = (p_ext <<< 3); sub_t1 = 1'b1; t2_mag = (p_ext <<< 1); sub_t2 = 1'b1; end
            3'd7: begin t1_mag = (p_ext <<< 2); sub_t1 = 1'b1; t2_mag = p_ext;         sub_t2 = 1'b1; end
        endcase
    end

    logic [18:0] t1, t2;
    assign t1 = sub_t1 ? (~t1_mag) : t1_mag;
    assign t2 = sub_t2 ? (~t2_mag) : t2_mag;

    logic [18:0] csa_sum, csa_cout;
    assign csa_sum  = t0 ^ t1 ^ t2;
    assign csa_cout = (t0 & t1) | (t1 & t2) | (t0 & t2);

    logic [18:0] csa_carry_shifted;
    assign csa_carry_shifted = {csa_cout[17:0], sub_t1};

    assign scaled_prod = $signed(csa_sum + csa_carry_shifted + 19'(sub_t2));

endmodule


// =============================================================================
// Module: bidirectional_shifter — Q7 (19b) x 2^net_shift, net_shift = q + (FRAC-7)
// =============================================================================
module bidirectional_shifter #(
    parameter int FRAC_BITS = 16
)(
    input  logic signed [18:0] scaled_prod,
    input  logic signed [5:0]  q,
    output logic signed [31:0] aligned_prod_32
);

    localparam int BASE_SHIFT = FRAC_BITS - 7;

    logic signed [6:0] net_shift;
    assign net_shift = $signed({q[5], q}) + 7'(signed'(BASE_SHIFT));

    logic [3:0]         lshift_amt;
    logic [4:0]         rshift_amt;
    logic signed [33:0] lshifted_wide;
    logic signed [18:0] rshifted_narrow;

    logic signed [33:0] wide16;
    logic               sat_pos_hi;
    logic               sat_neg_hi;

    assign lshift_amt      = net_shift[3:0];
    assign rshift_amt      = (-net_shift > 7'sd19) ? 5'd19 : unsigned'(-net_shift[4:0]);
    assign lshifted_wide   = 34'(scaled_prod) <<< lshift_amt;
    assign rshifted_narrow = scaled_prod >>> rshift_amt;

    assign wide16     = 34'(scaled_prod) <<< net_shift[3:0];
    assign sat_pos_hi = ~wide16[33] & (|wide16[32:15]);
    assign sat_neg_hi =  wide16[33] & ~(&wide16[32:15]);

    always_comb begin
        if (scaled_prod == 19'sd0) begin
            aligned_prod_32 = 32'sd0;
        end else if (net_shift < 7'sd0) begin
            aligned_prod_32 = {{13{rshifted_narrow[18]}}, rshifted_narrow};
        end else if (net_shift > 7'sd15) begin
            if (net_shift >= 7'sd26) begin
                aligned_prod_32 = scaled_prod[18] ? 32'sh8000_0000 : 32'sh7FFF_FFFF;
            end else if (sat_pos_hi) begin
                aligned_prod_32 = 32'sh7FFF_FFFF;
            end else if (sat_neg_hi) begin
                aligned_prod_32 = 32'sh8000_0000;
            end else begin
                aligned_prod_32 = {wide16[15:0], 16'd0};
            end
        end else begin
            if (lshifted_wide > 34'sd2147483647) begin
                aligned_prod_32 = 32'sh7FFF_FFFF;
            end else if (lshifted_wide < -34'sd2147483648) begin
                aligned_prod_32 = 32'sh8000_0000;
            end else begin
                aligned_prod_32 = lshifted_wide[31:0];
            end
        end
    end

endmodule


// =============================================================================
// Module: accum_adder — 32-bit saturating signed adder (sat_add32)
// =============================================================================
module accum_adder (
    input  logic signed [31:0] a,
    input  logic signed [31:0] b,
    output logic signed [31:0] sum
);

    function automatic logic signed [31:0] sat_add32(
        input logic signed [31:0] a,
        input logic signed [31:0] b
    );
        logic signed [32:0] s;
        begin
            s = $signed({a[31], a}) + $signed({b[31], b});
            if (s > 33'sd2147483647)       sat_add32 = 32'sh7FFF_FFFF;
            else if (s < -33'sd2147483648) sat_add32 = 32'sh8000_0000;
            else                           sat_add32 = s[31:0];
        end
    endfunction

    assign sum = sat_add32(a, b);

endmodule
