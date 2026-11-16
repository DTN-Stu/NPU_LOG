// =============================================================================
// Module: nvfp4_scale_mac_lane.sv
// Description: Baseline NVFP4 (FP8 E4M3 Block-Scale) MAC Lane for IEEE TCAD/TVLSI
//              Hardware Ablation & Synthesis Comparison against vsq_matmul_engine.
//
// Mathematical & Datapath Comparison (Block Size B = 16):
//   - Common Stage 1A Dot-Product:
//       P_int = sum_{k=0..15} A_k * W_k   (13-bit signed in Q2, via dot_product_16)
//   - NVFP4 Scale Format (FP8 E4M3 per 16-element block):
//       S_A = 2^(E_a - 7) * (1 + M_a / 8),   E_a in [0..15], M_a in [0..7]
//       S_W = 2^(E_w - 7) * (1 + M_w / 8),   E_w in [0..15], M_w in [0..7]
//   - NVFP4 Hardware Bottleneck (vs. Proposed Log8 3-Term CSD):
//       1. Multiplier #1 (4b x 4b unsigned):
//            M_prod = (1.M_a) * (1.M_w)   -> 8-bit unsigned (Q2.6)
//       2. Multiplier #2 (13b signed x 8b unsigned):
//            P_scaled = P_int * M_prod    -> 21-bit signed (Q4.6 = Q8)
//       3. Bidirectional Barrel Shifter:
//            Shifts 21-bit P_scaled by q = (E_a + E_w - 14) into 32-bit Q16.16.
// =============================================================================
`timescale 1ns/1ps
`include "pe_defines.svh"

// -----------------------------------------------------------------------------
// Submodule 1: nvfp4_e4m3_scale_combiner
//   Combines two FP8 E4M3 block scales S_A and S_W:
//     - Adds 4-bit biased exponents -> signed 6-bit octave shift q
//     - Multiplies two 4-bit significands (1.M_a * 1.M_w) -> 8-bit Q2.6 product
// -----------------------------------------------------------------------------
module nvfp4_e4m3_scale_combiner (
    input  logic [7:0]        scale_a_e4m3, // {sign(0), exp[6:3], mant[2:0]}
    input  logic [7:0]        scale_w_e4m3, // {sign(0), exp[6:3], mant[2:0]}
    output logic signed [5:0] q_exp,        // Combined unbiased octave exponent
    output logic [7:0]        mant_prod_q6  // 4b x 4b unsigned product in Q2.6
);

    wire [3:0] ea_raw = scale_a_e4m3[6:3];
    wire [2:0] ma_raw = scale_a_e4m3[2:0];
    wire [3:0] ew_raw = scale_w_e4m3[6:3];
    wire [2:0] mw_raw = scale_w_e4m3[2:0];

    // Normal vs. subnormal E4M3 handling (bias = 7)
    wire       a_is_sub = (ea_raw == 4'd0);
    wire       w_is_sub = (ew_raw == 4'd0);

    wire [3:0] sig_a = a_is_sub ? {1'b0, ma_raw} : {1'b1, ma_raw}; // Q1.3
    wire [3:0] sig_w = w_is_sub ? {1'b0, mw_raw} : {1'b1, mw_raw}; // Q1.3

    wire [3:0] eff_ea = a_is_sub ? 4'd1 : ea_raw;
    wire [3:0] eff_ew = w_is_sub ? 4'd1 : ew_raw;

    // Combined octave exponent: (eff_ea - 7) + (eff_ew - 7) = eff_ea + eff_ew - 14
    assign q_exp = $signed({2'b00, eff_ea}) + $signed({2'b00, eff_ew}) - 6'sd14;

    // [HARDWARE OVERHEAD #1]: 4-bit x 4-bit Mantissa Multiplier (absent in Log8)
    assign mant_prod_q6 = sig_a * sig_w; // 8-bit unsigned in Q2.6 (range 0..225)

endmodule


// -----------------------------------------------------------------------------
// Submodule 2: nvfp4_mant_product_scaler
//   Multiplies the 13-bit signed intra-block dot-product (Q2) by the 8-bit
//   unsigned scale mantissa product (Q2.6) -> 21-bit signed product in Q8.
//   (In Proposed Log8, this 13b x 8b multiplier is replaced by a 3-term CSD CSA)
// -----------------------------------------------------------------------------
module nvfp4_mant_product_scaler (
    input  logic signed [12:0] prod_in_q2,    // 13-bit signed Q2 dot-product
    input  logic        [7:0]  mant_prod_q6,  // 8-bit unsigned Q2.6 scale mantissa
    output logic signed [20:0] scaled_prod_q8 // 21-bit signed Q8 product
);

    wire signed [8:0] mant_signed = $signed({1'b0, mant_prod_q6});

    // [HARDWARE OVERHEAD #2]: 13-bit x 8-bit Signed-Unsigned Multiplier
    assign scaled_prod_q8 = prod_in_q2 * mant_signed;

endmodule


// -----------------------------------------------------------------------------
// Submodule 3: nvfp4_bidirectional_shifter
//   Aligns the 21-bit Q8 scaled product into the 32-bit signed Q(32-FRAC).FRAC
//   fixed-point accumulator format with signed saturation.
// -----------------------------------------------------------------------------
module nvfp4_bidirectional_shifter #(
    parameter int FRAC_BITS = 16
)(
    input  logic signed [20:0] scaled_prod_q8,  // 21-bit signed Q8
    input  logic signed [5:0]  q_exp,           // Octave shift from E4M3 scales
    output logic signed [31:0] aligned_prod_32  // 32-bit signed Q16.16 (saturated)
);

    localparam int BASE_SHIFT = FRAC_BITS - 8;  // Q8 -> Q16 alignment (+8)

    logic signed [6:0] net_shift;
    assign net_shift = $signed({q_exp[5], q_exp}) + 7'(signed'(BASE_SHIFT));

    logic [3:0]         lshift_amt;
    logic [4:0]         rshift_amt;
    logic signed [35:0] lshifted_wide;
    logic signed [20:0] rshifted_narrow;

    // Magnitude-checked saturation for net_shift in [16..31]:
    //   W = scaled_prod_q8 <<< (net_shift - 16) = scaled_prod_q8 <<< net_shift[3:0]
    //   True value = W * 2^16, which fits int32 iff -2^15 <= W < +2^15.
    logic signed [35:0] wide16;
    logic               sat_pos_hi;
    logic               sat_neg_hi;

    assign lshift_amt      = net_shift[3:0];
    assign rshift_amt      = (-net_shift > 7'sd21) ? 5'd21 : unsigned'(-net_shift[4:0]);
    assign lshifted_wide   = 36'(scaled_prod_q8) <<< lshift_amt;
    assign rshifted_narrow = scaled_prod_q8 >>> rshift_amt;

    assign wide16     = 36'(scaled_prod_q8) <<< net_shift[3:0];
    assign sat_pos_hi = ~wide16[35] & (|wide16[34:15]);
    assign sat_neg_hi =  wide16[35] & ~(&wide16[34:15]);

    always_comb begin
        if (scaled_prod_q8 == 21'sd0) begin
            aligned_prod_32 = 32'sd0;
        end else if (net_shift < 7'sd0) begin
            aligned_prod_32 = {{11{rshifted_narrow[20]}}, rshifted_narrow};
        end else if (net_shift > 7'sd15) begin
            if (net_shift >= 7'sd32) begin
                aligned_prod_32 = scaled_prod_q8[20] ? 32'sh8000_0000 : 32'sh7FFF_FFFF;
            end else if (sat_pos_hi) begin
                aligned_prod_32 = 32'sh7FFF_FFFF;
            end else if (sat_neg_hi) begin
                aligned_prod_32 = 32'sh8000_0000;
            end else begin
                aligned_prod_32 = {wide16[15:0], 16'd0};
            end
        end else begin
            if (lshifted_wide > 36'sd2147483647) begin
                aligned_prod_32 = 32'sh7FFF_FFFF;
            end else if (lshifted_wide < -36'sd2147483648) begin
                aligned_prod_32 = 32'sh8000_0000;
            end else begin
                aligned_prod_32 = lshifted_wide[31:0];
            end
        end
    end

endmodule


// =============================================================================
// Top-Level Baseline Lane Module: nvfp4_scale_mac_lane
//   Matches the exact pipeline staging (Stage 1A -> Stage 1B -> Stage 2 RMW)
//   of a single lane inside vsq_matmul_engine.sv for 1-to-1 synthesis comparison.
// =============================================================================
module nvfp4_scale_mac_lane #(
    parameter int PKT_W     = 72, // 64b FP4 E2M1 + 8b E4M3 Scale
    parameter int FRAC_BITS = 16
)(
    input  logic               clk,
    input  logic               rst_n,

    // Control & Input Packets
    input  logic               lane_valid_i,
    input  logic [PKT_W-1:0]   act_pkt_i,       // {scale_a_e4m3[7:0], act_fp4[63:0]}
    input  logic [PKT_W-1:0]   wt_pkt_i,        // {scale_w_e4m3[7:0], wt_fp4[63:0]}

    // Accumulator RMW Interface (Stage 2)
    input  logic signed [31:0] acc_old_i,
    output logic               lane_valid_stage2_o,
    output logic signed [31:0] aligned_prod_stage2_o,
    output logic signed [31:0] acc_new_stage2_o
);

    // -------------------------------------------------------------------------
    // Stage 1A: 16-element FP4 Dot Product + E4M3 Scale Combiner (4b x 4b mult)
    // -------------------------------------------------------------------------
    logic signed [15:0] dot_out_s1a;
    logic signed [5:0]  q_exp_s1a;
    logic        [7:0]  mant_prod_s1a;

    dot_product_16 u_dot_product_16 (
        .act_vec (act_pkt_i[63:0]),
        .wt_vec  (wt_pkt_i[63:0]),
        .dot_out (dot_out_s1a)
    );

    nvfp4_e4m3_scale_combiner u_scale_combiner (
        .scale_a_e4m3 (act_pkt_i[71:64]),
        .scale_w_e4m3 (wt_pkt_i[71:64]),
        .q_exp        (q_exp_s1a),
        .mant_prod_q6 (mant_prod_s1a)
    );

    // Pipeline Register Stage 1A -> Stage 1B
    logic               valid_q1a;
    logic signed [12:0] dot_out_q1a;
    logic signed [5:0]  q_exp_q1a;
    logic        [7:0]  mant_prod_q1a;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_q1a     <= 1'b0;
            dot_out_q1a   <= 13'sd0;
            q_exp_q1a     <= 6'sd0;
            mant_prod_q1a <= 8'd0;
        end else begin
            valid_q1a <= lane_valid_i;
            if (lane_valid_i) begin
                dot_out_q1a   <= dot_out_s1a[12:0];
                q_exp_q1a     <= q_exp_s1a;
                mant_prod_q1a <= mant_prod_s1a;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Stage 1B: 13b x 8b Scale-Dot Multiplier + Bidirectional Shifter
    // -------------------------------------------------------------------------
    logic signed [20:0] scaled_prod_s1b;
    logic signed [31:0] aligned_prod_s1b;

    nvfp4_mant_product_scaler u_mant_scaler (
        .prod_in_q2     (dot_out_q1a),
        .mant_prod_q6   (mant_prod_q1a),
        .scaled_prod_q8 (scaled_prod_s1b)
    );

    nvfp4_bidirectional_shifter #(
        .FRAC_BITS (FRAC_BITS)
    ) u_shifter (
        .scaled_prod_q8  (scaled_prod_s1b),
        .q_exp           (q_exp_q1a),
        .aligned_prod_32 (aligned_prod_s1b)
    );

    // Pipeline Register Stage 1B -> Stage 2
    logic               valid_q1b;
    logic signed [31:0] aligned_prod_q1b;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_q1b        <= 1'b0;
            aligned_prod_q1b <= 32'sd0;
        end else begin
            valid_q1b <= valid_q1a;
            if (valid_q1a) begin
                aligned_prod_q1b <= aligned_prod_s1b;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Stage 2: 32-bit Saturating Accumulator Add
    // -------------------------------------------------------------------------
    logic signed [32:0] sum_ext;
    logic signed [31:0] sat_sum;

    assign sum_ext = $signed({aligned_prod_q1b[31], aligned_prod_q1b}) +
                     $signed({acc_old_i[31], acc_old_i});

    always_comb begin
        if (sum_ext > 33'sd2147483647)       sat_sum = 32'sh7FFF_FFFF;
        else if (sum_ext < -33'sd2147483648) sat_sum = 32'sh8000_0000;
        else                                 sat_sum = sum_ext[31:0];
    end

    assign lane_valid_stage2_o   = valid_q1b;
    assign aligned_prod_stage2_o = aligned_prod_q1b;
    assign acc_new_stage2_o      = valid_q1b ? sat_sum : acc_old_i;

endmodule
