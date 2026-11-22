// =============================================================================
// Module: LOG8_MUL (mirrors MXFP4_g16_MUL / NVFP4_MUL structure 1-to-1)
//   - Same E2M1 decode + 2bx2b mult + 5b signed + shift + 10->11->12->13b adder tree
//   - Log8 scale combine (scale_a + scale_w = 8*q + r) + CSD logscale_frac_scaler
//   - 1-cycle output register (Stage 1 -> Stage 2), matching MXFP4_g16_MUL/NVFP4_MUL
// =============================================================================
`timescale 1ns/1ps

module LOG8_MUL #(
    parameter int group_size     = 16,
    parameter int PRODUCT_WIDTH  = 9,
    parameter int LOG2_SUB_SIZE  = $clog2(group_size),           // 4
    parameter int dot_out_bits   = PRODUCT_WIDTH + LOG2_SUB_SIZE // 13
) (
    input  logic                              clk,
    input  logic                              rst_n,

    input  logic signed [7:0]                 log8_a_scale,
    input  logic [group_size-1:0][3:0]        log8_a_elem,
    input  logic signed [7:0]                 log8_b_scale,
    input  logic [group_size-1:0][3:0]        log8_b_elem,

    output logic signed [5:0]                 out_q,
    output logic signed [18:0]                out_scaled_prod
);

    localparam [1:0] FP4_EXP_BIAS    = 2'd1;
    localparam int   FULL_MANT_W     = 2;
    localparam int   UNSIGNED_PROD_W = 4;
    localparam int   SIGNED_PROD_W   = 5;

    logic [group_size-1:0]                           act_sign, wgt_sign;
    logic [group_size-1:0][1:0]                      act_exp_raw, wgt_exp_raw;
    logic [group_size-1:0]                           act_mant_raw, wgt_mant_raw;
    logic [group_size-1:0][1:0]                      act_exp, wgt_exp;
    logic [group_size-1:0][FULL_MANT_W-1:0]          act_mant, wgt_mant;
    logic [group_size-1:0][UNSIGNED_PROD_W-1:0]      prod_raw;
    logic signed [group_size-1:0][SIGNED_PROD_W-1:0] prod_sgn;
    logic [group_size-1:0][2:0]                      shamt;
    logic signed [group_size-1:0][PRODUCT_WIDTH-1:0] prod_fixed;

    logic signed [5:0]              q_pre;
    logic        [2:0]              r_pre;
    logic signed [dot_out_bits-1:0] dot_sum_pre;
    logic signed [18:0]             scaled_prod_pre;

    // Per-element E2M1 decode + 2x2 mult + fixed-point conversion (identical to NVFP4_MUL)
    always_comb begin
        for (int i = 0; i < group_size; i++) begin
            act_sign[i]     = log8_a_elem[i][3];
            act_exp_raw[i]  = log8_a_elem[i][2:1];
            act_mant_raw[i] = log8_a_elem[i][0];

            wgt_sign[i]     = log8_b_elem[i][3];
            wgt_exp_raw[i]  = log8_b_elem[i][2:1];
            wgt_mant_raw[i] = log8_b_elem[i][0];

            act_exp[i] = (act_exp_raw[i] == 2'b00) ? 2'b00 : act_exp_raw[i] - FP4_EXP_BIAS;
            wgt_exp[i] = (wgt_exp_raw[i] == 2'b00) ? 2'b00 : wgt_exp_raw[i] - FP4_EXP_BIAS;

            act_mant[i] = {(act_exp_raw[i] != 2'b00), act_mant_raw[i]};
            wgt_mant[i] = {(wgt_exp_raw[i] != 2'b00), wgt_mant_raw[i]};

            prod_raw[i] = act_mant[i] * wgt_mant[i];

            prod_sgn[i] = (act_sign[i] ^ wgt_sign[i])
                        ? -$signed({1'b0, prod_raw[i]})
                        :  $signed({1'b0, prod_raw[i]});

            shamt[i] = {1'b0, act_exp[i]} + {1'b0, wgt_exp[i]};

            prod_fixed[i] = PRODUCT_WIDTH'(signed'(prod_sgn[i])) << shamt[i];
        end
    end

    // 16-element adder tree with per-stage 1-bit width growth (identical to NVFP4_MUL)
    genvar s, i;
    generate
        for (s = 0; s <= LOG2_SUB_SIZE; s++) begin : tree
            localparam int W = PRODUCT_WIDTH + s;
            localparam int N = group_size >> s;
            logic signed [W-1:0] val [N-1:0];

            if (s == 0) begin : init
                for (i = 0; i < N; i++) begin : elem
                    assign val[i] = prod_fixed[i];
                end
            end else begin : reduce
                for (i = 0; i < N; i++) begin : elem
                    assign val[i] = W'(signed'(tree[s-1].val[2*i]))
                                  + W'(signed'(tree[s-1].val[2*i+1]));
                end
            end
        end
    endgenerate

    assign dot_sum_pre = tree[LOG2_SUB_SIZE].val[0];

    // Log8 scale combine + CSD fractional scaler (Stage 1)
    scale_adder u_scale_adder (
        .scale_a (log8_a_scale),
        .scale_w (log8_b_scale),
        .q       (q_pre),
        .r       (r_pre)
    );

    logscale_frac_scaler u_logscale_frac_scaler (
        .prod_in     (dot_sum_pre),
        .r           (r_pre),
        .scaled_prod (scaled_prod_pre)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_q           <= 6'sd0;
            out_scaled_prod <= 19'sd0;
        end else begin
            out_q           <= q_pre;
            out_scaled_prod <= scaled_prod_pre;
        end
    end

endmodule
