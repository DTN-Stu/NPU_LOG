// =============================================================================
// Module: log8_fp4_mac_array.sv
// Description: Pure Compute Log8-FP4 MAC Array (16 Lanes x 16 Elements = 256 MACs/cycle).
//              Contains ONLY arithmetic compute logic (zero storage, zero rescale,
//              zero FP16 drain normalizer):
//                - Stage 1A: dot_product_16 (fp4_mult_lut + compressor_4to2) + scale_adder
//                - Stage 1B: logscale_frac_scaler (3-term CSD) + bidirectional_shifter
//                - Stage 2 : 16x 32-bit saturating accum_adder (RMW with Acc Buffer)
// =============================================================================
`timescale 1ns/1ps
`include "pe_defines.svh"

module log8_fp4_mac_array #(
    parameter int NUM_LANES_P = NUM_LANES,
    parameter int ACC_DEPTH_P = ACCUMULATOR_DEPTH,
    parameter int PKT_W       = PACKET_WIDTH,
    parameter int FRAC_BITS   = 16,
    localparam int SLOT_AW    = $clog2(ACC_DEPTH_P)
)(
    input  logic                                clk,
    input  logic                                reset_n,

    // Stage 1 Inputs
    input  logic [NUM_LANES_P-1:0]              lane_valid_i,
    input  logic [SLOT_AW-1:0]                  lane_acc_idx_i,
    input  logic [PKT_W-1:0]                    act_broadcast_i,
    input  logic [NUM_LANES_P-1:0][PKT_W-1:0]   wt_in_i,

    // Stage 2 RMW Feedback Input (from rmw_latch_acc_buffer via attn_o_rescaler_bank)
    input  logic signed [NUM_LANES_P-1:0][31:0] effective_old_res_i,

    // Stage 2 Writeback Outputs (to rmw_latch_acc_buffer)
    output logic [NUM_LANES_P-1:0]              lane_valid_q1_o,
    output logic [SLOT_AW-1:0]                  acc_waddr_o,
    output logic [NUM_LANES_P*32-1:0]           acc_wdata_o
);

    // -------------------------------------------------------------------------
    // Stage 1A: 16-Element FP4 Dot Product + Log8 Scale Adder (Combinational)
    // -------------------------------------------------------------------------
    logic signed [NUM_LANES_P-1:0][15:0] dot_out_array;
    logic signed [NUM_LANES_P-1:0][5:0]  q_array;
    logic        [NUM_LANES_P-1:0][2:0]  r_array;

    genvar i;
    generate
        for (i = 0; i < NUM_LANES_P; i++) begin : GEN_S1A
            dot_product_16 u_dot_product_16 (
                .act_vec (act_broadcast_i[63:0]),
                .wt_vec  (wt_in_i[i][63:0]),
                .dot_out (dot_out_array[i])
            );

            scale_adder u_scale_adder (
                .scale_a (signed'(act_broadcast_i[71:64])),
                .scale_w (signed'(wt_in_i[i][71:64])),
                .q       (q_array[i]),
                .r       (r_array[i])
            );
        end
    endgenerate

    // Stage 1A -> Stage 1B Pipeline Registers
    logic [NUM_LANES_P-1:0]              lane_valid_s1a;
    logic [SLOT_AW-1:0]                  lane_acc_idx_s1a;
    logic signed [NUM_LANES_P-1:0][15:0] dot_out_q1;
    logic signed [NUM_LANES_P-1:0][5:0]  q_array_q1;
    logic        [NUM_LANES_P-1:0][2:0]  r_array_q1;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            lane_valid_s1a   <= '0;
            lane_acc_idx_s1a <= '0;
            for (int l = 0; l < NUM_LANES_P; l++) begin
                dot_out_q1[l] <= 16'sd0;
                q_array_q1[l] <= 6'sd0;
                r_array_q1[l] <= 3'd0;
            end
        end else begin
            lane_valid_s1a   <= lane_valid_i;
            lane_acc_idx_s1a <= lane_acc_idx_i;
            for (int l = 0; l < NUM_LANES_P; l++) begin
                if (lane_valid_i[l]) begin
                    dot_out_q1[l] <= dot_out_array[l];
                    q_array_q1[l] <= q_array[l];
                    r_array_q1[l] <= r_array[l];
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Stage 1B: 3-Term CSD Fractional Scaler + Bidirectional Octave Shifter
    // -------------------------------------------------------------------------
    logic signed [NUM_LANES_P-1:0][18:0] scaled_prod;
    logic signed [NUM_LANES_P-1:0][31:0] fe_aligned_prod;

    generate
        for (i = 0; i < NUM_LANES_P; i++) begin : GEN_S1B
            logscale_frac_scaler u_logscale_frac_scaler (
                .prod_in     (signed'(dot_out_q1[i][12:0])),
                .r           (r_array_q1[i]),
                .scaled_prod (scaled_prod[i])
            );

            bidirectional_shifter #(
                .FRAC_BITS (FRAC_BITS)
            ) u_bidirectional_shifter (
                .scaled_prod     (scaled_prod[i]),
                .q               (q_array_q1[i]),
                .aligned_prod_32 (fe_aligned_prod[i])
            );
        end
    endgenerate

    // Stage 1B -> Stage 2 Pipeline Registers
    logic [NUM_LANES_P-1:0]              lane_valid_q1;
    logic [SLOT_AW-1:0]                  lane_acc_idx_q1;
    logic signed [NUM_LANES_P-1:0][31:0] aligned_prod_q1;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            lane_valid_q1   <= '0;
            lane_acc_idx_q1 <= '0;
            for (int l = 0; l < NUM_LANES_P; l++) begin
                aligned_prod_q1[l] <= 32'sd0;
            end
        end else begin
            lane_valid_q1   <= lane_valid_s1a;
            lane_acc_idx_q1 <= lane_acc_idx_s1a;
            for (int l = 0; l < NUM_LANES_P; l++) begin
                if (lane_valid_s1a[l]) begin
                    aligned_prod_q1[l] <= fe_aligned_prod[l];
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Stage 2: 16x 32-bit Saturating Accumulator Adder (RMW)
    // -------------------------------------------------------------------------
    assign lane_valid_q1_o = lane_valid_q1;
    assign acc_waddr_o     = lane_acc_idx_q1;

    generate
        for (i = 0; i < NUM_LANES_P; i++) begin : GEN_S2_ADD
            logic signed [31:0] new_res;
            accum_adder u_accum_adder (
                .a   (aligned_prod_q1[i]),
                .b   (effective_old_res_i[i]),
                .sum (new_res)
            );
            assign acc_wdata_o[i*32 +: 32] = lane_valid_q1[i] ? new_res : effective_old_res_i[i];
        end
    endgenerate

endmodule
