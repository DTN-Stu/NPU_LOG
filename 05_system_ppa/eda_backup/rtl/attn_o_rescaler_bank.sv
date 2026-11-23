// =============================================================================
// Module: attn_o_rescaler_bank.sv
// Description: FlashAttention O-Accumulator Online Rescaler Bank (16 Lanes).
//              Isolated from GEMM MAC compute:
//                - 2-stage pipeline delay on {acc_rescale_req, resc_q, resc_r}
//                  to stay aligned with log8_fp4_mac_array Stage 1A -> 1B -> 2
//                - 16x vsq_rescaler_32b (3-term CSD + arithmetic right shifter)
//                - Bypass mux when acc_rescale_req_q1 == 0
// =============================================================================
`timescale 1ns/1ps
`include "pe_defines.svh"

module attn_o_rescaler_bank #(
    parameter int NUM_LANES_P = NUM_LANES
)(
    input  logic                                clk,
    input  logic                                reset_n,

    // Online rescale sideband from P-FIFO (Stage 1A timing)
    input  logic                                acc_rescale_req_i,
    input  logic signed [7:0]                   resc_q_i,
    input  logic        [2:0]                   resc_r_i,

    // Stage 2 RMW feedback from rmw_latch_acc_buffer
    input  logic signed [NUM_LANES_P-1:0][31:0] old_res_i,

    // Stage 2 Outputs (to log8_fp4_mac_array & rmw_latch_acc_buffer)
    output logic                                acc_rescale_req_q1_o,
    output logic signed [NUM_LANES_P-1:0][31:0] effective_old_res_o
);

    // Stage 1A -> 1B -> 2 pipeline alignment
    logic              acc_rescale_req_s1a, acc_rescale_req_q1;
    logic signed [7:0] resc_q_s1a,          resc_q_q1;
    logic        [2:0] resc_r_s1a,          resc_r_q1;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            acc_rescale_req_s1a <= 1'b0;
            resc_q_s1a          <= 8'sd0;
            resc_r_s1a          <= 3'd0;
            acc_rescale_req_q1  <= 1'b0;
            resc_q_q1           <= 8'sd0;
            resc_r_q1           <= 3'd0;
        end else begin
            acc_rescale_req_s1a <= acc_rescale_req_i;
            resc_q_s1a          <= resc_q_i;
            resc_r_s1a          <= resc_r_i;
            acc_rescale_req_q1  <= acc_rescale_req_s1a;
            resc_q_q1           <= resc_q_s1a;
            resc_r_q1           <= resc_r_s1a;
        end
    end

    assign acc_rescale_req_q1_o = acc_rescale_req_q1;

    genvar i;
    generate
        for (i = 0; i < NUM_LANES_P; i++) begin : GEN_RESCALE_LANE
            // Operand isolation: gate rescaler inputs to 0 when no rescale is active
            wire signed [31:0] gated_old_res = acc_rescale_req_q1 ? old_res_i[i] : 32'sd0;
            logic signed [31:0] rescaled_old_res;
            vsq_rescaler_32b u_rescaler (
                .val_in     (gated_old_res),
                .resc_q     (resc_q_q1),
                .resc_r     (resc_r_q1),
                .scaled_out (rescaled_old_res)
            );
            assign effective_old_res_o[i] = acc_rescale_req_q1 ? rescaled_old_res : old_res_i[i];
        end
    endgenerate

endmodule
