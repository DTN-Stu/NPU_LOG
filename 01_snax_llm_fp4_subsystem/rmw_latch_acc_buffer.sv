// =============================================================================
// Module: rmw_latch_acc_buffer.sv
// Description: 1W2R D-Latch Accumulator Buffer + Virtual Clear Mask + Rolling Drain.
//              Pure storage subsystem decoupled from MAC compute and rescaling:
//                - Port A: Same-cycle RMW read (old_res_o) & writeback (acc_wdata_i)
//                - Port B: Non-blocking rolling drain (raw_drained_word_o)
// =============================================================================
`timescale 1ns/1ps
`include "pe_defines.svh"

module rmw_latch_acc_buffer #(
    parameter int NUM_LANES_P = NUM_LANES,
    parameter int ACC_DEPTH_P = ACCUMULATOR_DEPTH,
    parameter bit ACC_USE_FF  = 1'b0,
    localparam int SLOT_AW    = $clog2(ACC_DEPTH_P),
    localparam int TOTAL_W    = NUM_LANES_P * 32
)(
    input  logic                                clk,
    input  logic                                reset_n,

    // Stage 2 RMW Port A (from log8_fp4_mac_array & attn_o_rescaler_bank)
    input  logic [NUM_LANES_P-1:0]              lane_valid_q1_i,
    input  logic                                acc_rescale_req_q1_i,
    input  logic [SLOT_AW-1:0]                  acc_addr_q1_i,
    input  logic [TOTAL_W-1:0]                  acc_wdata_i,
    output logic signed [NUM_LANES_P-1:0][31:0] old_res_o,

    // Port B Rolling Drain Interface
    input  logic                                tile_drain_req_i,
    input  logic [SLOT_AW-1:0]                  tile_drain_acc_idx_i,
    input  logic                                tile_clear_on_drain_i,
    output logic [TOTAL_W-1:0]                  raw_drained_word_o,
    output logic                                raw_drained_valid_o
);

    wire any_lane_valid_q1 = |lane_valid_q1_i;
    wire port_a_active     = any_lane_valid_q1 | acc_rescale_req_q1_i;

    logic [TOTAL_W-1:0]     rdata1, rdata2;
    logic [ACC_DEPTH_P-1:0] slot_cleared;

    latch_array #(
        .DEPTH  (ACC_DEPTH_P),
        .DW     (TOTAL_W),
        .USE_FF (ACC_USE_FF)
    ) u_latch_array (
        .clk     (clk),
        .reset_n (reset_n),
        .waddr   (acc_addr_q1_i),
        .wdata   (acc_wdata_i),
        .wen     (port_a_active),
        .ren1    (port_a_active),
        .ren2    (tile_drain_req_i),
        .raddr1  (acc_addr_q1_i),
        .raddr2  (tile_drain_acc_idx_i),
        .rdata1  (rdata1),
        .rdata2  (rdata2)
    );

    // Mask cleared slots to 0 on Port A read
    genvar i;
    generate
        for (i = 0; i < NUM_LANES_P; i++) begin : GEN_OLD_RES
            assign old_res_o[i] = slot_cleared[acc_addr_q1_i] ? 32'sd0
                                                              : signed'(rdata1[i*32 +: 32]);
        end
    endgenerate

    // Virtual clear mask
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            slot_cleared <= '1;
        end else begin
            if (any_lane_valid_q1) begin
                slot_cleared[acc_addr_q1_i] <= 1'b0;
            end
            if (tile_drain_req_i && tile_clear_on_drain_i) begin
                slot_cleared[tile_drain_acc_idx_i] <= 1'b1;
            end
        end
    end

    // Port B drain mux: same-slot Stage-2 bypass + clear mask
    logic [TOTAL_W-1:0] drain_word_comb;
    always_comb begin
        if (any_lane_valid_q1 && (tile_drain_acc_idx_i == acc_addr_q1_i)) begin
            drain_word_comb = acc_wdata_i;
        end else if (slot_cleared[tile_drain_acc_idx_i]) begin
            drain_word_comb = '0;
        end else begin
            drain_word_comb = rdata2;
        end
    end

    // Registered Port B drain output (latency 1 cycle from tile_drain_req_i)
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            raw_drained_valid_o <= 1'b0;
            raw_drained_word_o  <= '0;
        end else begin
            raw_drained_valid_o <= tile_drain_req_i;
            if (tile_drain_req_i) begin
                raw_drained_word_o <= drain_word_comb;
            end
        end
    end

endmodule
