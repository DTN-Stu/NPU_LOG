// =============================================================================
// Config #4: Log8-Scale (LOG8_MUL) + INT32 Accumulator + Lazy FP16 Drain (FF)
//   - Iso-Throughput: 16 Lanes x 16 Elements = 256 MACs/cycle
//   - Uses clean log8_int32_lazy_core with ACC_USE_FF=1 (Flip-Flop)
// =============================================================================
`timescale 1ns/1ps

module cfg4_log8_int32_lazy_ff #(
    parameter int NUM_LANES_P = 16,
    parameter int ACC_DEPTH_P = 16,
    parameter int PKT_W       = 72,
    parameter int FRAC_BITS   = 16
)(
    input  logic                                clk,
    input  logic                                reset_n,

    input  logic [NUM_LANES_P-1:0]              lane_valid_in,
    input  logic [$clog2(ACC_DEPTH_P)-1:0]      lane_acc_idx,
    input  logic [PKT_W-1:0]                    act_broadcast,
    input  logic [NUM_LANES_P-1:0][PKT_W-1:0]   wt_in,

    input  logic                                tile_drain_req,
    input  logic [$clog2(ACC_DEPTH_P)-1:0]      tile_drain_acc_idx,
    input  logic                                tile_clear_on_drain,

    output logic [NUM_LANES_P-1:0][15:0]        out_fp16_vec,
    output logic                                out_valid
);

    log8_int32_lazy_core #(
        .NUM_LANES_P (NUM_LANES_P),
        .ACC_DEPTH_P (ACC_DEPTH_P),
        .PKT_W       (PKT_W),
        .FRAC_BITS   (FRAC_BITS),
        .ACC_USE_FF  (1'b1)
    ) u_core (
        .clk                 (clk),
        .reset_n             (reset_n),
        .lane_valid_in       (lane_valid_in),
        .lane_acc_idx        (lane_acc_idx),
        .act_broadcast       (act_broadcast),
        .wt_in               (wt_in),
        .tile_drain_req      (tile_drain_req),
        .tile_drain_acc_idx  (tile_drain_acc_idx),
        .tile_clear_on_drain (tile_clear_on_drain),
        .out_fp16_vec        (out_fp16_vec),
        .out_valid           (out_valid)
    );

endmodule
