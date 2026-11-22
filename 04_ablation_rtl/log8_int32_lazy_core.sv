// =============================================================================
// Module: log8_int32_lazy_core
// Description: Clean Log8-FP4 + INT32 Accumulator + Operand-Isolated Lazy FP16 Drain.
//   - Mirrors cfg1/cfg2/cfg6 2-stage pipeline and uses identical LOG8_MUL tree.
//   - ACC_USE_FF = 1 : Flip-Flop accumulator (Config #4)
//   - ACC_USE_FF = 0 : Clock-gated D-Latch accumulator (Config #5)
// =============================================================================
`timescale 1ns/1ps

module log8_int32_lazy_core #(
    parameter int NUM_LANES_P = 16,
    parameter int ACC_DEPTH_P = 16,
    parameter int PKT_W       = 72,
    parameter int FRAC_BITS   = 16,
    parameter bit ACC_USE_FF  = 1'b0
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

    localparam int SLOT_AW = $clog2(ACC_DEPTH_P);
    localparam int TOTAL_W = NUM_LANES_P * 32;

    // Delay control signals 1 cycle to align with LOG8_MUL's output register
    logic [NUM_LANES_P-1:0] lane_valid_q1;
    logic [SLOT_AW-1:0]     lane_acc_idx_q1;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            lane_valid_q1   <= '0;
            lane_acc_idx_q1 <= '0;
        end else begin
            lane_valid_q1   <= lane_valid_in;
            lane_acc_idx_q1 <= lane_acc_idx;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 1 (LOG8_MUL) + Stage 2 (Shifter + INT32 Sat Add + Acc Buffer)
    // -------------------------------------------------------------------------
    logic [TOTAL_W-1:0]     wdata, rdata1, rdata2;
    logic [ACC_DEPTH_P-1:0] slot_cleared;

    genvar i;
    generate
        for (i = 0; i < NUM_LANES_P; i++) begin : GEN_LANE
            logic signed [5:0]  mul_out_q;
            logic signed [18:0] mul_out_scaled_prod;
            logic signed [31:0] aligned_prod_32;
            logic signed [31:0] acc_new;

            logic [15:0][3:0] act_elems, wt_elems;
            for (genvar e = 0; e < 16; e++) begin : GEN_UNPACK
                assign act_elems[e] = act_broadcast[e*4 +: 4];
                assign wt_elems[e]  = wt_in[i][e*4 +: 4];
            end

            LOG8_MUL #(
                .group_size (16)
            ) u_log8_mul (
                .clk             (clk),
                .rst_n           (reset_n),
                .log8_a_scale    (signed'(act_broadcast[71:64])),
                .log8_a_elem     (act_elems),
                .log8_b_scale    (signed'(wt_in[i][71:64])),
                .log8_b_elem     (wt_elems),
                .out_q           (mul_out_q),
                .out_scaled_prod (mul_out_scaled_prod)
            );

            bidirectional_shifter #(
                .FRAC_BITS (FRAC_BITS)
            ) u_bidirectional_shifter (
                .scaled_prod     (mul_out_scaled_prod),
                .q               (mul_out_q),
                .aligned_prod_32 (aligned_prod_32)
            );

            wire signed [31:0] acc_old = slot_cleared[lane_acc_idx_q1]
                                       ? 32'sd0
                                       : signed'(rdata1[i*32 +: 32]);

            accum_adder u_accum_adder (
                .a   (acc_old),
                .b   (aligned_prod_32),
                .sum (acc_new)
            );

            assign wdata[i*32 +: 32] = lane_valid_q1[i] ? unsigned'(acc_new) : unsigned'(acc_old);
        end
    endgenerate

    wire any_lane_valid_q1 = |lane_valid_q1;

    latch_array #(
        .DEPTH        (ACC_DEPTH_P),
        .DW           (TOTAL_W),
        .USE_FF       (ACC_USE_FF),
        .FF_ASYNC_RST (1'b0)
    ) u_acc (
        .clk     (clk),
        .reset_n (reset_n),
        .waddr   (lane_acc_idx_q1),
        .wdata   (wdata),
        .wen     (any_lane_valid_q1),
        .ren1    (any_lane_valid_q1),
        .ren2    (tile_drain_req),
        .raddr1  (lane_acc_idx_q1),
        .raddr2  (tile_drain_acc_idx),
        .rdata1  (rdata1),
        .rdata2  (rdata2)
    );

    // -------------------------------------------------------------------------
    // Operand-Isolated Lazy FP16 Drain Normalizer (Port B)
    // -------------------------------------------------------------------------
    logic [TOTAL_W-1:0]            drain_word_comb;
    logic [NUM_LANES_P-1:0][15:0]  drain_fp16_comb;

    always_comb begin
        if (any_lane_valid_q1 && (tile_drain_acc_idx == lane_acc_idx_q1)) begin
            drain_word_comb = wdata;
        end else if (slot_cleared[tile_drain_acc_idx]) begin
            drain_word_comb = '0;
        end else begin
            drain_word_comb = rdata2;
        end
    end

    generate
        for (i = 0; i < NUM_LANES_P; i++) begin : GEN_DRAIN_NORM
            wire signed [31:0] norm_in_isolated = tile_drain_req
                                                ? signed'(drain_word_comb[i*32 +: 32])
                                                : 32'sd0;
            logscale_lazy_drain_norm #(
                .FRAC_BITS (FRAC_BITS)
            ) u_drain_norm (
                .norm_in  (norm_in_isolated),
                .fp16_out (drain_fp16_comb[i])
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            slot_cleared <= {ACC_DEPTH_P{1'b1}};
            out_fp16_vec <= '0;
            out_valid    <= 1'b0;
        end else begin
            if (any_lane_valid_q1) slot_cleared[lane_acc_idx_q1] <= 1'b0;
            if (tile_drain_req && tile_clear_on_drain) slot_cleared[tile_drain_acc_idx] <= 1'b1;

            out_valid <= tile_drain_req;
            if (tile_drain_req) begin
                out_fp16_vec <= drain_fp16_comb;
            end
        end
    end

endmodule
