// =============================================================================
// Config #2: NVFP4 (from MiX RTL) + FP32 Accumulator + Norm&Round In-Loop
//   - Iso-Throughput: 16 Lanes x 16 Elements = 256 MACs/cycle
//   - Uses MiX's genuine NVFP4_MUL #(.group_size(16), .sub_group_size(16))
//   - Accumulator: FF-based 1W2R array (USE_FF=1, DEPTH=16) storing FP32
// =============================================================================
`timescale 1ns/1ps

module cfg2_nvfp4_fp32_ff #(
    parameter int NUM_LANES_P = 16,
    parameter int ACC_DEPTH_P = 16,
    parameter int PKT_W       = 72
)(
    input  logic                                clk,
    input  logic                                reset_n,

    input  logic [NUM_LANES_P-1:0]              lane_valid_in,
    input  logic [$clog2(ACC_DEPTH_P)-1:0]      lane_acc_idx,
    input  logic [PKT_W-1:0]                    act_broadcast, // {E4M3[7:0], fp4[63:0]}
    input  logic [NUM_LANES_P-1:0][PKT_W-1:0]   wt_in,

    input  logic                                tile_drain_req,
    input  logic [$clog2(ACC_DEPTH_P)-1:0]      tile_drain_acc_idx,
    input  logic                                tile_clear_on_drain,

    output logic [NUM_LANES_P-1:0][31:0]        out_fp32_vec,
    output logic                                out_valid
);

    localparam int SLOT_AW = $clog2(ACC_DEPTH_P);
    localparam int TOTAL_W = NUM_LANES_P * 32;

    // Delay control signals 1 cycle to align with NVFP4_MUL's output register
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
    // Stage 1 (MiX NVFP4_MUL) + Stage 2 (13x8 Mult + FP32 Norm/Add + FF Acc)
    // -------------------------------------------------------------------------
    logic [TOTAL_W-1:0]     wdata, rdata1, rdata2;
    logic [ACC_DEPTH_P-1:0] slot_cleared;

    genvar i;
    generate
        for (i = 0; i < NUM_LANES_P; i++) begin : GEN_LANE
            logic [0:0][4:0]         mul_scale_exp_sum;
            logic signed [0:0][12:0] mul_out_sum;
            logic [0:0][7:0]         mul_scale_mant_product;

            logic [15:0][3:0] act_elems, wt_elems;
            for (genvar e = 0; e < 16; e++) begin : GEN_UNPACK
                assign act_elems[e] = act_broadcast[e*4 +: 4];
                assign wt_elems[e]  = wt_in[i][e*4 +: 4];
            end

            NVFP4_MUL #(
                .group_size     (16),
                .sub_group_size (16),
                .scale_bits     (8),
                .elem_bits      (4)
            ) u_mix_nvfp4_mul (
                .clk                    (clk),
                .rst_n                  (reset_n),
                .nvfp4_a_scale          (act_broadcast[71:64]),
                .nvfp4_a_elem           (act_elems),
                .nvfp4_b_scale          (wt_in[i][71:64]),
                .nvfp4_b_elem           (wt_elems),
                .out_scale_exp_sum      (mul_scale_exp_sum),
                .out_sum                (mul_out_sum),
                .out_scale_mant_product (mul_scale_mant_product)
            );

            // 13b signed x 8b unsigned -> 21-bit magnitude
            wire        p_sign = mul_out_sum[0][12];
            wire [12:0] p_abs  = p_sign ? 13'(-mul_out_sum[0]) : 13'(mul_out_sum[0]);
            wire [20:0] ext_prod = p_abs * mul_scale_mant_product[0];

            wire [31:0] acc_old = slot_cleared[lane_acc_idx_q1] ? 32'd0 : rdata1[i*32 +: 32];
            logic [31:0] acc_new;

            fp32_norm_add_2in #(
                .MAG_W           (21),
                .EXP_IN_W        (9),
                .EXP_BIAS_OFFSET (105) // 127 (FP32) - 14 (2*E4M3 bias) - 8 (Q2+Q6) = 105
            ) u_fp32_acc (
                .prod_sign_i    (p_sign),
                .prod_mag_i     (ext_prod),
                .prod_exp_i     ({4'd0, mul_scale_exp_sum[0]}),
                .acc_old_fp32_i (acc_old),
                .acc_new_fp32_o (acc_new)
            );

            assign wdata[i*32 +: 32] = lane_valid_q1[i] ? acc_new : acc_old;
        end
    endgenerate

    latch_array #(
        .DEPTH  (ACC_DEPTH_P),
        .DW     (TOTAL_W),
        .USE_FF (1'b1)
    ) u_ff_acc (
        .clk     (clk),
        .reset_n (reset_n),
        .waddr   (lane_acc_idx_q1),
        .wdata   (wdata),
        .wen     (|lane_valid_q1),
        .ren1    (|lane_valid_q1),
        .ren2    (tile_drain_req),
        .raddr1  (lane_acc_idx_q1),
        .raddr2  (tile_drain_acc_idx),
        .rdata1  (rdata1),
        .rdata2  (rdata2)
    );

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            slot_cleared <= {ACC_DEPTH_P{1'b1}};
            out_fp32_vec <= '0;
            out_valid    <= 1'b0;
        end else begin
            if (|lane_valid_q1) slot_cleared[lane_acc_idx_q1] <= 1'b0;
            if (tile_drain_req && tile_clear_on_drain) slot_cleared[tile_drain_acc_idx] <= 1'b1;

            out_valid <= tile_drain_req;
            if (tile_drain_req) begin
                for (int l = 0; l < NUM_LANES_P; l++) begin
                    out_fp32_vec[l] <= slot_cleared[tile_drain_acc_idx] ? 32'd0 : rdata2[l*32 +: 32];
                end
            end
        end
    end

endmodule
