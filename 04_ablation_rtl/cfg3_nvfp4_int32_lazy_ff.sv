// =============================================================================
// Config #3: NVFP4 (E4M3) + INT32 Accumulator + Lazy FP16 Drain (FF DEPTH=16)
//   - Iso-Throughput: 16 Lanes x 16 Elements = 256 MACs/cycle
//   - Uses nvfp4_scale_mac_lane (4b x 4b + 13b x 8b + Q16.16 shifter)
//   - Replaces FP32 in-loop norm/add with INT32 RMW + Lazy FP16 Drain Normalizer
//   - Accumulator: FF-based 1W2R array (USE_FF=1, DEPTH=16)
// =============================================================================
`timescale 1ns/1ps

module cfg3_nvfp4_int32_lazy_ff #(
    parameter int NUM_LANES_P = 16,
    parameter int ACC_DEPTH_P = 16,
    parameter int PKT_W       = 72,
    parameter int FRAC_BITS   = 16
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

    output logic [NUM_LANES_P-1:0][15:0]        out_fp16_vec,
    output logic                                out_valid
);

    localparam int SLOT_AW = $clog2(ACC_DEPTH_P);
    localparam int TOTAL_W = NUM_LANES_P * 32;

    // Delay lane_acc_idx by 2 cycles to match nvfp4_scale_mac_lane Stage 2
    logic [SLOT_AW-1:0] acc_idx_q1a, acc_idx_q1b;
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            acc_idx_q1a <= '0;
            acc_idx_q1b <= '0;
        end else begin
            acc_idx_q1a <= lane_acc_idx;
            acc_idx_q1b <= acc_idx_q1a;
        end
    end

    logic [NUM_LANES_P-1:0] lane_valid_s2;
    logic [TOTAL_W-1:0]     wdata, rdata1, rdata2;
    logic [ACC_DEPTH_P-1:0] slot_cleared;

    genvar i;
    generate
        for (i = 0; i < NUM_LANES_P; i++) begin : GEN_LANE
            logic signed [31:0] acc_old;
            logic signed [31:0] aligned_prod_s2;
            logic signed [31:0] acc_new_s2;

            assign acc_old = slot_cleared[acc_idx_q1b] ? 32'sd0
                                                       : signed'(rdata1[i*32 +: 32]);

            nvfp4_scale_mac_lane #(
                .PKT_W     (PKT_W),
                .FRAC_BITS (FRAC_BITS)
            ) u_lane (
                .clk                   (clk),
                .rst_n                 (reset_n),
                .lane_valid_i          (lane_valid_in[i]),
                .act_pkt_i             (act_broadcast),
                .wt_pkt_i              (wt_in[i]),
                .acc_old_i             (acc_old),
                .lane_valid_stage2_o   (lane_valid_s2[i]),
                .aligned_prod_stage2_o (aligned_prod_s2),
                .acc_new_stage2_o      (acc_new_s2)
            );

            assign wdata[i*32 +: 32] = lane_valid_s2[i] ? acc_new_s2 : acc_old;
        end
    endgenerate

    latch_array #(
        .DEPTH  (ACC_DEPTH_P),
        .DW     (TOTAL_W),
        .USE_FF (1'b1)
    ) u_ff_acc (
        .clk     (clk),
        .reset_n (reset_n),
        .waddr   (acc_idx_q1b),
        .wdata   (wdata),
        .wen     (|lane_valid_s2),
        .ren1    (|lane_valid_s2),
        .ren2    (tile_drain_req),
        .raddr1  (acc_idx_q1b),
        .raddr2  (tile_drain_acc_idx),
        .rdata1  (rdata1),
        .rdata2  (rdata2)
    );

    // Drain + Lazy FP16 Normalizer (1 cycle drain reg + combinational LZC -> reg)
    logic [TOTAL_W-1:0] drained_word_q;
    logic               drained_valid_q;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            slot_cleared    <= {ACC_DEPTH_P{1'b1}};
            drained_word_q  <= '0;
            drained_valid_q <= 1'b0;
        end else begin
            if (|lane_valid_s2) slot_cleared[acc_idx_q1b] <= 1'b0;
            if (tile_drain_req && tile_clear_on_drain) slot_cleared[tile_drain_acc_idx] <= 1'b1;

            drained_valid_q <= tile_drain_req;
            if (tile_drain_req) begin
                drained_word_q <= slot_cleared[tile_drain_acc_idx] ? '0 : rdata2;
            end
        end
    end

    logic [NUM_LANES_P-1:0][15:0] fp16_norm;
    generate
        for (i = 0; i < NUM_LANES_P; i++) begin : GEN_DRAIN_NORM
            logscale_lazy_drain_norm #(
                .FRAC_BITS (FRAC_BITS)
            ) u_drain_norm (
                .norm_in  (signed'(drained_word_q[i*32 +: 32])),
                .fp16_out (fp16_norm[i])
            );
        end
    endgenerate

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            out_fp16_vec <= '0;
            out_valid    <= 1'b0;
        end else begin
            out_valid <= drained_valid_q;
            if (drained_valid_q) begin
                out_fp16_vec <= fp16_norm;
            end
        end
    end

endmodule
