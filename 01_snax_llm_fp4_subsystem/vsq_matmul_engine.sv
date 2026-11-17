`timescale 1ns/1ps
`include "pe_defines.svh"

// =============================================================================
// vsq_matmul_engine — 16-Lane LogScale-Lazy MatMul Engine
//   Front-end: FP4 dot-product → log-scale split (q, r) → CSD frac scaler →
//              bidirectional octave shifter → 32-bit saturating RMW add
//   Accumulator: 1W2R latch array (write commits next cycle per two-phase
//              protocol; read is combinational; Port B = rolling slot drain)
//
// REQUIRED FILELIST (compiled TOGETHER):
//     fp4_mult_lut.sv  compressor_4to2.sv  dot_product_16.sv  latch_array.sv
//     vsq_matmul_engine.sv  + testbench
//   Do NOT co-compile with vsq_feba_core_top*.sv or vsq_pe_*.sv: they define
//   helper modules with the same names (logscale_lazy_drain_norm, etc.).
//
// INTERFACE CONVENTION (deliberately different from legacy vsq_feba_core_top):
//   - act_broadcast : ONE shared activation packet (64b FP4 + 8b exp) → all lanes
//   - wt_in[0:15]   : PER-LANE weight packets (16 output channels)
//   (legacy top was the opposite: act_in[16] + wt_broadcast)
//
// PIPELINE (2 stages — timing-relaxed, mirrors vsq_feba_core_top_latch_2r1w):
//   - Stage 1 (cycle T)    : dot → (q, r) → CSD scaler → shifter (feed-forward)
//   - Stage 1→2 registers  : lane_valid_q1 / lane_acc_idx_q1 / aligned_prod_q1
//   - Stage 2 (cycle T+1)  : same-cycle RMW (raddr1 == waddr == idx_q1) + write
//     command; commits per the latch_array two-phase protocol.
//   - Drain : request at Td → drain word registered (Td+1) → FP16 normalize
//     (combinational) → output registered (Td+2); core_vec_valid pulses at Td+2.
//     Rolling-drain overlap = drain request aligned with the accumulate's
//     STAGE-2 cycle (one cycle after its operand input), matching the legacy
//     golden-model semantics.
//   NOTE: Shared arithmetic primitives (logscale_lazy_drain_norm, scale_adder,
//         logscale_frac_scaler, bidirectional_shifter, accum_adder) now live in
//         log8_fp4_arith_units.sv. Compile log8_fp4_arith_units.sv together with
//         this file when elaborating vsq_matmul_engine.
// =============================================================================

// =============================================================================
// Module: vsq_matmul_engine
// =============================================================================
module vsq_matmul_engine #(
    parameter int NUM_LANES_P = NUM_LANES,        
    parameter int ACC_DEPTH_P = ACCUMULATOR_DEPTH,   
    parameter int PKT_W       = PACKET_WIDTH,      // 72 bits (64b FP4 + 8b Exp)
    parameter int FRAC_BITS   = 16,
    parameter bit ACC_USE_FF  = 1'b0,              // 0: D-latch (DHL), 1: Flip-flop (DFF)
    parameter bit ENABLE_RESCALE    = 1'b1,        // 1: include FlashAttention online rescaler
    parameter bit ENABLE_PACKED_OUT = 1'b1         // 1: include redundant core_vec_packed register
)
(
    input logic                            clk, 
    input logic                            reset_n, 

    input  logic [NUM_LANES_P-1:0]         lane_valid_in,
    input  logic [$clog2(ACC_DEPTH_P)-1:0] lane_acc_idx,             // Current accumulation slot
    input  logic [PKT_W-1:0]               act_broadcast,            // Shared Act vector broadcast
    input  logic [NUM_LANES_P-1:0][PKT_W-1:0] wt_in,  // Per-lane Weight vectors

    // Drain & Tile Control (Supports Simultaneous Compute & Rolling Drain)
    input  logic                           tile_drain_req,
    input  logic [$clog2(ACC_DEPTH_P)-1:0] tile_drain_acc_idx,       // Slot to drain across all lanes
    input  logic                           tile_clear_on_drain,

    // Online Rescale Interface (for FlashAttention O-accumulator)
    input  logic                           acc_rescale_req,
    input  logic signed [7:0]              resc_q,
    input  logic [2:0]                     resc_r,

    // Direct Raw 32-bit Score Output (for Softmax Requantizer Bypass)
    output logic [NUM_LANES_P*32-1:0]      raw_drained_word,
    output logic                           raw_drained_valid,

    // Final FP16 Vector Scaled Output (256-bit SIMD / 16 x FP16)
    output logic [NUM_LANES_P-1:0][15:0]   core_vec_out,
    output logic [NUM_LANES_P*16-1:0]      core_vec_packed,
    output logic                           core_vec_valid
);

    localparam int TOTAL_W = NUM_LANES_P * 32;   // 512 bits per slot
    localparam int SLOT_AW = $clog2(ACC_DEPTH_P);

    // -------------------------------------------------------------------------
    // STAGE 1 (cycle T): per-lane front-end
    //   dot → (q, r) split → CSD frac scaler → bidirectional shifter
    // -------------------------------------------------------------------------
    int mm_l_s1a;
    int mm_l_s1b;
    int mm_l_out;

    logic signed [NUM_LANES_P-1:0][15:0] dot_out_array;
    logic signed [NUM_LANES_P-1:0][5:0]  q_array;
    logic        [NUM_LANES_P-1:0][2:0]  r_array;
    logic signed [NUM_LANES_P-1:0][18:0] scaled_prod;
    logic signed [NUM_LANES_P-1:0][31:0] fe_aligned_prod;

    // -------------------------------------------------------------------------
    // Stage 1A -> Stage 1B pipeline registers (Dot-product & Scale adder output)
    // -------------------------------------------------------------------------
    logic [NUM_LANES_P-1:0]              lane_valid_s1a;
    logic [SLOT_AW-1:0]                  lane_acc_idx_s1a;
    logic                                acc_rescale_req_s1a;
    logic signed [7:0]                   resc_q_s1a;
    logic        [2:0]                   resc_r_s1a;
    logic signed [NUM_LANES_P-1:0][15:0] dot_out_q1;
    logic signed [NUM_LANES_P-1:0][5:0]  q_array_q1;
    logic        [NUM_LANES_P-1:0][2:0]  r_array_q1;

    logic                   acc_rescale_req_q1;
    logic signed [7:0]      resc_q_q1;
    logic        [2:0]      resc_r_q1;

    generate
        if (ENABLE_RESCALE) begin : GEN_RESCALE_REGS
            always_ff @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    acc_rescale_req_s1a <= 1'b0;
                    resc_q_s1a          <= 8'sd0;
                    resc_r_s1a          <= 3'd0;
                    acc_rescale_req_q1  <= 1'b0;
                    resc_q_q1           <= 8'sd0;
                    resc_r_q1           <= 3'd0;
                end else begin
                    acc_rescale_req_s1a <= acc_rescale_req;
                    resc_q_s1a          <= resc_q;
                    resc_r_s1a          <= resc_r;
                    acc_rescale_req_q1  <= acc_rescale_req_s1a;
                    resc_q_q1           <= resc_q_s1a;
                    resc_r_q1           <= resc_r_s1a;
                end
            end
        end else begin : GEN_NO_RESCALE_REGS
            assign acc_rescale_req_s1a = 1'b0;
            assign resc_q_s1a          = 8'sd0;
            assign resc_r_s1a          = 3'd0;
            assign acc_rescale_req_q1  = 1'b0;
            assign resc_q_q1           = 8'sd0;
            assign resc_r_q1           = 3'd0;
        end
    endgenerate

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            lane_valid_s1a   <= '0;
            lane_acc_idx_s1a <= '0;
            for (mm_l_s1a = 0; mm_l_s1a < NUM_LANES_P; mm_l_s1a++) begin
                dot_out_q1[mm_l_s1a] <= 16'sd0;
                q_array_q1[mm_l_s1a] <= 6'sd0;
                r_array_q1[mm_l_s1a] <= 3'd0;
            end
        end else begin
            lane_valid_s1a   <= lane_valid_in;
            lane_acc_idx_s1a <= lane_acc_idx;
            for (mm_l_s1a = 0; mm_l_s1a < NUM_LANES_P; mm_l_s1a++) begin
                if (lane_valid_in[mm_l_s1a]) begin
                    dot_out_q1[mm_l_s1a] <= dot_out_array[mm_l_s1a];
                    q_array_q1[mm_l_s1a] <= q_array[mm_l_s1a];
                    r_array_q1[mm_l_s1a] <= r_array[mm_l_s1a];
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // Stage 1B -> Stage 2 pipeline registers (Aligned product & RMW setup)
    // -------------------------------------------------------------------------
    logic [NUM_LANES_P-1:0] lane_valid_q1;
    logic [SLOT_AW-1:0]     lane_acc_idx_q1;
    logic signed [NUM_LANES_P-1:0][31:0] aligned_prod_q1;

        always_ff @(posedge clk or negedge reset_n) begin
            if (!reset_n) begin
                lane_valid_q1   <= '0;
                lane_acc_idx_q1 <= '0;
                for (mm_l_s1b = 0; mm_l_s1b < NUM_LANES_P; mm_l_s1b++) begin
                    aligned_prod_q1[mm_l_s1b] <= 32'sd0;
                end
            end else begin
                lane_valid_q1   <= lane_valid_s1a;
                lane_acc_idx_q1 <= lane_acc_idx_s1a;
                for (mm_l_s1b = 0; mm_l_s1b < NUM_LANES_P; mm_l_s1b++) begin
                    if (lane_valid_s1a[mm_l_s1b]) begin
                        aligned_prod_q1[mm_l_s1b] <= fe_aligned_prod[mm_l_s1b];
                    end
                end
            end
        end

    wire any_lane_valid_q1 = |lane_valid_q1;

    // -------------------------------------------------------------------------
    // STAGE 2 (cycle T+1): 32-bit saturating RMW add + latch-array write command
    // -------------------------------------------------------------------------
    logic signed [NUM_LANES_P-1:0][31:0] old_res;
    logic signed [NUM_LANES_P-1:0][31:0] new_res;

    logic [TOTAL_W-1:0] wdata, rdata1, rdata2;

    // Virtual clear mask (`slot_cleared` semantics, identical to legacy top):
    //   SET on reset / drain-with-clear; CLEARED when Stage 2 commits.
    logic [ACC_DEPTH_P-1:0] slot_cleared;

    genvar i;
    generate 
        for (i = 0; i < NUM_LANES_P; i++) begin: GEN_MAC_LANE
            dot_product_16 u_dot_product_16(
                .act_vec        (act_broadcast[63:0]),
                .wt_vec         (wt_in[i][63:0]),
                .dot_out        (dot_out_array[i])
            );

            scale_adder u_scale_adder(
                .scale_a    (signed'(act_broadcast[71:64])),
                .scale_w    (signed'(wt_in[i][71:64])),
                .q          (q_array[i]),
                .r          (r_array[i])
            );  

            logscale_frac_scaler u_logscale_frac_scaler(
                .prod_in            (signed'(dot_out_q1[i][12:0])),
                .r                  (r_array_q1[i]),
                .scaled_prod        (scaled_prod[i])
            ); 

            bidirectional_shifter #(.FRAC_BITS(FRAC_BITS))
            u_bidirectional_shifter (
                .scaled_prod        (scaled_prod[i]),
                .q                  (q_array_q1[i]),
                .aligned_prod_32    (fe_aligned_prod[i])
            );

            // Stage-2 RMW: old accumulator (masked if slot is 'cleared')
            assign old_res[i] = slot_cleared[lane_acc_idx_q1] ? 32'sd0
                                                       : signed'(rdata1[i*32 +: 32]);

            logic signed [31:0] effective_old_res;

            if (ENABLE_RESCALE) begin : GEN_RESCALE
                logic signed [31:0] rescaled_old_res;
                vsq_rescaler_32b u_rescaler (
                    .val_in     (acc_rescale_req_q1 ? old_res[i] : 32'sd0),
                    .resc_q     (resc_q_q1),
                    .resc_r     (resc_r_q1),
                    .scaled_out (rescaled_old_res)
                );
                assign effective_old_res = acc_rescale_req_q1 ? rescaled_old_res : old_res[i];
            end else begin : GEN_NO_RESCALE
                assign effective_old_res = old_res[i];
            end

            accum_adder u_accum_adder(
                .a         (aligned_prod_q1[i]),  
                .b         (effective_old_res),  
                .sum       (new_res[i]) 
            );

            // Per-lane gating: invalid lanes write back their (masked) effective old value
            assign wdata[i*32 +: 32] = lane_valid_q1[i] ? new_res[i] : effective_old_res;
        end 
    endgenerate

    // -------------------------------------------------------------------------
    // 1-Write 2-Read Latch Array RMW Accumulator
    //   Port A: same-cycle RMW read (raddr1 == waddr)
    //   Port B: rolling slot drain read
    // -------------------------------------------------------------------------
    latch_array #(
        .DEPTH  (ACC_DEPTH_P),
        .DW     (TOTAL_W),
        .USE_FF (ACC_USE_FF)
    ) u_latch_array (
        .clk                (clk),    
        .reset_n            (reset_n),

        .waddr              (lane_acc_idx_q1),  
        .wdata              (wdata),  
        .wen                (any_lane_valid_q1 | acc_rescale_req_q1),
        .ren1               (any_lane_valid_q1 | acc_rescale_req_q1),  // Port-A read gated to accumulate or rescale cycles
        .ren2               (tile_drain_req),     // Port-B read gated to drain windows
        .raddr1             (lane_acc_idx_q1), 
        .raddr2             (tile_drain_acc_idx), 
        .rdata1             (rdata1), 
        .rdata2             (rdata2)  
    );

    // -------------------------------------------------------------------------
    // Virtual clear mask
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            slot_cleared <= '1;
        end else begin
            if (any_lane_valid_q1) begin
                slot_cleared[lane_acc_idx_q1] <= 1'b0;
            end
            if (tile_drain_req && tile_clear_on_drain) begin
                slot_cleared[tile_drain_acc_idx] <= 1'b1;   // drain+clear wins
            end
        end
    end

    // -------------------------------------------------------------------------
    // Drain path: Port-B read + same-slot bypass + zero-mask
    // -------------------------------------------------------------------------
    logic [TOTAL_W-1:0] drain_word_comb;

    always_comb begin
        if (any_lane_valid_q1 && (tile_drain_acc_idx == lane_acc_idx_q1)) begin
            drain_word_comb = wdata;        // slot finishing Stage 2 -> bypass
        end else if (slot_cleared[tile_drain_acc_idx]) begin
            drain_word_comb = '0;
        end else begin
            drain_word_comb = rdata2;
        end
    end

    // Operand-isolated drain register (only updates on drain requests)
    logic               drain_req_q1;
    logic [TOTAL_W-1:0] drained_int32_word_q1;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            drain_req_q1          <= 1'b0;
            drained_int32_word_q1 <= '0;
        end else begin
            drain_req_q1 <= tile_drain_req;
            if (tile_drain_req) begin
                drained_int32_word_q1 <= drain_word_comb;
            end
        end
    end

    // 16× lazy FP16 normalizer (switching activity only during drain cycles)
    logic [NUM_LANES_P-1:0][15:0] drained_fp16;
    logic [NUM_LANES_P*16-1:0] drained_fp16_packed;

    generate
        for (i = 0; i < NUM_LANES_P; i++) begin : GEN_DRAIN_NORM
            logscale_lazy_drain_norm #(
                .FRAC_BITS (FRAC_BITS)
            ) u_drain_norm (
                .norm_in  (signed'(drained_int32_word_q1[i*32 +: 32])),
                .fp16_out (drained_fp16[i])
            );
            assign drained_fp16_packed[i*16 +: 16] = drained_fp16[i];
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Final output registers (drain latency: request Td → outputs at Td+2)
    // -------------------------------------------------------------------------
    assign raw_drained_word  = drained_int32_word_q1;
    assign raw_drained_valid = drain_req_q1;

    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            core_vec_valid <= 1'b0;
            for (mm_l_out = 0; mm_l_out < NUM_LANES_P; mm_l_out++) begin
                core_vec_out[mm_l_out] <= 16'h0000;
            end
        end else begin
            core_vec_valid <= drain_req_q1;
            if (drain_req_q1) begin
                for (mm_l_out = 0; mm_l_out < NUM_LANES_P; mm_l_out++) begin
                    core_vec_out[mm_l_out] <= drained_fp16[mm_l_out];
                end
            end
        end
    end

    generate
        if (ENABLE_PACKED_OUT) begin : GEN_PACKED_OUT
            always_ff @(posedge clk or negedge reset_n) begin
                if (!reset_n) begin
                    core_vec_packed <= '0;
                end else if (drain_req_q1) begin
                    core_vec_packed <= drained_fp16_packed;
                end
            end
        end else begin : GEN_NO_PACKED_OUT
            assign core_vec_packed = '0;
        end
    endgenerate

endmodule
