// =============================================================================
// Module: snax_llm_sfu.sv
// Description: 16-Lane SIMD Special Function Unit (SFU) for SNAX FP4 LLM Subsystem.
//              Fully compatible with the 256-bit (16 x FP16) valid/ready
//              streaming interfaces of snax_fused_llm_core / snax_stream_read_fp16
//              / snax_logq_shared / snax_streamer_write_out.
//
// Supported Operations (cfg_op_i):
//   - 3'd0 (SFU_BYPASS)   : out = in_a
//   - 3'd1 (SFU_RESIDUAL) : out[i] = in_a[i] + in_b[i]
//   - 3'd2 (SFU_RMSNORM)  : Buffers token x (up to MAX_VEC_BLOCKS x 16 FP16),
//                           accumulates sum(x^2), computes inv_rms = rsqrt(sum*inv_n + eps),
//                           then streams out[i] = x[i] * inv_rms * gamma[i] (gamma on in_b)
//   - 3'd3 (SFU_ROPE)     : Pairwise rotary embedding (8 pairs per 256b vector):
//                           in_b[2k] = cos_k, in_b[2k+1] = sin_k
//                           out[2k]   = in_a[2k]*cos_k - in_a[2k+1]*sin_k
//                           out[2k+1] = in_a[2k]*sin_k + in_a[2k+1]*cos_k
//   - 3'd4 (SFU_SWIGLU)   : out[i] = (in_a[i] * sigmoid_pwl(in_a[i])) * in_b[i]
// =============================================================================
`timescale 1ns/1ps

module snax_llm_sfu #(
    parameter int NUM_LANES      = 16,
    parameter int MAX_VEC_BLOCKS = 256  // Max 16-elem vectors per token (256*16 = 4096 hidden dim)
)(
    input  logic                       clk,
    input  logic                       rst_n,

    // Control & CSR Configuration
    input  logic                       start_i,         // Pulse to start a new token/job (used for RMSNorm)
    input  logic [2:0]                 cfg_op_i,        // 0:Bypass, 1:Residual, 2:RMSNorm, 3:RoPE, 4:SwiGLU
    input  logic [15:0]                cfg_num_vecs_i,  // Number of 256b vectors per token (RMSNorm)
    input  logic [31:0]                cfg_inv_n_fp32_i,// Precomputed 1/D in IEEE FP32
    input  logic [31:0]                cfg_eps_fp32_i,  // Epsilon in IEEE FP32 (e.g. 1e-5 = 32'h3727C5AC)
    output logic                       busy_o,
    output logic                       done_o,

    // Primary 256-bit FP16 Input Stream (from Core or Read Streamer)
    input  logic [NUM_LANES*16-1:0]    in_a_fp16_i,
    input  logic                       in_a_valid_i,
    output logic                       in_a_ready_o,

    // Secondary 256-bit FP16 Input Stream (Residual Y / Gamma / {Sin,Cos} / Up-proj)
    input  logic [NUM_LANES*16-1:0]    in_b_fp16_i,
    input  logic                       in_b_valid_i,
    output logic                       in_b_ready_o,

    // 256-bit FP16 Output Stream (to Shared Log-Quantizer or Write Streamer)
    output logic [NUM_LANES*16-1:0]    out_fp16_o,
    output logic                       out_valid_o,
    input  logic                       out_ready_i
);

    localparam logic [2:0] SFU_BYPASS   = 3'd0;
    localparam logic [2:0] SFU_RESIDUAL = 3'd1;
    localparam logic [2:0] SFU_RMSNORM  = 3'd2;
    localparam logic [2:0] SFU_ROPE     = 3'd3;
    localparam logic [2:0] SFU_SWIGLU   = 3'd4;

    localparam int BUF_AW = $clog2(MAX_VEC_BLOCKS);

    // -------------------------------------------------------------------------
    // 1. Unpack FP16 Inputs -> FP32 (16 lanes)
    // -------------------------------------------------------------------------
    logic [NUM_LANES-1:0][31:0] a_fp32, b_fp32;
    genvar i;
    generate
        for (i = 0; i < NUM_LANES; i++) begin : GEN_UNPACK
            sfu_fp16_to_fp32 u_unp_a (.fp16_i(in_a_fp16_i[i*16 +: 16]), .fp32_o(a_fp32[i]));
            sfu_fp16_to_fp32 u_unp_b (.fp16_i(in_b_fp16_i[i*16 +: 16]), .fp32_o(b_fp32[i]));
        end
    endgenerate

    // -------------------------------------------------------------------------
    // 2. RMSNorm Token Buffer & Reduction State Machine
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {
        RMS_IDLE  = 2'd0,
        RMS_ACCUM = 2'd1,
        RMS_RSQRT = 2'd2,
        RMS_SCALE = 2'd3
    } rms_state_t;

    rms_state_t rms_state;
    logic [15:0] vec_cnt;
    logic [31:0] rms_sum_sq_q;
    logic [31:0] inv_rms_q;

    logic [NUM_LANES*16-1:0] rms_buf [MAX_VEC_BLOCKS];
    logic [NUM_LANES*16-1:0] rms_buf_rdata;
    logic [NUM_LANES-1:0][31:0] rms_x_fp32;

    assign rms_buf_rdata = rms_buf[vec_cnt[BUF_AW-1:0]];

    generate
        for (i = 0; i < NUM_LANES; i++) begin : GEN_RMS_UNPACK
            sfu_fp16_to_fp32 u_unp_rms (.fp16_i(rms_buf_rdata[i*16 +: 16]), .fp32_o(rms_x_fp32[i]));
        end
    endgenerate

    // 16-lane x^2 + 16->1 FP32 adder tree for RMSNorm Pass 1
    logic [NUM_LANES-1:0][31:0] sq_fp32;
    logic [7:0][31:0]           tree_l1;
    logic [3:0][31:0]           tree_l2;
    logic [1:0][31:0]           tree_l3;
    logic [31:0]                vec_sum_sq;
    logic [31:0]                rms_sum_next;

    generate
        for (i = 0; i < NUM_LANES; i++) begin : GEN_SQ
            sfu_fp32_mul u_sq (.a_i(a_fp32[i]), .b_i(a_fp32[i]), .res_o(sq_fp32[i]));
        end
        for (i = 0; i < 8; i++) begin : GEN_L1
            sfu_fp32_addsub u_l1 (.a_i(sq_fp32[2*i]), .b_i(sq_fp32[2*i+1]), .sub_i(1'b0), .res_o(tree_l1[i]));
        end
        for (i = 0; i < 4; i++) begin : GEN_L2
            sfu_fp32_addsub u_l2 (.a_i(tree_l1[2*i]), .b_i(tree_l1[2*i+1]), .sub_i(1'b0), .res_o(tree_l2[i]));
        end
        for (i = 0; i < 2; i++) begin : GEN_L3
            sfu_fp32_addsub u_l3 (.a_i(tree_l2[2*i]), .b_i(tree_l2[2*i+1]), .sub_i(1'b0), .res_o(tree_l3[i]));
        end
    endgenerate
    sfu_fp32_addsub u_l4   (.a_i(tree_l3[0]), .b_i(tree_l3[1]), .sub_i(1'b0), .res_o(vec_sum_sq));
    sfu_fp32_addsub u_racc (.a_i(rms_sum_sq_q), .b_i(vec_sum_sq), .sub_i(1'b0), .res_o(rms_sum_next));

    // Shared 1/sqrt(mean(x^2) + eps) unit
    logic [31:0] mean_sq_fp32, mean_plus_eps_fp32, inv_rms_comb;
    sfu_fp32_mul    u_mean (.a_i(rms_sum_sq_q), .b_i(cfg_inv_n_fp32_i), .res_o(mean_sq_fp32));
    sfu_fp32_addsub u_eps  (.a_i(mean_sq_fp32), .b_i(cfg_eps_fp32_i),   .sub_i(1'b0), .res_o(mean_plus_eps_fp32));
    sfu_rsqrt_fp32  u_rsq  (.x_i(mean_plus_eps_fp32), .inv_sqrt_o(inv_rms_comb));

    // -------------------------------------------------------------------------
    // 3. Shared 16-Lane SIMD Arithmetic Datapath
    //    Reused across Residual Add, RMSNorm Scale, RoPE, and SwiGLU
    // -------------------------------------------------------------------------
    logic [NUM_LANES-1:0][31:0] sig_fp32, silu_fp32, swiglu_fp32;
    logic [NUM_LANES-1:0][31:0] res_add_fp32;
    logic [NUM_LANES-1:0][31:0] rms_scaled_x, rms_out_fp32;
    logic [NUM_LANES-1:0][31:0] rope_out_fp32;
    logic [NUM_LANES-1:0][31:0] mux_fp32;
    logic [NUM_LANES*16-1:0]    packed_fp16_comb;

    generate
        for (i = 0; i < NUM_LANES; i++) begin : GEN_SIMD_LANES
            // Residual Add: a + b
            sfu_fp32_addsub u_res_add (
                .a_i   (a_fp32[i]),
                .b_i   (b_fp32[i]),
                .sub_i (1'b0),
                .res_o (res_add_fp32[i])
            );

            // SwiGLU: (a * sigmoid(a)) * b
            sfu_pwl_sigmoid_fp32 u_sig (
                .x_i   (a_fp32[i]),
                .sig_o (sig_fp32[i])
            );
            sfu_fp32_mul u_silu (
                .a_i   (a_fp32[i]),
                .b_i   (sig_fp32[i]),
                .res_o (silu_fp32[i])
            );
            sfu_fp32_mul u_swiglu (
                .a_i   (silu_fp32[i]),
                .b_i   (b_fp32[i]),
                .res_o (swiglu_fp32[i])
            );

            // RMSNorm Pass 2: (x_buf * inv_rms) * gamma
            sfu_fp32_mul u_rms_m1 (
                .a_i   (rms_x_fp32[i]),
                .b_i   (inv_rms_q),
                .res_o (rms_scaled_x[i])
            );
            sfu_fp32_mul u_rms_m2 (
                .a_i   (rms_scaled_x[i]),
                .b_i   (b_fp32[i]),
                .res_o (rms_out_fp32[i])
            );
        end

        // RoPE: 8 pairs per 16-lane vector
        //   Pair k uses: x_e = a[2k], x_o = a[2k+1], cos = b[2k], sin = b[2k+1]
        for (i = 0; i < NUM_LANES/2; i++) begin : GEN_ROPE_PAIRS
            logic [31:0] xe_cos, xo_sin, xe_sin, xo_cos;
            sfu_fp32_mul    u_m0 (.a_i(a_fp32[2*i]),   .b_i(b_fp32[2*i]),   .res_o(xe_cos));
            sfu_fp32_mul    u_m1 (.a_i(a_fp32[2*i+1]), .b_i(b_fp32[2*i+1]), .res_o(xo_sin));
            sfu_fp32_mul    u_m2 (.a_i(a_fp32[2*i]),   .b_i(b_fp32[2*i+1]), .res_o(xe_sin));
            sfu_fp32_mul    u_m3 (.a_i(a_fp32[2*i+1]), .b_i(b_fp32[2*i]),   .res_o(xo_cos));

            sfu_fp32_addsub u_even (.a_i(xe_cos), .b_i(xo_sin), .sub_i(1'b1), .res_o(rope_out_fp32[2*i]));
            sfu_fp32_addsub u_odd  (.a_i(xe_sin), .b_i(xo_cos), .sub_i(1'b0), .res_o(rope_out_fp32[2*i+1]));
        end

        // Output Format Mux & FP32 -> FP16 Packer
        for (i = 0; i < NUM_LANES; i++) begin : GEN_PACK
            always_comb begin
                unique case (cfg_op_i)
                    SFU_RESIDUAL: mux_fp32[i] = res_add_fp32[i];
                    SFU_RMSNORM:  mux_fp32[i] = rms_out_fp32[i];
                    SFU_ROPE:     mux_fp32[i] = rope_out_fp32[i];
                    SFU_SWIGLU:   mux_fp32[i] = swiglu_fp32[i];
                    default:      mux_fp32[i] = a_fp32[i];
                endcase
            end
            sfu_fp32_to_fp16 u_pack (.fp32_i(mux_fp32[i]), .fp16_o(packed_fp16_comb[i*16 +: 16]));
        end
    endgenerate

    // -------------------------------------------------------------------------
    // 4. Streaming Handshake & RMSNorm FSM
    // -------------------------------------------------------------------------
    wire out_can_accept = !out_valid_o || out_ready_i;

    // Streaming modes (Bypass, Residual, RoPE, SwiGLU) fire in 1 cycle
    wire stream_fire = (cfg_op_i != SFU_RMSNORM) &&
                       in_a_valid_i &&
                       ((cfg_op_i == SFU_BYPASS) || in_b_valid_i) &&
                       out_can_accept;

    always_comb begin
        if (cfg_op_i == SFU_RMSNORM) begin
            in_a_ready_o = (rms_state == RMS_ACCUM);
            in_b_ready_o = (rms_state == RMS_SCALE) && out_can_accept;
        end else if (cfg_op_i == SFU_BYPASS) begin
            in_a_ready_o = out_can_accept;
            in_b_ready_o = 1'b0;
        end else begin
            in_a_ready_o = in_b_valid_i && out_can_accept;
            in_b_ready_o = in_a_valid_i && out_can_accept;
        end
    end

    assign busy_o = (cfg_op_i == SFU_RMSNORM) && (rms_state != RMS_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rms_state    <= RMS_IDLE;
            vec_cnt      <= 16'd0;
            rms_sum_sq_q <= 32'd0;
            inv_rms_q    <= 32'd0;
            out_valid_o  <= 1'b0;
            out_fp16_o   <= '0;
            done_o       <= 1'b0;
        end else begin
            done_o <= 1'b0;

            if (cfg_op_i != SFU_RMSNORM) begin
                rms_state <= RMS_IDLE;
                if (stream_fire) begin
                    out_valid_o <= 1'b1;
                    out_fp16_o  <= (cfg_op_i == SFU_BYPASS) ? in_a_fp16_i : packed_fp16_comb;
                end else if (out_ready_i) begin
                    out_valid_o <= 1'b0;
                end
            end else begin
                // RMSNorm 2-pass FSM
                case (rms_state)
                    RMS_IDLE: begin
                        if (out_ready_i) out_valid_o <= 1'b0;
                        if (start_i) begin
                            rms_state    <= RMS_ACCUM;
                            vec_cnt      <= 16'd0;
                            rms_sum_sq_q <= 32'd0;
                        end
                    end

                    RMS_ACCUM: begin
                        if (in_a_valid_i) begin
                            rms_buf[vec_cnt[BUF_AW-1:0]] <= in_a_fp16_i;
                            rms_sum_sq_q                 <= rms_sum_next;
                            if (vec_cnt + 16'd1 >= cfg_num_vecs_i) begin
                                rms_state <= RMS_RSQRT;
                                vec_cnt   <= 16'd0;
                            end else begin
                                vec_cnt <= vec_cnt + 16'd1;
                            end
                        end
                    end

                    RMS_RSQRT: begin
                        inv_rms_q <= inv_rms_comb;
                        rms_state <= RMS_SCALE;
                        vec_cnt   <= 16'd0;
                    end

                    RMS_SCALE: begin
                        if (in_b_valid_i && out_can_accept) begin
                            out_valid_o <= 1'b1;
                            out_fp16_o  <= packed_fp16_comb;
                            if (vec_cnt + 16'd1 >= cfg_num_vecs_i) begin
                                rms_state <= RMS_IDLE;
                                done_o    <= 1'b1;
                            end else begin
                                vec_cnt <= vec_cnt + 16'd1;
                            end
                        end else if (out_ready_i) begin
                            out_valid_o <= 1'b0;
                        end
                    end
                endcase
            end
        end
    end

endmodule
