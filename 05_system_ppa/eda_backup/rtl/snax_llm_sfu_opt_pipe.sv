// =============================================================================
// Module: snax_llm_sfu_opt_pipe.sv
// Description: Deeply Pipelined Parameterized SIMD SFU (ALU_LANES = 16, 8, or 4)
//              for 500-700+ MHz operation in ASAP7 7nm.
//              - Drop-in interface compatible with snax_llm_sfu_opt (valid/ready + busy/done).
//              - Every combinational stage performs at most ONE FP20 operation
//                (Unpack, PWL-S1, PWL-S2, Mul, Add, or Pack), eliminating the
//                3.75 ns multi-op chain in snax_llm_sfu_opt.
//              - Reuses GEN_SHARED_ALU for 1/sqrt Newton-Raphson (removes dedicated
//                3x FP20 mul + 1x FP20 add inside sfu_rsqrt_fp20).
// =============================================================================
`timescale 1ns/1ps

module snax_llm_sfu_opt_pipe #(
    parameter int NUM_LANES = 16,
    parameter int ALU_LANES = 8   // Supported: 16, 8, 4
)(
    input  logic                       clk,
    input  logic                       rst_n,

    // Control & CSR Configuration
    input  logic                       start_i,          // Pulse to start RMSNorm 2-pass job
    input  logic [2:0]                 cfg_op_i,         // 0:Bypass, 1:Residual, 2:RMSNorm, 3:RoPE, 4:SwiGLU
    input  logic [15:0]                cfg_num_vecs_i,   // Number of 256b vectors per token (RMSNorm)
    input  logic [31:0]                cfg_inv_n_fp32_i, // Precomputed 1/D in IEEE FP32
    input  logic [31:0]                cfg_eps_fp32_i,   // Epsilon in IEEE FP32
    output logic                       busy_o,
    output logic                       done_o,

    // Primary 256-bit FP16 Input Stream
    input  logic [NUM_LANES*16-1:0]    in_a_fp16_i,
    input  logic                       in_a_valid_i,
    output logic                       in_a_ready_o,

    // Secondary 256-bit FP16 Input Stream
    input  logic [NUM_LANES*16-1:0]    in_b_fp16_i,
    input  logic                       in_b_valid_i,
    output logic                       in_b_ready_o,

    // 256-bit FP16 Output Stream
    output logic [NUM_LANES*16-1:0]    out_fp16_o,
    output logic                       out_valid_o,
    input  logic                       out_ready_i
);

    localparam logic [2:0] SFU_BYPASS   = 3'd0;
    localparam logic [2:0] SFU_RESIDUAL = 3'd1;
    localparam logic [2:0] SFU_RMSNORM  = 3'd2;
    localparam logic [2:0] SFU_ROPE     = 3'd3;
    localparam logic [2:0] SFU_SWIGLU   = 3'd4;

    localparam int NUM_CHUNKS = NUM_LANES / ALU_LANES; // 1, 2, or 4
    localparam int CHUNK_W    = (NUM_CHUNKS > 1) ? $clog2(NUM_CHUNKS) : 1;
    localparam int ALU_BITS   = ALU_LANES * 16;

    // -------------------------------------------------------------------------
    // 1. Registered Input Holding & Chunk Unpack (FP16 -> FP20)
    // -------------------------------------------------------------------------
    logic [NUM_LANES*16-1:0] in_a_hold_q, in_b_hold_q;
    logic [CHUNK_W-1:0]      chunk_q;
    logic                    active_vec_q;

    logic [ALU_BITS-1:0] cur_a_slice, cur_b_slice;
    always_comb begin
        if (NUM_CHUNKS == 1) begin
            cur_a_slice = in_a_hold_q[ALU_BITS-1:0];
            cur_b_slice = in_b_hold_q[ALU_BITS-1:0];
        end else begin
            cur_a_slice = in_a_hold_q[chunk_q * ALU_BITS +: ALU_BITS];
            cur_b_slice = in_b_hold_q[chunk_q * ALU_BITS +: ALU_BITS];
        end
    end

    logic [ALU_LANES-1:0][19:0] a_fp20_comb, b_fp20_comb;
    logic [ALU_LANES-1:0][19:0] a_fp20_q, b_fp20_q;

    genvar i;
    generate
        for (i = 0; i < ALU_LANES; i++) begin : GEN_UNP
            sfu_fp16_to_fp20 u_ua (.fp16_i(cur_a_slice[i*16 +: 16]), .fp20_o(a_fp20_comb[i]));
            sfu_fp16_to_fp20 u_ub (.fp16_i(cur_b_slice[i*16 +: 16]), .fp20_o(b_fp20_comb[i]));
        end
    endgenerate

    wire [19:0] inv_n_fp20 = {cfg_inv_n_fp32_i[31], cfg_inv_n_fp32_i[30:23], cfg_inv_n_fp32_i[22:12]};
    wire [19:0] eps_fp20   = {cfg_eps_fp32_i[31],   cfg_eps_fp32_i[30:23],   cfg_eps_fp32_i[22:12]};

    // -------------------------------------------------------------------------
    // 2. Shared Pipelined ALU (ALU_LANES x Mul + ALU_LANES x AddSub)
    // -------------------------------------------------------------------------
    logic [ALU_LANES-1:0][19:0] mul_a, mul_b, mul_res, mul_res_q;
    logic [ALU_LANES-1:0][19:0] add_a, add_b, add_res, add_res_q;
    logic [ALU_LANES-1:0]       add_sub;

    generate
        for (i = 0; i < ALU_LANES; i++) begin : GEN_SHARED_ALU
            sfu_fp20_mul u_mul (
                .a_i   (mul_a[i]),
                .b_i   (mul_b[i]),
                .res_o (mul_res[i])
            );
            sfu_fp20_addsub u_add (
                .a_i   (add_a[i]),
                .b_i   (add_b[i]),
                .sub_i (add_sub[i]),
                .res_o (add_res[i])
            );
        end
    endgenerate

    // -------------------------------------------------------------------------
    // 3. 2-Stage Multiplier-Less PWL Sigmoid (uses add_a/add_b? No, dedicated
    //    lightweight 2-stage exp-shift PWL: Stage1=|x|-x_off, Stage2=b+2^-s*dx)
    // -------------------------------------------------------------------------
    // Stage 1 (combinational on a_fp20_q): compute |x| - x_off using addsub,
    // registered into sig_dx_q + sig_meta_q. Stage 2 uses addsub -> sig_fp20_q.
    // Wait: we can reuse GEN_SHARED_ALU adders for PWL Stage 1 & Stage 2!
    // In SwiGLU, adders are completely idle, so ZERO extra adders needed!
    localparam logic [19:0] FP20_ZERO = 20'h00000;
    localparam logic [19:0] FP20_ONE  = 20'h3F800; // 1.0
    localparam logic [19:0] FP20_2P5  = 20'h40200; // 2.5
    localparam logic [19:0] FP20_4P5  = 20'h40900; // 4.5
    localparam logic [19:0] PWL_B0    = 20'h3F000; // 0.500
    localparam logic [19:0] PWL_B1    = 20'h3F3B2; // 0.731
    localparam logic [19:0] PWL_B2    = 20'h3F6C9; // 0.924

    logic [ALU_LANES-1:0][19:0] pwl_x_off, pwl_intercept_comb, pwl_intercept_q;
    logic [ALU_LANES-1:0][2:0]  pwl_rshift_comb, pwl_rshift_q;
    logic [ALU_LANES-1:0]       pwl_xsign_q, pwl_sat_q;
    logic [ALU_LANES-1:0][19:0] pwl_scaled_dx;

    always_comb begin
        for (int l = 0; l < ALU_LANES; l++) begin
            logic [19:0] abs_x;
            abs_x = {1'b0, a_fp20_q[l][18:0]};
            if (abs_x < FP20_ONE) begin
                pwl_x_off[l]          = FP20_ZERO;
                pwl_intercept_comb[l] = PWL_B0;
                pwl_rshift_comb[l]    = 3'd2;
            end else if (abs_x < FP20_2P5) begin
                pwl_x_off[l]          = FP20_ONE;
                pwl_intercept_comb[l] = PWL_B1;
                pwl_rshift_comb[l]    = 3'd3;
            end else if (abs_x < FP20_4P5) begin
                pwl_x_off[l]          = FP20_2P5;
                pwl_intercept_comb[l] = PWL_B2;
                pwl_rshift_comb[l]    = 3'd5;
            end else begin
                pwl_x_off[l]          = FP20_ZERO;
                pwl_intercept_comb[l] = FP20_ONE;
                pwl_rshift_comb[l]    = 3'd0;
            end

            // Stage 2 exponent-shift on add_res_q[l]
            if (pwl_rshift_q[l] == 3'd0 || add_res_q[l][18:11] <= {5'd0, pwl_rshift_q[l]}) begin
                pwl_scaled_dx[l] = FP20_ZERO;
            end else begin
                pwl_scaled_dx[l] = {1'b0, add_res_q[l][18:11] - {5'd0, pwl_rshift_q[l]}, add_res_q[l][10:0]};
            end
        end
    end

    // -------------------------------------------------------------------------
    // 4. Shared rsqrt LUT (0 multipliers, 16-entry ROM + exp calc)
    // -------------------------------------------------------------------------
    logic [19:0] rsqrt_x_q, rsqrt_y0_q;
    logic [19:0] rsqrt_y0_comb, rsqrt_half_x_comb;

    always_comb begin
        logic        e_lsb;
        logic [3:0]  m_idx;
        logic signed [8:0] k_unbiased, y0_exp_s;
        logic [7:0]  lut_exp;
        logic [10:0] lut_mant;

        e_lsb = rsqrt_x_q[11];
        m_idx = rsqrt_x_q[10:7];
        k_unbiased = $signed({1'b0, rsqrt_x_q[18:11]}) - 9'sd127;
        if (e_lsb) begin
            y0_exp_s = 9'sd127 - (k_unbiased >>> 1) - ((m_idx == 4'd0) ? 9'sd0 : 9'sd1);
        end else begin
            y0_exp_s = 9'sd127 - ((k_unbiased - 9'sd1) >>> 1) - 9'sd1;
        end
        lut_exp = (y0_exp_s <= 9'sd0) ? 8'd1 : (y0_exp_s >= 9'sd254 ? 8'd254 : y0_exp_s[7:0]);

        if (e_lsb) begin
            case (m_idx)
                4'd0:  lut_mant = 11'h000; 4'd1:  lut_mant = 11'h737;
                4'd2:  lut_mant = 11'h680; 4'd3:  lut_mant = 11'h5E4;
                4'd4:  lut_mant = 11'h559; 4'd5:  lut_mant = 11'h4DD;
                4'd6:  lut_mant = 11'h46F; 4'd7:  lut_mant = 11'h40C;
                4'd8:  lut_mant = 11'h3B3; 4'd9:  lut_mant = 11'h363;
                4'd10: lut_mant = 11'h31A; 4'd11: lut_mant = 11'h2D9;
                4'd12: lut_mant = 11'h29D; 4'd13: lut_mant = 11'h267;
                4'd14: lut_mant = 11'h236; default: lut_mant = 11'h20A;
            endcase
        end else begin
            case (m_idx)
                4'd0:  lut_mant = 11'h324; 4'd1:  lut_mant = 11'h2BB;
                4'd2:  lut_mant = 11'h25A; 4'd3:  lut_mant = 11'h201;
                4'd4:  lut_mant = 11'h1AE; 4'd5:  lut_mant = 11'h163;
                4'd6:  lut_mant = 11'h11C; 4'd7:  lut_mant = 11'h0DA;
                4'd8:  lut_mant = 11'h09D; 4'd9:  lut_mant = 11'h063;
                4'd10: lut_mant = 11'h02D; 4'd11: lut_mant = 11'h0C0;
                4'd12: lut_mant = 11'h090; 4'd13: lut_mant = 11'h060;
                4'd14: lut_mant = 11'h035; default: lut_mant = 11'h010;
            endcase
        end
        rsqrt_y0_comb     = {1'b0, lut_exp, lut_mant};
        rsqrt_half_x_comb = {1'b0, (rsqrt_x_q[18:11] > 8'd1 ? (rsqrt_x_q[18:11] - 8'd1) : 8'd0), rsqrt_x_q[10:0]};
    end

    // -------------------------------------------------------------------------
    // 5. Pack FP20 -> FP16 (reads ONLY registered add_res_q or mul_res_q)
    // -------------------------------------------------------------------------
    logic [ALU_LANES-1:0][19:0] pack_in_fp20;
    logic [ALU_BITS-1:0]        slice_fp16_comb;
    logic                       pack_from_add;

    generate
        for (i = 0; i < ALU_LANES; i++) begin : GEN_PACK
            assign pack_in_fp20[i] = pack_from_add
                                   ? ((cfg_op_i == SFU_ROPE) ? add_res_q[i/2] : add_res_q[i])
                                   : mul_res_q[i];
            sfu_fp20_to_fp16 u_pack (.fp20_i(pack_in_fp20[i]), .fp16_o(slice_fp16_comb[i*16 +: 16]));
        end
    endgenerate

    // -------------------------------------------------------------------------
    // 6. Unified Pipelined Control FSM
    // -------------------------------------------------------------------------
    typedef enum logic [4:0] {
        ST_IDLE       = 5'd0,
        ST_UNPACK     = 5'd1,
        // Residual
        ST_RES_ADD    = 5'd2,
        ST_PACK       = 5'd3,
        // RoPE
        ST_ROPE_M0    = 5'd4,
        ST_ROPE_A0_M1 = 5'd5,
        ST_ROPE_P0_A1 = 5'd6,
        ST_ROPE_P1    = 5'd7,
        // SwiGLU
        ST_SWI_PWL1   = 5'd8,
        ST_SWI_PWL2   = 5'd9,
        ST_SWI_NEG    = 5'd10,
        ST_SWI_MUL1   = 5'd11,
        ST_SWI_MUL2   = 5'd12,
        // RMSNorm
        ST_RMS_P1_SQ  = 5'd13,
        ST_RMS_P1_TR1 = 5'd14,
        ST_RMS_P1_TR2 = 5'd15,
        ST_RMS_P1_ACC = 5'd16
    } pipe_st_t;

    typedef enum logic [2:0] {
        RMS_IDLE  = 3'd0,
        RMS_PASS1 = 3'd1,
        RMS_RSQRT = 3'd2,
        RMS_PASS2 = 3'd3
    } rms_mode_t;

    pipe_st_t  st_q;
    rms_mode_t rms_mode_q;
    logic [3:0]  sub_step_q; // Sub-steps for tree L3 / rsqrt / pass2
    logic [15:0] vec_cnt_q;
    logic [19:0] rms_sum_sq_q, inv_rms_q;

    // Operand muxing for GEN_SHARED_ALU (reads ONLY registered state!)
    always_comb begin
        for (int l = 0; l < ALU_LANES; l++) begin
            mul_a[l]   = a_fp20_q[l];
            mul_b[l]   = b_fp20_q[l];
            add_a[l]   = a_fp20_q[l];
            add_b[l]   = b_fp20_q[l];
            add_sub[l] = 1'b0;
        end
        pack_from_add = 1'b0;

        if (cfg_op_i == SFU_RESIDUAL) begin
            pack_from_add = 1'b1;
            // ST_RES_ADD uses default add_a = a_fp20_q, add_b = b_fp20_q
        end else if (cfg_op_i == SFU_ROPE) begin
            pack_from_add = 1'b1;
            if (st_q == ST_ROPE_M0) begin
                for (int k = 0; k < ALU_LANES/2; k++) begin
                    mul_a[2*k]   = a_fp20_q[2*k];   // xe
                    mul_b[2*k]   = b_fp20_q[2*k];   // cos
                    mul_a[2*k+1] = a_fp20_q[2*k+1]; // xo
                    mul_b[2*k+1] = b_fp20_q[2*k+1]; // sin
                end
            end else if (st_q == ST_ROPE_A0_M1) begin
                for (int k = 0; k < ALU_LANES/2; k++) begin
                    add_a[k]     = mul_res_q[2*k];
                    add_b[k]     = mul_res_q[2*k+1];
                    add_sub[k]   = 1'b1;            // xe*cos - xo*sin
                    mul_a[2*k]   = a_fp20_q[2*k];   // xe
                    mul_b[2*k]   = b_fp20_q[2*k+1]; // sin
                    mul_a[2*k+1] = a_fp20_q[2*k+1]; // xo
                    mul_b[2*k+1] = b_fp20_q[2*k];   // cos
                end
            end else if (st_q == ST_ROPE_P0_A1) begin
                for (int k = 0; k < ALU_LANES/2; k++) begin
                    add_a[k]     = mul_res_q[2*k];
                    add_b[k]     = mul_res_q[2*k+1];
                    add_sub[k]   = 1'b0;            // xe*sin + xo*cos
                end
            end
        end else if (cfg_op_i == SFU_SWIGLU) begin
            pack_from_add = 1'b0;
            if (st_q == ST_SWI_PWL1) begin
                for (int l = 0; l < ALU_LANES; l++) begin
                    add_a[l]   = {1'b0, a_fp20_q[l][18:0]};
                    add_b[l]   = pwl_x_off[l];
                    add_sub[l] = 1'b1; // |x| - x_off
                end
            end else if (st_q == ST_SWI_PWL2) begin
                for (int l = 0; l < ALU_LANES; l++) begin
                    add_a[l]   = pwl_intercept_q[l];
                    add_b[l]   = pwl_scaled_dx[l];
                    add_sub[l] = 1'b0;
                end
            end else if (st_q == ST_SWI_NEG) begin
                for (int l = 0; l < ALU_LANES; l++) begin
                    add_a[l]   = FP20_ONE;
                    add_b[l]   = pwl_sat_q[l] ? FP20_ONE : add_res_q[l];
                    add_sub[l] = 1'b1; // 1.0 - sig_pos
                end
            end else if (st_q == ST_SWI_MUL1) begin
                for (int l = 0; l < ALU_LANES; l++) begin
                    mul_a[l] = a_fp20_q[l];
                    mul_b[l] = add_res_q[l];
                end
            end else if (st_q == ST_SWI_MUL2) begin
                for (int l = 0; l < ALU_LANES; l++) begin
                    mul_a[l] = mul_res_q[l]; // silu(x)
                    mul_b[l] = b_fp20_q[l];  // up
                end
            end
        end else if (cfg_op_i == SFU_RMSNORM) begin
            pack_from_add = 1'b0;
            if (rms_mode_q == RMS_PASS1) begin
                if (st_q == ST_RMS_P1_SQ) begin
                    for (int l = 0; l < ALU_LANES; l++) begin
                        mul_a[l] = a_fp20_q[l];
                        mul_b[l] = a_fp20_q[l];
                    end
                end else if (st_q == ST_RMS_P1_TR1) begin
                    for (int k = 0; k < ALU_LANES/2; k++) begin
                        add_a[k] = mul_res_q[2*k];
                        add_b[k] = mul_res_q[2*k+1];
                    end
                end else if (st_q == ST_RMS_P1_TR2) begin
                    for (int k = 0; k < ALU_LANES/4; k++) begin
                        add_a[k] = add_res_q[2*k];
                        add_b[k] = add_res_q[2*k+1];
                    end
                end else if (st_q == ST_RMS_P1_ACC) begin
                    if (ALU_LANES == 16 && sub_step_q == 4'd0) begin
                        add_a[0] = add_res_q[0]; add_b[0] = add_res_q[1];
                        add_a[1] = add_res_q[2]; add_b[1] = add_res_q[3];
                    end else if ((ALU_LANES == 16 && sub_step_q == 4'd1) || ((ALU_LANES == 8 || ALU_LANES == 4) && sub_step_q == 4'd0)) begin
                        add_a[0] = add_res_q[0]; add_b[0] = add_res_q[1];
                    end else begin
                        add_a[0] = rms_sum_sq_q; add_b[0] = add_res_q[0];
                    end
                end
            end else if (rms_mode_q == RMS_RSQRT) begin
                if (sub_step_q == 4'd0) begin
                    mul_a[0] = rms_sum_sq_q;
                    mul_b[0] = inv_n_fp20;
                end else if (sub_step_q == 4'd1) begin
                    add_a[0] = mul_res_q[0];
                    add_b[0] = eps_fp20;
                end else if (sub_step_q == 4'd2) begin
                    mul_a[0] = rsqrt_y0_comb;
                    mul_b[0] = rsqrt_y0_comb;
                end else if (sub_step_q == 4'd3) begin
                    mul_a[0] = rsqrt_half_x_comb;
                    mul_b[0] = mul_res_q[0];
                end else if (sub_step_q == 4'd4) begin
                    add_a[0]   = 20'h3FC00; // 1.5
                    add_b[0]   = mul_res_q[0];
                    add_sub[0] = 1'b1;
                end else if (sub_step_q == 4'd5) begin
                    mul_a[0] = rsqrt_y0_q;
                    mul_b[0] = add_res_q[0];
                end
            end else if (rms_mode_q == RMS_PASS2) begin
                if (sub_step_q == 4'd0) begin
                    for (int l = 0; l < ALU_LANES; l++) begin
                        mul_a[l] = a_fp20_q[l];
                        mul_b[l] = inv_rms_q;
                    end
                end else if (sub_step_q == 4'd1) begin
                    for (int l = 0; l < ALU_LANES; l++) begin
                        mul_a[l] = mul_res_q[l];
                        mul_b[l] = b_fp20_q[l];
                    end
                end
            end
        end
    end

    // -------------------------------------------------------------------------
    // 7. Handshaking & Sequential Pipeline FSM (100% FF-decoupled interfaces)
    // -------------------------------------------------------------------------
    logic has_a_q, has_b_q;
    wire  last_chunk = (NUM_CHUNKS == 1) || (chunk_q == CHUNK_W'(NUM_CHUNKS - 1));

    always_comb begin
        in_a_ready_o = 1'b0;
        in_b_ready_o = 1'b0;
        if (cfg_op_i == SFU_RMSNORM) begin
            if (rms_mode_q == RMS_PASS1 && st_q == ST_IDLE && !active_vec_q) begin
                in_a_ready_o = 1'b1;
            end else if (rms_mode_q == RMS_PASS2 && st_q == ST_IDLE && !active_vec_q) begin
                in_a_ready_o = !has_a_q;
                in_b_ready_o = !has_b_q;
            end
        end else if (cfg_op_i == SFU_BYPASS) begin
            in_a_ready_o = !out_valid_o;
        end else if (st_q == ST_IDLE && !active_vec_q) begin
            in_a_ready_o = !has_a_q;
            in_b_ready_o = !has_b_q;
        end
    end

    wire accept_a = in_a_valid_i && in_a_ready_o;
    wire accept_b = in_b_valid_i && in_b_ready_o;
    wire got_a    = has_a_q || accept_a;
    wire got_b    = has_b_q || accept_b;

    assign busy_o = (cfg_op_i == SFU_RMSNORM) ? (rms_mode_q != RMS_IDLE)
                                              : (active_vec_q || has_a_q || has_b_q || (st_q != ST_IDLE));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st_q         <= ST_IDLE;
            rms_mode_q   <= RMS_IDLE;
            sub_step_q   <= 4'd0;
            chunk_q      <= '0;
            active_vec_q <= 1'b0;
            has_a_q      <= 1'b0;
            has_b_q      <= 1'b0;
            vec_cnt_q    <= 16'd0;
            rms_sum_sq_q <= 20'd0;
            inv_rms_q    <= 20'd0;
            rsqrt_x_q    <= 20'd0;
            rsqrt_y0_q   <= 20'd0;
            in_a_hold_q  <= '0;
            in_b_hold_q  <= '0;
            out_valid_o  <= 1'b0;
            out_fp16_o   <= '0;
            done_o       <= 1'b0;
            for (int l = 0; l < ALU_LANES; l++) begin
                a_fp20_q[l]        <= 20'd0;
                b_fp20_q[l]        <= 20'd0;
                mul_res_q[l]       <= 20'd0;
                add_res_q[l]       <= 20'd0;
                pwl_intercept_q[l] <= 20'd0;
                pwl_rshift_q[l]    <= 3'd0;
                pwl_xsign_q[l]     <= 1'b0;
                pwl_sat_q[l]       <= 1'b0;
            end
        end else begin
            done_o <= 1'b0;
            if (out_valid_o && out_ready_i) begin
                out_valid_o <= 1'b0;
            end

            if (cfg_op_i != SFU_RMSNORM) begin
                rms_mode_q <= RMS_IDLE;

                if (cfg_op_i == SFU_BYPASS) begin
                    st_q         <= ST_IDLE;
                    active_vec_q <= 1'b0;
                    has_a_q      <= 1'b0;
                    has_b_q      <= 1'b0;
                    chunk_q      <= '0;
                    if (in_a_valid_i && !out_valid_o) begin
                        out_valid_o <= 1'b1;
                        out_fp16_o  <= in_a_fp16_i;
                    end
                end else begin
                    case (st_q)
                        ST_IDLE: begin
                            if (!active_vec_q) begin
                                if (accept_a) begin
                                    in_a_hold_q <= in_a_fp16_i;
                                    has_a_q     <= 1'b1;
                                end
                                if (accept_b) begin
                                    in_b_hold_q <= in_b_fp16_i;
                                    has_b_q     <= 1'b1;
                                end
                                if (got_a && got_b) begin
                                    active_vec_q <= 1'b1;
                                    has_a_q      <= 1'b0;
                                    has_b_q      <= 1'b0;
                                    chunk_q      <= '0;
                                    st_q         <= ST_UNPACK;
                                end
                            end else begin
                                st_q <= ST_UNPACK;
                            end
                        end

                        ST_UNPACK: begin
                            a_fp20_q <= a_fp20_comb;
                            b_fp20_q <= b_fp20_comb;
                            if      (cfg_op_i == SFU_RESIDUAL) st_q <= ST_RES_ADD;
                            else if (cfg_op_i == SFU_ROPE)     st_q <= ST_ROPE_M0;
                            else                               st_q <= ST_SWI_PWL1;
                        end

                        // --- Residual ---
                        ST_RES_ADD: begin
                            add_res_q <= add_res;
                            st_q      <= ST_PACK;
                        end
                        ST_PACK: begin
                            if (!out_valid_o) begin
                                out_fp16_o[chunk_q * ALU_BITS +: ALU_BITS] <= slice_fp16_comb;
                                if (last_chunk) begin
                                    active_vec_q <= 1'b0;
                                    chunk_q      <= '0;
                                    out_valid_o  <= 1'b1;
                                    st_q         <= ST_IDLE;
                                end else begin
                                    chunk_q <= chunk_q + 1'b1;
                                    st_q    <= ST_UNPACK;
                                end
                            end
                        end

                        // --- RoPE ---
                        ST_ROPE_M0: begin
                            mul_res_q <= mul_res;
                            st_q      <= ST_ROPE_A0_M1;
                        end
                        ST_ROPE_A0_M1: begin
                            add_res_q <= add_res;
                            mul_res_q <= mul_res;
                            st_q      <= ST_ROPE_P0_A1;
                        end
                        ST_ROPE_P0_A1: begin
                            if (!out_valid_o) begin
                                for (int k = 0; k < ALU_LANES/2; k++) begin
                                    out_fp16_o[(chunk_q*ALU_LANES + 2*k)*16 +: 16] <= slice_fp16_comb[(2*k)*16 +: 16];
                                end
                                add_res_q <= add_res;
                                st_q      <= ST_ROPE_P1;
                            end
                        end
                        ST_ROPE_P1: begin
                            for (int k = 0; k < ALU_LANES/2; k++) begin
                                out_fp16_o[(chunk_q*ALU_LANES + 2*k + 1)*16 +: 16] <= slice_fp16_comb[(2*k+1)*16 +: 16];
                            end
                            if (last_chunk) begin
                                active_vec_q <= 1'b0;
                                chunk_q      <= '0;
                                out_valid_o  <= 1'b1;
                                st_q         <= ST_IDLE;
                            end else begin
                                chunk_q <= chunk_q + 1'b1;
                                st_q    <= ST_UNPACK;
                            end
                        end

                        // --- SwiGLU ---
                        ST_SWI_PWL1: begin
                            add_res_q <= add_res;
                            for (int l = 0; l < ALU_LANES; l++) begin
                                pwl_intercept_q[l] <= pwl_intercept_comb[l];
                                pwl_rshift_q[l]    <= pwl_rshift_comb[l];
                                pwl_xsign_q[l]     <= a_fp20_q[l][19];
                                pwl_sat_q[l]       <= ({1'b0, a_fp20_q[l][18:0]} >= FP20_4P5);
                            end
                            st_q <= ST_SWI_PWL2;
                        end
                        ST_SWI_PWL2: begin
                            for (int l = 0; l < ALU_LANES; l++) begin
                                add_res_q[l] <= pwl_sat_q[l] ? FP20_ONE : add_res[l];
                            end
                            st_q <= ST_SWI_NEG;
                        end
                        ST_SWI_NEG: begin
                            for (int l = 0; l < ALU_LANES; l++) begin
                                if (pwl_xsign_q[l]) add_res_q[l] <= add_res[l];
                            end
                            st_q <= ST_SWI_MUL1;
                        end
                        ST_SWI_MUL1: begin
                            mul_res_q <= mul_res;
                            st_q      <= ST_SWI_MUL2;
                        end
                        ST_SWI_MUL2: begin
                            mul_res_q <= mul_res;
                            st_q      <= ST_PACK;
                        end

                        default: st_q <= ST_IDLE;
                    endcase
                end
            end else begin
                // --- RMSNorm 2-Pass Pipelined FSM ---
                case (rms_mode_q)
                    RMS_IDLE: begin
                        st_q         <= ST_IDLE;
                        active_vec_q <= 1'b0;
                        has_a_q      <= 1'b0;
                        has_b_q      <= 1'b0;
                        chunk_q      <= '0;
                        sub_step_q   <= 4'd0;
                        if (start_i) begin
                            rms_mode_q   <= RMS_PASS1;
                            vec_cnt_q    <= 16'd0;
                            rms_sum_sq_q <= 20'd0;
                        end
                    end

                    RMS_PASS1: begin
                        has_a_q <= 1'b0;
                        has_b_q <= 1'b0;
                        case (st_q)
                            ST_IDLE: begin
                                if (in_a_valid_i) begin
                                    in_a_hold_q  <= in_a_fp16_i;
                                    active_vec_q <= 1'b1;
                                    chunk_q      <= '0;
                                    st_q         <= ST_UNPACK;
                                end
                            end
                            ST_UNPACK: begin
                                a_fp20_q <= a_fp20_comb;
                                st_q     <= ST_RMS_P1_SQ;
                            end
                            ST_RMS_P1_SQ: begin
                                mul_res_q <= mul_res;
                                st_q      <= ST_RMS_P1_TR1;
                            end
                            ST_RMS_P1_TR1: begin
                                add_res_q <= add_res;
                                if (ALU_LANES >= 8) begin
                                    st_q <= ST_RMS_P1_TR2;
                                end else begin
                                    sub_step_q <= 4'd0;
                                    st_q       <= ST_RMS_P1_ACC;
                                end
                            end
                            ST_RMS_P1_TR2: begin
                                add_res_q  <= add_res;
                                sub_step_q <= 4'd0;
                                st_q       <= ST_RMS_P1_ACC;
                            end
                            ST_RMS_P1_ACC: begin
                                if ((ALU_LANES == 16 && sub_step_q < 4'd2) || ((ALU_LANES == 8 || ALU_LANES == 4) && sub_step_q == 4'd0)) begin
                                    add_res_q  <= add_res;
                                    sub_step_q <= sub_step_q + 4'd1;
                                end else begin
                                    rms_sum_sq_q <= add_res[0];
                                    sub_step_q   <= 4'd0;
                                    if (last_chunk) begin
                                        active_vec_q <= 1'b0;
                                        chunk_q      <= '0;
                                        st_q         <= ST_IDLE;
                                        if (vec_cnt_q + 16'd1 >= cfg_num_vecs_i) begin
                                            rms_mode_q <= RMS_RSQRT;
                                            vec_cnt_q  <= 16'd0;
                                        end else begin
                                            vec_cnt_q <= vec_cnt_q + 16'd1;
                                        end
                                    end else begin
                                        chunk_q <= chunk_q + 1'b1;
                                        st_q    <= ST_UNPACK;
                                    end
                                end
                            end
                            default: st_q <= ST_IDLE;
                        endcase
                    end

                    RMS_RSQRT: begin
                        if (sub_step_q == 4'd0) begin
                            mul_res_q  <= mul_res; // sum_sq * inv_n
                            sub_step_q <= 4'd1;
                        end else if (sub_step_q == 4'd1) begin
                            rsqrt_x_q  <= add_res[0]; // + eps
                            sub_step_q <= 4'd2;
                        end else if (sub_step_q == 4'd2) begin
                            rsqrt_y0_q <= rsqrt_y0_comb;
                            mul_res_q  <= mul_res; // y0 * y0
                            sub_step_q <= 4'd3;
                        end else if (sub_step_q == 4'd3) begin
                            mul_res_q  <= mul_res; // half_x * y0_sq
                            sub_step_q <= 4'd4;
                        end else if (sub_step_q == 4'd4) begin
                            add_res_q  <= add_res; // 1.5 - half_x * y0_sq
                            sub_step_q <= 4'd5;
                        end else begin
                            inv_rms_q  <= (rsqrt_x_q[18:11] == 8'd0) ? 20'd0 : mul_res[0];
                            sub_step_q <= 4'd0;
                            rms_mode_q <= RMS_PASS2;
                            st_q       <= ST_IDLE;
                        end
                    end

                    RMS_PASS2: begin
                        case (st_q)
                            ST_IDLE: begin
                                if (!active_vec_q) begin
                                    if (accept_a) begin
                                        in_a_hold_q <= in_a_fp16_i;
                                        has_a_q     <= 1'b1;
                                    end
                                    if (accept_b) begin
                                        in_b_hold_q <= in_b_fp16_i;
                                        has_b_q     <= 1'b1;
                                    end
                                    if (got_a && got_b) begin
                                        active_vec_q <= 1'b1;
                                        has_a_q      <= 1'b0;
                                        has_b_q      <= 1'b0;
                                        chunk_q      <= '0;
                                        sub_step_q   <= 4'd0;
                                        st_q         <= ST_UNPACK;
                                    end
                                end else begin
                                    st_q <= ST_UNPACK;
                                end
                            end
                            ST_UNPACK: begin
                                a_fp20_q   <= a_fp20_comb;
                                b_fp20_q   <= b_fp20_comb;
                                sub_step_q <= 4'd0;
                                st_q       <= ST_SWI_MUL1; // reuse MUL1/MUL2 sequence for x*inv_rms then *gamma
                            end
                            ST_SWI_MUL1: begin
                                mul_res_q  <= mul_res; // x * inv_rms
                                sub_step_q <= 4'd1;
                                st_q       <= ST_SWI_MUL2;
                            end
                            ST_SWI_MUL2: begin
                                mul_res_q  <= mul_res; // * gamma
                                sub_step_q <= 4'd0;
                                st_q       <= ST_PACK;
                            end
                            ST_PACK: begin
                                if (!out_valid_o) begin
                                    out_fp16_o[chunk_q * ALU_BITS +: ALU_BITS] <= slice_fp16_comb;
                                    if (last_chunk) begin
                                        active_vec_q <= 1'b0;
                                        chunk_q      <= '0;
                                        out_valid_o  <= 1'b1;
                                        st_q         <= ST_IDLE;
                                        if (vec_cnt_q + 16'd1 >= cfg_num_vecs_i) begin
                                            rms_mode_q <= RMS_IDLE;
                                            done_o     <= 1'b1;
                                        end else begin
                                            vec_cnt_q <= vec_cnt_q + 16'd1;
                                        end
                                    end else begin
                                        chunk_q <= chunk_q + 1'b1;
                                        st_q    <= ST_UNPACK;
                                    end
                                end
                            end
                            default: st_q <= ST_IDLE;
                        endcase
                    end

                    default: rms_mode_q <= RMS_IDLE;
                endcase
            end
        end
    end

endmodule
