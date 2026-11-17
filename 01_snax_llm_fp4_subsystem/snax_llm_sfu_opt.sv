// =============================================================================
// Module: snax_llm_sfu_opt.sv
// Description: Parameterized Area-Minimized SIMD SFU (ALU_LANES = 16, 8, or 4)
//              for SNAX FP4 LLM Subsystem.
//              External interface stays fixed at 256-bit (16 x FP16) valid/ready.
//              Internal ALU instantiates ONLY ALU_LANES FP20 multipliers/adders,
//              unpackers, packers, and PWL sigmoids, time-multiplexed over
//              NUM_CHUNKS = 16 / ALU_LANES chunks:
//                - ALU_LANES = 16 : 1 chunk  (1x ALU, max throughput)
//                - ALU_LANES = 8  : 2 chunks (0.5x ALU, matches 64b TCDM BW)
//                - ALU_LANES = 4  : 4 chunks (0.25x ALU, ultra-low area)
// =============================================================================
`timescale 1ns/1ps

module snax_llm_sfu_opt #(
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
    // 1. Input Vector Holding Registers & Active Chunk Slice
    // -------------------------------------------------------------------------
    logic [NUM_LANES*16-1:0] in_a_hold_q, in_b_hold_q;
    logic [CHUNK_W-1:0]      chunk_q;
    logic                    phase_q;
    logic                    active_vec_q; // High while processing chunks > 0 or phase 1

    // On the very first cycle of a vector (!active_vec_q), read directly from
    // input ports; on subsequent cycles, read from the held 256-bit registers.
    wire [NUM_LANES*16-1:0] cur_a_vec = active_vec_q ? in_a_hold_q : in_a_fp16_i;
    wire [NUM_LANES*16-1:0] cur_b_vec = active_vec_q ? in_b_hold_q : in_b_fp16_i;

    logic [ALU_BITS-1:0] cur_a_slice, cur_b_slice;
    always_comb begin
        if (NUM_CHUNKS == 1) begin
            cur_a_slice = cur_a_vec[ALU_BITS-1:0];
            cur_b_slice = cur_b_vec[ALU_BITS-1:0];
        end else begin
            cur_a_slice = cur_a_vec[chunk_q * ALU_BITS +: ALU_BITS];
            cur_b_slice = cur_b_vec[chunk_q * ALU_BITS +: ALU_BITS];
        end
    end

    // Unpack ONLY ALU_LANES elements per cycle -> FP20
    logic [ALU_LANES-1:0][19:0] a_fp20, b_fp20, sig_fp20;
    genvar i;
    generate
        for (i = 0; i < ALU_LANES; i++) begin : GEN_UNP
            sfu_fp16_to_fp20     u_ua  (.fp16_i(cur_a_slice[i*16 +: 16]), .fp20_o(a_fp20[i]));
            sfu_fp16_to_fp20     u_ub  (.fp16_i(cur_b_slice[i*16 +: 16]), .fp20_o(b_fp20[i]));
            sfu_sigmoid_pwl_fp20 u_sig (.x_i(a_fp20[i]),                  .sig_o(sig_fp20[i]));
        end
    endgenerate

    wire [19:0] inv_n_fp20 = {cfg_inv_n_fp32_i[31], cfg_inv_n_fp32_i[30:23], cfg_inv_n_fp32_i[22:12]};
    wire [19:0] eps_fp20   = {cfg_eps_fp32_i[31],   cfg_eps_fp32_i[30:23],   cfg_eps_fp32_i[22:12]};

    // -------------------------------------------------------------------------
    // 2. RMSNorm FSM & Temporary Holding Registers
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {
        ST_IDLE      = 2'd0,
        ST_RMS_PASS1 = 2'd1,
        ST_RMS_RSQRT = 2'd2,
        ST_RMS_PASS2 = 2'd3
    } sfu_rms_st_t;

    sfu_rms_st_t rms_st;
    logic [15:0] vec_cnt;
    logic [19:0] rms_sum_sq_q, inv_rms_q;
    logic [ALU_LANES-1:0][19:0] tmp_chunk_q;

    // -------------------------------------------------------------------------
    // 3. Shared ALU_LANES x FP20 Multipliers + ALU_LANES x FP20 Adders
    // -------------------------------------------------------------------------
    logic [ALU_LANES-1:0][19:0] mul_a, mul_b, mul_res;
    logic [ALU_LANES-1:0][19:0] add_a, add_b, add_res;
    logic [ALU_LANES-1:0]       add_sub;

    always_comb begin
        for (int l = 0; l < ALU_LANES; l++) begin
            mul_a[l]   = a_fp20[l];
            mul_b[l]   = b_fp20[l];
            add_a[l]   = a_fp20[l];
            add_b[l]   = b_fp20[l];
            add_sub[l] = 1'b0;
        end

        if (cfg_op_i == SFU_RMSNORM) begin
            if (rms_st == ST_RMS_PASS1) begin
                // ALU_LANES squares + (ALU_LANES - 1) tree adders + 1 accumulator adder
                for (int l = 0; l < ALU_LANES; l++) begin
                    mul_a[l] = a_fp20[l];
                    mul_b[l] = a_fp20[l];
                end
                // Tree Level 1: ALU_LANES/2 adders [0 .. ALU_LANES/2 - 1]
                for (int k = 0; k < ALU_LANES/2; k++) begin
                    add_a[k] = mul_res[2*k];
                    add_b[k] = mul_res[2*k+1];
                end
                // Tree Level 2 (if ALU_LANES >= 8): adders [ALU_LANES/2 .. 3*ALU_LANES/4 - 1]
                if (ALU_LANES >= 8) begin
                    for (int k = 0; k < ALU_LANES/4; k++) begin
                        add_a[ALU_LANES/2 + k] = add_res[2*k];
                        add_b[ALU_LANES/2 + k] = add_res[2*k+1];
                    end
                end
                // Tree Level 3 (if ALU_LANES >= 16): adders [12..13]
                if (ALU_LANES >= 16) begin
                    add_a[12] = add_res[8];
                    add_b[12] = add_res[9];
                    add_a[13] = add_res[10];
                    add_b[13] = add_res[11];
                    add_a[14] = add_res[12];
                    add_b[14] = add_res[13];
                end else if (ALU_LANES == 8) begin
                    add_a[6] = add_res[4];
                    add_b[6] = add_res[5];
                end else begin // ALU_LANES == 4
                    add_a[2] = add_res[0];
                    add_b[2] = add_res[1];
                end
                // Accumulator adder at index [ALU_LANES - 1]
                add_a[ALU_LANES - 1] = rms_sum_sq_q;
                add_b[ALU_LANES - 1] = add_res[ALU_LANES - 2];
            end else if (rms_st == ST_RMS_RSQRT) begin
                mul_a[0] = rms_sum_sq_q;
                mul_b[0] = inv_n_fp20;
                add_a[0] = mul_res[0];
                add_b[0] = eps_fp20;
            end else if (rms_st == ST_RMS_PASS2) begin
                if (!phase_q) begin
                    for (int l = 0; l < ALU_LANES; l++) begin
                        mul_a[l] = a_fp20[l];
                        mul_b[l] = inv_rms_q;
                    end
                end else begin
                    for (int l = 0; l < ALU_LANES; l++) begin
                        mul_a[l] = tmp_chunk_q[l];
                        mul_b[l] = b_fp20[l];
                    end
                end
            end
        end else if (cfg_op_i == SFU_ROPE) begin
            if (!phase_q) begin
                for (int k = 0; k < ALU_LANES/2; k++) begin
                    mul_a[2*k]   = a_fp20[2*k];   // xe
                    mul_b[2*k]   = b_fp20[2*k];   // cos
                    mul_a[2*k+1] = a_fp20[2*k+1]; // xo
                    mul_b[2*k+1] = b_fp20[2*k+1]; // sin
                    add_a[k]     = mul_res[2*k];
                    add_b[k]     = mul_res[2*k+1];
                    add_sub[k]   = 1'b1;          // xe*cos - xo*sin
                end
            end else begin
                for (int k = 0; k < ALU_LANES/2; k++) begin
                    mul_a[2*k]   = a_fp20[2*k];   // xe
                    mul_b[2*k]   = b_fp20[2*k+1]; // sin
                    mul_a[2*k+1] = a_fp20[2*k+1]; // xo
                    mul_b[2*k+1] = b_fp20[2*k];   // cos
                    add_a[k]     = mul_res[2*k];
                    add_b[k]     = mul_res[2*k+1];
                    add_sub[k]   = 1'b0;          // xe*sin + xo*cos
                end
            end
        end else if (cfg_op_i == SFU_SWIGLU) begin
            if (!phase_q) begin
                for (int l = 0; l < ALU_LANES; l++) begin
                    mul_a[l] = a_fp20[l];
                    mul_b[l] = sig_fp20[l];
                end
            end else begin
                for (int l = 0; l < ALU_LANES; l++) begin
                    mul_a[l] = tmp_chunk_q[l];
                    mul_b[l] = b_fp20[l];
                end
            end
        end
    end

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

    // Shared 1/sqrt unit
    logic [19:0] inv_rms_comb;
    sfu_rsqrt_fp20 u_rsqrt (
        .x_i        (add_res[0]),
        .inv_sqrt_o (inv_rms_comb)
    );

    // Pack ALU_LANES FP20 results -> ALU_BITS FP16 slice
    logic [ALU_LANES-1:0][19:0] out_fp20_mux;
    logic [ALU_BITS-1:0]        slice_fp16_comb;
    generate
        for (i = 0; i < ALU_LANES; i++) begin : GEN_PACK
            always_comb begin
                if (cfg_op_i == SFU_RESIDUAL) begin
                    out_fp20_mux[i] = add_res[i];
                end else if (cfg_op_i == SFU_ROPE) begin
                    out_fp20_mux[i] = add_res[i/2];
                end else begin
                    out_fp20_mux[i] = mul_res[i];
                end
            end
            sfu_fp20_to_fp16 u_pack (.fp20_i(out_fp20_mux[i]), .fp16_o(slice_fp16_comb[i*16 +: 16]));
        end
    endgenerate

    // -------------------------------------------------------------------------
    // 4. Handshaking & Multi-Chunk Sequential Control
    // -------------------------------------------------------------------------
    wire out_can_accept = !out_valid_o || out_ready_i;
    wire is_2phase_op   = (cfg_op_i == SFU_ROPE) || (cfg_op_i == SFU_SWIGLU);
    wire last_chunk     = (NUM_CHUNKS == 1) || (chunk_q == CHUNK_W'(NUM_CHUNKS - 1));

    always_comb begin
        in_a_ready_o = 1'b0;
        in_b_ready_o = 1'b0;
        if (cfg_op_i == SFU_RMSNORM) begin
            if (rms_st == ST_RMS_PASS1) begin
                in_a_ready_o = !active_vec_q;
            end else if (rms_st == ST_RMS_PASS2) begin
                in_a_ready_o = !active_vec_q && in_b_valid_i && out_can_accept;
                in_b_ready_o = !active_vec_q && in_a_valid_i && out_can_accept;
            end
        end else if (cfg_op_i == SFU_BYPASS) begin
            in_a_ready_o = out_can_accept;
        end else begin
            in_a_ready_o = !active_vec_q && in_b_valid_i && out_can_accept;
            in_b_ready_o = !active_vec_q && in_a_valid_i && out_can_accept;
        end
    end

    assign busy_o = (cfg_op_i == SFU_RMSNORM) ? (rms_st != ST_IDLE) : active_vec_q;

    // Helper task/logic to write slice into out_fp16_o
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rms_st       <= ST_IDLE;
            chunk_q      <= '0;
            phase_q      <= 1'b0;
            active_vec_q <= 1'b0;
            vec_cnt      <= 16'd0;
            rms_sum_sq_q <= 20'd0;
            inv_rms_q    <= 20'd0;
            in_a_hold_q  <= '0;
            in_b_hold_q  <= '0;
            out_valid_o  <= 1'b0;
            out_fp16_o   <= '0;
            done_o       <= 1'b0;
            for (int l = 0; l < ALU_LANES; l++) tmp_chunk_q[l] <= 20'd0;
        end else begin
            done_o <= 1'b0;

            if (cfg_op_i != SFU_RMSNORM) begin
                rms_st <= ST_IDLE;
                if (cfg_op_i == SFU_BYPASS) begin
                    active_vec_q <= 1'b0;
                    chunk_q      <= '0;
                    phase_q      <= 1'b0;
                    if (in_a_valid_i && out_can_accept) begin
                        out_valid_o <= 1'b1;
                        out_fp16_o  <= in_a_fp16_i;
                    end else if (out_ready_i) begin
                        out_valid_o <= 1'b0;
                    end
                end else if (cfg_op_i == SFU_RESIDUAL) begin
                    phase_q <= 1'b0;
                    if (!active_vec_q) begin
                        if (out_ready_i) out_valid_o <= 1'b0;
                        if (in_a_valid_i && in_b_valid_i && out_can_accept) begin
                            if (NUM_CHUNKS == 1) begin
                                out_valid_o <= 1'b1;
                                out_fp16_o  <= slice_fp16_comb;
                            end else begin
                                active_vec_q <= 1'b1;
                                chunk_q      <= CHUNK_W'(1);
                                in_a_hold_q  <= in_a_fp16_i;
                                in_b_hold_q  <= in_b_fp16_i;
                                out_fp16_o[0 +: ALU_BITS] <= slice_fp16_comb;
                            end
                        end
                    end else begin
                        out_fp16_o[chunk_q * ALU_BITS +: ALU_BITS] <= slice_fp16_comb;
                        if (last_chunk) begin
                            active_vec_q <= 1'b0;
                            chunk_q      <= '0;
                            out_valid_o  <= 1'b1;
                        end else begin
                            chunk_q <= chunk_q + 1'b1;
                        end
                    end
                end else begin
                    // 2-phase ops (RoPE / SwiGLU) over NUM_CHUNKS
                    if (!active_vec_q) begin
                        if (out_ready_i) out_valid_o <= 1'b0;
                        if (in_a_valid_i && in_b_valid_i && out_can_accept) begin
                            active_vec_q <= 1'b1;
                            chunk_q      <= '0;
                            phase_q      <= 1'b1;
                            in_a_hold_q  <= in_a_fp16_i;
                            in_b_hold_q  <= in_b_fp16_i;
                            if (cfg_op_i == SFU_ROPE) begin
                                for (int k = 0; k < ALU_LANES/2; k++) begin
                                    out_fp16_o[(2*k)*16 +: 16] <= slice_fp16_comb[(2*k)*16 +: 16];
                                end
                            end else begin
                                for (int l = 0; l < ALU_LANES; l++) tmp_chunk_q[l] <= mul_res[l];
                            end
                        end
                    end else if (!phase_q) begin
                        phase_q <= 1'b1;
                        if (cfg_op_i == SFU_ROPE) begin
                            for (int k = 0; k < ALU_LANES/2; k++) begin
                                out_fp16_o[(chunk_q*ALU_LANES + 2*k)*16 +: 16] <= slice_fp16_comb[(2*k)*16 +: 16];
                            end
                        end else begin
                            for (int l = 0; l < ALU_LANES; l++) tmp_chunk_q[l] <= mul_res[l];
                        end
                    end else begin
                        phase_q <= 1'b0;
                        if (cfg_op_i == SFU_ROPE) begin
                            for (int k = 0; k < ALU_LANES/2; k++) begin
                                out_fp16_o[(chunk_q*ALU_LANES + 2*k + 1)*16 +: 16] <= slice_fp16_comb[(2*k+1)*16 +: 16];
                            end
                        end else begin
                            out_fp16_o[chunk_q * ALU_BITS +: ALU_BITS] <= slice_fp16_comb;
                        end

                        if (last_chunk) begin
                            active_vec_q <= 1'b0;
                            chunk_q      <= '0;
                            out_valid_o  <= 1'b1;
                        end else begin
                            chunk_q <= chunk_q + 1'b1;
                        end
                    end
                end
            end else begin
                // 2-Pass Streaming RMSNorm FSM
                case (rms_st)
                    ST_IDLE: begin
                        active_vec_q <= 1'b0;
                        chunk_q      <= '0;
                        phase_q      <= 1'b0;
                        if (out_ready_i) out_valid_o <= 1'b0;
                        if (start_i) begin
                            rms_st       <= ST_RMS_PASS1;
                            vec_cnt      <= 16'd0;
                            rms_sum_sq_q <= 20'd0;
                        end
                    end

                    ST_RMS_PASS1: begin
                        if (!active_vec_q) begin
                            if (in_a_valid_i) begin
                                rms_sum_sq_q <= add_res[ALU_LANES - 1];
                                if (NUM_CHUNKS == 1) begin
                                    if (vec_cnt + 16'd1 >= cfg_num_vecs_i) begin
                                        rms_st  <= ST_RMS_RSQRT;
                                        vec_cnt <= 16'd0;
                                    end else begin
                                        vec_cnt <= vec_cnt + 16'd1;
                                    end
                                end else begin
                                    active_vec_q <= 1'b1;
                                    chunk_q      <= CHUNK_W'(1);
                                    in_a_hold_q  <= in_a_fp16_i;
                                end
                            end
                        end else begin
                            rms_sum_sq_q <= add_res[ALU_LANES - 1];
                            if (last_chunk) begin
                                active_vec_q <= 1'b0;
                                chunk_q      <= '0;
                                if (vec_cnt + 16'd1 >= cfg_num_vecs_i) begin
                                    rms_st  <= ST_RMS_RSQRT;
                                    vec_cnt <= 16'd0;
                                end else begin
                                    vec_cnt <= vec_cnt + 16'd1;
                                end
                            end else begin
                                chunk_q <= chunk_q + 1'b1;
                            end
                        end
                    end

                    ST_RMS_RSQRT: begin
                        inv_rms_q    <= inv_rms_comb;
                        rms_st       <= ST_RMS_PASS2;
                        vec_cnt      <= 16'd0;
                        active_vec_q <= 1'b0;
                        chunk_q      <= '0;
                        phase_q      <= 1'b0;
                    end

                    ST_RMS_PASS2: begin
                        if (!active_vec_q) begin
                            if (out_ready_i) out_valid_o <= 1'b0;
                            if (in_a_valid_i && in_b_valid_i && out_can_accept) begin
                                active_vec_q <= 1'b1;
                                chunk_q      <= '0;
                                phase_q      <= 1'b1;
                                in_a_hold_q  <= in_a_fp16_i;
                                in_b_hold_q  <= in_b_fp16_i;
                                for (int l = 0; l < ALU_LANES; l++) tmp_chunk_q[l] <= mul_res[l];
                            end
                        end else if (!phase_q) begin
                            phase_q <= 1'b1;
                            for (int l = 0; l < ALU_LANES; l++) tmp_chunk_q[l] <= mul_res[l];
                        end else begin
                            phase_q <= 1'b0;
                            out_fp16_o[chunk_q * ALU_BITS +: ALU_BITS] <= slice_fp16_comb;
                            if (last_chunk) begin
                                active_vec_q <= 1'b0;
                                chunk_q      <= '0;
                                out_valid_o  <= 1'b1;
                                if (vec_cnt + 16'd1 >= cfg_num_vecs_i) begin
                                    rms_st <= ST_IDLE;
                                    done_o <= 1'b1;
                                end else begin
                                    vec_cnt <= vec_cnt + 16'd1;
                                end
                            end else begin
                                chunk_q <= chunk_q + 1'b1;
                            end
                        end
                    end
                endcase
            end
        end
    end

endmodule
