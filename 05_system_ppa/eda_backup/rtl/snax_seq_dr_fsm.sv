// =============================================================================
// Module: snax_seq_dr_fsm.sv
// Description: Sub-FSM 2 of snax_llm_sequencer -- Shared Port-B Drain,
//              OUT_PIPE_CREDITS Counter, and WAR Hazard Guard.
// =============================================================================

`timescale 1ns/1ps

module snax_seq_dr_fsm #(
    parameter int unsigned NUM_LANES        = 16,
    parameter int unsigned OUT_PIPE_CREDITS = 4
)(
    input  logic                 clk,
    input  logic                 rst_n,

    input  logic                 start_i,
    input  logic                 is_gemm_i,
    input  logic                 is_attn_i,
    input  logic                 cfg_gemm_decode_i,
    input  logic [3:0]           cfg_attn_m_q_i,

    input  logic                 core_fp16_valid_i,
    input  logic                 core_fp16_ready_i,
    input  logic                 core_out_fifo_ready_i,

    // From Compute FSM
    input  logic                 drain_start_pulse_i,
    input  logic [3:0]           cnt_gbeat_i,
    input  logic                 sm_drain_req_i,
    input  logic [3:0]           sm_drain_idx_i,
    input  logic                 sm_row_start_i,
    input  logic [NUM_LANES-1:0] auto_causal_mask_i,

    // Status to Compute / DMA / Top
    output logic                 dr_busy_o,
    output logic                 gemm_beat_ok_o,
    output logic [3:0]           inflight_out_cnt_o,
    output logic [1:0]           dr_state_o,
    output logic [4:0]           dr_slot_o,

    // Core Port-B Drain & Softmax Mask Interface
    output logic                 tile_drain_req_o,
    output logic [3:0]           tile_drain_idx_o,
    output logic                 tile_clear_on_drain_o,
    output logic                 cfg_row_start_o,
    output logic [NUM_LANES-1:0] cfg_key_mask_o
);

    typedef enum logic [1:0] {
        DR_IDLE    = 2'd0,
        DR_WAIT_D1 = 2'd1,
        DR_RUN     = 2'd2
    } dr_state_t;

    dr_state_t  dr_state_q, dr_state_d;
    logic [4:0] dr_slot_q,  dr_slot_d;
    logic [3:0] inflight_out_cnt_q;

    wire [4:0] dr_last_slot    = is_attn_i ? {1'b0, cfg_attn_m_q_i - 4'd1} :
                                             (cfg_gemm_decode_i ? 5'd0 : 5'd15);
    wire       dr_needs_wait   = is_gemm_i && cfg_gemm_decode_i;
    wire       out_retire_fire = (is_gemm_i || is_attn_i) && core_fp16_valid_i && core_fp16_ready_i &&  (inflight_out_cnt_q != 4'd0);
    wire       dr_can_issue    = (inflight_out_cnt_q < 4'(OUT_PIPE_CREDITS)) && core_out_fifo_ready_i;

    logic       dr_req, dr_clear;
    logic [3:0] dr_idx;

    assign dr_state_o         = dr_state_q;
    assign dr_slot_o          = dr_slot_q;
    assign inflight_out_cnt_o = inflight_out_cnt_q;

    // -------------------------------------------------------------------------
    // 2.1 Drain State Register
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            dr_state_q <= DR_IDLE;
        else if (start_i)
            dr_state_q <= DR_IDLE;
        else
            dr_state_q <= dr_state_d;
    end

    // -------------------------------------------------------------------------
    // 2.2 Drain Next-State Logic
    // -------------------------------------------------------------------------
    always_comb begin
        dr_state_d = dr_state_q;

        case (dr_state_q)
            DR_IDLE: begin
                if (drain_start_pulse_i)
                    dr_state_d = dr_needs_wait ? DR_WAIT_D1 : DR_RUN;
            end

            DR_WAIT_D1: begin
                dr_state_d = DR_RUN;
            end

            DR_RUN: begin
                if (dr_can_issue && (dr_slot_q == dr_last_slot))
                    dr_state_d = DR_IDLE;
            end

            default: dr_state_d = DR_IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // 2.3 Drain Combinational Outputs & WAR Hazard Guard
    // -------------------------------------------------------------------------
    always_comb begin
        dr_slot_d      = dr_slot_q;
        dr_req         = 1'b0;
        dr_idx         = 4'd0;
        dr_clear       = 1'b0;
        dr_busy_o      = (dr_state_q != DR_IDLE);
        gemm_beat_ok_o = (dr_state_q == DR_IDLE) ||
                         ((dr_state_q == DR_RUN) && ({1'b0, cnt_gbeat_i} < dr_slot_q));

        case (dr_state_q)
            DR_IDLE: begin
                if (drain_start_pulse_i)
                    dr_slot_d = 5'd0;
            end

            DR_WAIT_D1: begin
                dr_slot_d = 5'd0;
            end

            DR_RUN: begin
                if (dr_can_issue) begin
                    dr_req    = 1'b1;
                    dr_idx    = dr_slot_q[3:0];
                    dr_clear  = 1'b1;
                    dr_slot_d = (dr_slot_q == dr_last_slot) ? 5'd16 : (dr_slot_q + 5'd1);
                end
            end

            default: dr_slot_d = 5'd0;
        endcase

        // Port-B drain mux (Attention SM drain and Background O/GEMM drain are mutually exclusive)
        if (sm_drain_req_i) begin
            tile_drain_req_o      = 1'b1;
            tile_drain_idx_o      = sm_drain_idx_i;
            tile_clear_on_drain_o = 1'b1;
            cfg_row_start_o       = sm_row_start_i;
        end else begin
            tile_drain_req_o      = dr_req;
            tile_drain_idx_o      = dr_idx;
            tile_clear_on_drain_o = dr_clear;
            cfg_row_start_o       = 1'b0;
        end
        cfg_key_mask_o = auto_causal_mask_i;
    end

    // -------------------------------------------------------------------------
    // 2.4 Registered Drain Slot & Output Pipeline Credit Counter
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dr_slot_q          <= 5'd0;
            inflight_out_cnt_q <= 4'd0;
        end else if (start_i) begin
            dr_slot_q          <= 5'd0;
            inflight_out_cnt_q <= 4'd0;
        end else begin
            dr_slot_q          <= dr_slot_d;
            inflight_out_cnt_q <= inflight_out_cnt_q + (dr_req ? 4'd1 : 4'd0) - (out_retire_fire ? 4'd1 : 4'd0);
        end
    end

endmodule
