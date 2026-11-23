// =============================================================================
// Module: snax_stream_read_fp16
// Description: TCDM read stream that assembles 4 x 72b words into one 256b
//              FP16 vector.  Counterpart of snax_streamer_write_out FP16 mode.
//
// Layout (must match write_out FP16 mode):
//   word w of vector i = cfg_base_addr_i + (i*4 + w) * cfg_line_stride_i
//   fp16[64*w +: 64]   = word[63:0]      (the 8-bit high field is padding)
//
// FSM: IDLE -> REQ -> RSP -> (REQ | EMIT) -> (REQ | IDLE)
// =============================================================================
`timescale 1ns/1ps

module snax_stream_read_fp16 #(
    parameter int unsigned DATA_WIDTH = 72,
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned STRB_WIDTH = 9
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                    csr_start_i,
    output logic                    busy_o,
    output logic                    done_irq_o,

    input  logic [ADDR_WIDTH-1:0]   cfg_base_addr_i,
    input  logic [31:0]             cfg_num_vectors_i,
    input  logic [31:0]             cfg_line_stride_i,
    input  logic [31:0]             cfg_vec_stride_i,
    input  logic [31:0]             cfg_num_batches_i,
    input  logic [31:0]             cfg_batch_stride_i,
    input  logic                    cfg_wrap_en_i,
    input  logic                    cfg_wrap_bank_inc_i,
    input  logic [15:0]             cfg_wrap_words_m1_i,

    output logic                    m_tcdm_req_valid_o,
    output logic                    m_tcdm_req_write_o,
    output logic [ADDR_WIDTH-1:0]   m_tcdm_req_addr_o,
    output logic [DATA_WIDTH-1:0]   m_tcdm_req_wdata_o,
    output logic [STRB_WIDTH-1:0]   m_tcdm_req_strb_o,
    input  logic                    m_tcdm_req_ready_i,

    input  logic [DATA_WIDTH-1:0]   m_tcdm_rsp_rdata_i,
    input  logic                    m_tcdm_rsp_valid_i,

    output logic [255:0]            fp16_data_o,
    output logic                    fp16_valid_o,
    input  logic                    fp16_ready_i
);

    typedef enum logic [1:0] { IDLE, REQ, RSP, EMIT } state_t;
    state_t state_q, state_d;

    logic [31:0]           vec_cnt_q,       vec_cnt_d;
    logic [31:0]           batch_cnt_q,     batch_cnt_d;
    logic [1:0]            word_cnt_q,      word_cnt_d;
    logic [ADDR_WIDTH-1:0] addr_q,          addr_d;
    logic [ADDR_WIDTH-1:0] vec_base_q,      vec_base_d;
    logic [ADDR_WIDTH-1:0] batch_base_q,    batch_base_d;
    logic [15:0]           bank_word_cnt_q, bank_word_cnt_d;
    logic [3:0][63:0]      word_buf_q;
    logic                  start_q;

    // Latched descriptor registers on start_pulse
    logic [ADDR_WIDTH-1:0] base_q;
    logic [31:0]           nvec_q;
    logic [31:0]           lstride_q;
    logic [31:0]           vstride_q;
    logic [31:0]           nbatch_q;
    logic [31:0]           bstride_q;
    logic                  wrap_en_q;
    logic                  wrap_bank_inc_q;
    logic [15:0]           wrap_words_m1_q;

    wire start_pulse   = csr_start_i && !start_q;
    wire [31:0] nbatch_eff_in = (cfg_num_batches_i == 32'd0) ? 32'd1 : cfg_num_batches_i;
    wire bank_wrap_hit = wrap_en_q && (bank_word_cnt_q == wrap_words_m1_q);
    wire [ADDR_WIDTH-1:0] next_tcdm_addr = bank_wrap_hit
        ? {base_q[ADDR_WIDTH-1:9],
           addr_q[8:4] + (wrap_bank_inc_q ? 5'd1 : 5'd0),
           base_q[3:0]}
        : (addr_q + lstride_q);

    always_comb begin
        state_d         = state_q;
        vec_cnt_d       = vec_cnt_q;
        batch_cnt_d     = batch_cnt_q;
        word_cnt_d      = word_cnt_q;
        addr_d          = addr_q;
        vec_base_d      = vec_base_q;
        batch_base_d    = batch_base_q;
        bank_word_cnt_d = bank_word_cnt_q;
        done_irq_o      = 1'b0;

        m_tcdm_req_valid_o = 1'b0;
        m_tcdm_req_write_o = 1'b0;
        m_tcdm_req_addr_o  = addr_q;
        m_tcdm_req_wdata_o = '0;
        m_tcdm_req_strb_o  = '0;

        fp16_valid_o = 1'b0;

        case (state_q)
            IDLE: begin
                if (start_pulse) begin
                    if (cfg_num_vectors_i == 32'd0) begin
                        done_irq_o = 1'b1;
                    end else begin
                        vec_cnt_d       = '0;
                        batch_cnt_d     = '0;
                        word_cnt_d      = '0;
                        addr_d          = cfg_base_addr_i;
                        vec_base_d      = cfg_base_addr_i;
                        batch_base_d    = cfg_base_addr_i;
                        bank_word_cnt_d = 16'd0;
                        state_d         = REQ;
                    end
                end
            end

            REQ: begin
                m_tcdm_req_valid_o = 1'b1;
                if (m_tcdm_req_ready_i)
                    state_d = RSP;
            end

            RSP: begin
                if (m_tcdm_rsp_valid_i) begin
                    if (word_cnt_q < 2'd3) begin
                        word_cnt_d      = word_cnt_q + 1'b1;
                        addr_d          = next_tcdm_addr;
                        bank_word_cnt_d = bank_wrap_hit ? 16'd0 : (bank_word_cnt_q + 16'd1);
                        state_d         = REQ;
                    end else begin
                        state_d = EMIT;
                    end
                end
            end

            EMIT: begin
                fp16_valid_o = 1'b1;
                if (fp16_ready_i) begin
                    word_cnt_d = '0;
                    if (vec_cnt_q < nvec_q - 1'b1) begin
                        vec_cnt_d = vec_cnt_q + 1'b1;
                        if (vstride_q != 32'd0) begin
                            vec_base_d      = vec_base_q + vstride_q;
                            addr_d          = vec_base_q + vstride_q;
                            bank_word_cnt_d = 16'd0;
                        end else begin
                            addr_d          = next_tcdm_addr;
                            bank_word_cnt_d = bank_wrap_hit ? 16'd0 : (bank_word_cnt_q + 16'd1);
                        end
                        state_d = REQ;
                    end else begin
                        vec_cnt_d = '0;
                        if (batch_cnt_q < nbatch_q - 1'b1) begin
                            batch_cnt_d     = batch_cnt_q + 1'b1;
                            batch_base_d    = batch_base_q + bstride_q;
                            vec_base_d      = batch_base_q + bstride_q;
                            addr_d          = batch_base_q + bstride_q;
                            bank_word_cnt_d = 16'd0;
                            state_d         = REQ;
                        end else begin
                            done_irq_o = 1'b1;
                            state_d    = IDLE;
                        end
                    end
                end
            end

            default: state_d = IDLE;
        endcase
    end

    // Word capture is a small register array; write the arriving word on the
    // response cycle so EMIT can read all four lanes combinationally.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q         <= IDLE;
            vec_cnt_q       <= '0;
            batch_cnt_q     <= '0;
            word_cnt_q      <= '0;
            addr_q          <= '0;
            vec_base_q      <= '0;
            batch_base_q    <= '0;
            bank_word_cnt_q <= '0;
            start_q         <= 1'b0;
            word_buf_q      <= '0;
            base_q          <= '0;
            nvec_q          <= '0;
            lstride_q       <= '0;
            vstride_q       <= '0;
            nbatch_q        <= 32'd1;
            bstride_q       <= '0;
            wrap_en_q       <= 1'b0;
            wrap_bank_inc_q <= 1'b0;
            wrap_words_m1_q <= '0;
        end else begin
            state_q         <= state_d;
            vec_cnt_q       <= vec_cnt_d;
            batch_cnt_q     <= batch_cnt_d;
            word_cnt_q      <= word_cnt_d;
            addr_q          <= addr_d;
            vec_base_q      <= vec_base_d;
            batch_base_q    <= batch_base_d;
            bank_word_cnt_q <= bank_word_cnt_d;
            start_q         <= csr_start_i;

            if (state_q == IDLE && start_pulse && cfg_num_vectors_i != 32'd0) begin
                base_q          <= cfg_base_addr_i;
                nvec_q          <= cfg_num_vectors_i;
                lstride_q       <= cfg_line_stride_i;
                vstride_q       <= cfg_vec_stride_i;
                nbatch_q        <= nbatch_eff_in;
                bstride_q       <= cfg_batch_stride_i;
                wrap_en_q       <= cfg_wrap_en_i;
                wrap_bank_inc_q <= cfg_wrap_bank_inc_i;
                wrap_words_m1_q <= cfg_wrap_words_m1_i;
            end

            if (state_q == RSP && m_tcdm_rsp_valid_i)
                word_buf_q[word_cnt_q] <= m_tcdm_rsp_rdata_i[63:0];
        end
    end

    assign fp16_data_o = {word_buf_q[3], word_buf_q[2],
                          word_buf_q[1], word_buf_q[0]};

    assign busy_o = (state_q != IDLE);

endmodule
