// Embedding LayerNorm -> ONNX-style DynamicQuantizeLinear -> shared Q/K/V TPU.
//
// Software supplies the 384-dimensional token+position+type embedding sum
// in signed Q16.  A frame has 1..MAX_TOKENS tokens and s_frame_last on its
// last channel.  The LayerNorm output is also saved as a Q16 residual for a
// later attention Add.  Configure coefficients and Q/K/V memories before
// starting a frame.  All arithmetic in this top and its children is integer.
module embedding_norm_qkv #(
    parameter integer MAX_TOKENS = 128,
    parameter LUT_FILE = "rtl/transformer_qkv/invsqrt_q17_lut.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               s_valid,
    output logic               s_ready,
    input  logic signed [31:0] s_data_q16,
    input  logic               s_frame_last,
    input  logic               cfg_ln_we,
    input  logic        [8:0]  cfg_ln_addr,
    input  logic signed [31:0] cfg_ln_gamma_q16,
    input  logic signed [31:0] cfg_ln_beta_q16,
    input  logic               cfg_weight_we,
    input  logic        [14:0] cfg_weight_addr,
    input  logic        [31:0] cfg_weight_data,
    input  logic        [1:0]  cfg_weight_lane,
    input  logic               cfg_bias_we,
    input  logic        [10:0] cfg_bias_addr,
    input  logic signed [31:0] cfg_bias_q16,
    input  logic               cfg_scale_we,
    input  logic        [1:0]  cfg_scale_branch,
    input  logic        [31:0] cfg_scale_q32,
    input  logic               residual_req,
    input  logic        [15:0] residual_addr,
    output logic signed [31:0] residual_data_q16,
    output logic               m_valid,
    input  logic               m_ready,
    output logic signed [31:0] m_data_q16,
    output logic        [1:0]  m_branch,
    output logic        [7:0]  m_token_index,
    output logic        [8:0]  m_channel_index,
    output logic               m_token_last,
    output logic               m_frame_last,
    output logic               frame_error,
    // Handshake-qualified taps for simulation and system bring-up.
    output logic               dbg_norm_fire,
    output logic signed [31:0] dbg_norm_data_q16,
    output logic               dbg_quant_fire,
    output logic        [7:0]  dbg_quant_data,
    output logic        [32:0] dbg_quant_scale_q24,
    output logic        [7:0]  dbg_quant_zero_point
);
    logic [8:0] ln_coef_addr;
    logic ln_coef_req;
    logic signed [31:0] ln_coef_gamma, ln_coef_beta;
    (* ram_style = "block" *) logic signed [31:0] gamma_mem [0:383];
    (* ram_style = "block" *) logic signed [31:0] beta_mem [0:383];
    always_ff @(posedge clk) begin
        if (cfg_ln_we) begin
            gamma_mem[cfg_ln_addr] <= cfg_ln_gamma_q16;
            beta_mem[cfg_ln_addr] <= cfg_ln_beta_q16;
        end
        if (ln_coef_req) begin
            ln_coef_gamma <= gamma_mem[ln_coef_addr];
            ln_coef_beta <= beta_mem[ln_coef_addr];
        end
    end

    logic source_closed, input_error;
    logic [7:0] src_token, frame_tokens, ln_out_token;
    logic [8:0] src_channel, ln_out_channel;
    logic ln_s_ready, ln_m_valid, ln_m_ready, ln_m_last;
    logic signed [31:0] ln_m_data;
    logic input_fire, norm_fire;
    logic source_input_enable;
    assign source_input_enable = rst_n && !source_closed &&
                                 (src_token < MAX_TOKENS);
    assign s_ready = source_input_enable && ln_s_ready;
    assign input_fire = s_valid && s_ready;
    assign norm_fire = ln_m_valid && ln_m_ready;
    assign dbg_norm_fire = norm_fire;
    assign dbg_norm_data_q16 = ln_m_data;

    layernorm_q16_pipelined #(.LUT_FILE(LUT_FILE)) layernorm (
        .clk(clk), .rst_n(rst_n),
        .s_valid(s_valid && source_input_enable), .s_ready(ln_s_ready),
        .s_data(s_data_q16),
        .coef_req(ln_coef_req), .coef_addr(ln_coef_addr),
        .coef_gamma(ln_coef_gamma), .coef_beta(ln_coef_beta),
        .m_valid(ln_m_valid), .m_ready(ln_m_ready),
        .m_data(ln_m_data), .m_last(ln_m_last)
    );

    // Preserve the normalized vector: attention's output projection later
    // adds it before its next LayerNorm.  One write and one external read
    // port map to a simple dual-port RAM (read latency one rising edge).
    (* ram_style = "ultra" *) logic signed [31:0] residual_mem
        [0:MAX_TOKENS*384-1];
    logic [15:0] residual_write_addr;
    assign residual_write_addr = ln_out_token * 384 + ln_out_channel;
    always_ff @(posedge clk) begin
        if (norm_fire) residual_mem[residual_write_addr] <= ln_m_data;
        if (residual_req) residual_data_q16 <= residual_mem[residual_addr];
    end

    logic quant_s_ready, quant_m_valid, quant_m_ready;
    logic [7:0] quant_m_data, quant_m_zero_point;
    logic [32:0] quant_m_scale_q24;
    logic quant_m_token_last, quant_m_frame_last, quant_error;
    logic quant_frame_last;
    assign ln_m_ready = quant_s_ready;
    assign quant_frame_last = ln_m_last &&
                              (ln_out_token == frame_tokens - 1'b1);
    dynamic_quantize_q16 #(.CHANNELS(384), .MAX_TOKENS(MAX_TOKENS))
    quantizer (
        .clk(clk), .rst_n(rst_n),
        .s_valid(ln_m_valid), .s_ready(quant_s_ready),
        .s_data(ln_m_data), .s_frame_last(quant_frame_last),
        .m_valid(quant_m_valid), .m_ready(quant_m_ready),
        .m_data(quant_m_data), .m_token_last(quant_m_token_last),
        .m_frame_last(quant_m_frame_last),
        .m_scale_q24(quant_m_scale_q24),
        .m_zero_point(quant_m_zero_point), .frame_error(quant_error)
    );
    assign dbg_quant_fire = quant_m_valid && quant_m_ready;
    assign dbg_quant_data = quant_m_data;
    assign dbg_quant_scale_q24 = quant_m_scale_q24;
    assign dbg_quant_zero_point = quant_m_zero_point;

    logic qkv_error;
    qkv_projection_q16 #(.CHANNELS(384), .MAX_TOKENS(MAX_TOKENS))
    qkv (
        .clk(clk), .rst_n(rst_n),
        .s_valid(quant_m_valid), .s_ready(quant_m_ready),
        .s_data(quant_m_data),
        .s_token_last(quant_m_token_last),
        .s_frame_last(quant_m_frame_last),
        .s_scale_q24(quant_m_scale_q24),
        .s_zero_point(quant_m_zero_point),
        .cfg_weight_we(cfg_weight_we),
        .cfg_weight_addr(cfg_weight_addr),
        .cfg_weight_data(cfg_weight_data),
        .cfg_weight_lane(cfg_weight_lane),
        .cfg_bias_we(cfg_bias_we), .cfg_bias_addr(cfg_bias_addr),
        .cfg_bias_q16(cfg_bias_q16), .cfg_scale_we(cfg_scale_we),
        .cfg_scale_branch(cfg_scale_branch),
        .cfg_scale_q32(cfg_scale_q32),
        .m_valid(m_valid), .m_ready(m_ready),
        .m_data_q16(m_data_q16), .m_branch(m_branch),
        .m_token_index(m_token_index), .m_channel_index(m_channel_index),
        .m_token_last(m_token_last), .m_frame_last(m_frame_last),
        .frame_error(qkv_error)
    );
    assign frame_error = input_error || quant_error || qkv_error;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            source_closed <= 1'b0;
            input_error <= 1'b0;
            src_token <= '0;
            src_channel <= '0;
            frame_tokens <= '0;
            ln_out_token <= '0;
            ln_out_channel <= '0;
        end else begin
            if (input_fire) begin
                if (s_frame_last && src_channel != 9'd383)
                    input_error <= 1'b1;
                if (src_channel == 9'd383) begin
                    src_channel <= '0;
                    if (s_frame_last) begin
                        frame_tokens <= src_token + 1'b1;
                        source_closed <= 1'b1;
                    end else if (src_token == MAX_TOKENS-1) begin
                        input_error <= 1'b1;
                        src_token <= src_token + 1'b1;
                    end else
                        src_token <= src_token + 1'b1;
                end else
                    src_channel <= src_channel + 1'b1;
            end
            if (norm_fire) begin
                if (ln_m_last != (ln_out_channel == 9'd383))
                    input_error <= 1'b1;
                if (ln_m_last) begin
                    ln_out_channel <= '0;
                    ln_out_token <= ln_out_token + 1'b1;
                end else
                    ln_out_channel <= ln_out_channel + 1'b1;
            end
            if (m_valid && m_ready && m_frame_last) begin
                source_closed <= 1'b0;
                src_token <= '0;
                src_channel <= '0;
                frame_tokens <= '0;
                ln_out_token <= '0;
                ln_out_channel <= '0;
            end
        end
    end
endmodule
