// Frame-wise DynamicQuantizeLinear for signed Q16 activations.
//
// Input frames contain 1..MAX_TOKENS tokens, each with CHANNELS samples.
// s_frame_last marks the final sample of the frame. The output is UINT8 and
// carries one Q24 scale and one zero point for the entire frame:
//
//   real_value ~= (m_data - m_zero_point) * m_scale_q24 / 2**24.
//
// The ONNX quantization interval includes zero. We calculate the exact
// integer-rational scale span_q16 / (255 * 2**16), then round it to Q24 for
// downstream use. Integer-rational rounding may differ at rare boundaries
// from ONNX's FP32 DynamicQuantizeLinear operation. No floating point is used.
//
// Two RAM banks overlap the collection of frame k+1 with output of frame k.
// A 28-step restoring divider calculates one reciprocal per frame; a small
// 40-step constant-divisor engine calculates the scale metadata in parallel.
// Per-sample quantization uses a pipelined multiply and exact correction,
// never a divider. Input and output can each accept one beat per cycle.
module dynamic_quantize_q16 #(
    parameter integer CHANNELS = 384,
    parameter integer MAX_TOKENS = 128,
    parameter integer MAX_WORDS = CHANNELS * MAX_TOKENS,
    parameter integer ADDR_W = $clog2(MAX_WORDS + 1)
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               s_valid,
    output logic               s_ready,
    input  logic signed [31:0] s_data,
    input  logic               s_frame_last,
    output logic               m_valid,
    input  logic               m_ready,
    output logic        [7:0]  m_data,
    output logic               m_token_last,
    output logic               m_frame_last,
    output logic        [32:0] m_scale_q24,
    output logic        [7:0]  m_zero_point,
    output logic               frame_error
);
    localparam logic [31:0] ZERO_SPAN_SUBSTITUTE = 32'd16711680; // 255 * 2**16

    function automatic [5:0] msb32(input logic [31:0] value);
        logic [15:0] group16;
        logic [7:0] group8;
        logic [3:0] group4;
        logic [1:0] group2;
        logic [5:0] result;
        begin
            if (|value[31:16]) begin group16 = value[31:16]; result = 6'd16; end
            else begin group16 = value[15:0]; result = 6'd0; end
            if (|group16[15:8]) begin group8 = group16[15:8]; result = result + 6'd8; end
            else group8 = group16[7:0];
            if (|group8[7:4]) begin group4 = group8[7:4]; result = result + 6'd4; end
            else group4 = group8[3:0];
            if (|group4[3:2]) begin group2 = group4[3:2]; result = result + 6'd2; end
            else group2 = group4[1:0];
            if (group2[1]) result = result + 6'd1;
            msb32 = result;
        end
    endfunction

    // KV260 UltraRAM holds the two large frame buffers, preserving BRAM for
    // the matrix engines. Neither array is reset; bank_busy prevents reads
    // of unwritten locations. Each array has one write and one read port.
    (* ram_style = "ultra" *) logic signed [31:0] x_mem0 [0:MAX_WORDS-1];
    (* ram_style = "ultra" *) logic signed [31:0] x_mem1 [0:MAX_WORDS-1];
    logic signed [31:0] x_rd0, x_rd1;
    logic write_bank, read_bank, fill_active;
    logic [1:0] bank_busy, calc_pending, bank_ready;
    logic [ADDR_W-1:0] write_index, read_index;
    logic [ADDR_W-1:0] words_bank [0:1];
    logic [8:0] write_channel, read_channel;
    logic signed [31:0] min_bank [0:1], max_bank [0:1];
    logic [31:0] span_bank [0:1];
    logic [27:0] factor_bank [0:1];
    logic [5:0] shift_bank [0:1];
    logic [32:0] scale_bank [0:1];
    logic [7:0] zero_bank [0:1];

    wire input_fire = s_valid && s_ready;
    wire output_advance = !m_valid || m_ready;
    wire output_fire = m_valid && m_ready;
    wire frame_done = output_fire && m_frame_last;
    wire read_issue = rst_n && output_advance && bank_ready[read_bank] &&
                      (read_index < words_bank[read_bank]);

    assign s_ready = rst_n && (fill_active || !bank_busy[write_bank]) &&
                     (write_index < MAX_WORDS);

    always_ff @(posedge clk) begin
        if (input_fire && !write_bank) x_mem0[write_index] <= s_data;
        if (read_issue && !read_bank) x_rd0 <= x_mem0[read_index];
    end
    always_ff @(posedge clk) begin
        if (input_fire && write_bank) x_mem1[write_index] <= s_data;
        if (read_issue && read_bank) x_rd1 <= x_mem1[read_index];
    end

    logic signed [31:0] min_before, max_before, min_after, max_after;
    logic signed [32:0] min_after_ext, max_after_ext;
    logic [32:0] input_span;
    always_comb begin
        min_before = fill_active ? min_bank[write_bank] : 32'sd0;
        max_before = fill_active ? max_bank[write_bank] : 32'sd0;
        min_after = ($signed(s_data) < min_before) ? s_data : min_before;
        max_after = ($signed(s_data) > max_before) ? s_data : max_before;
        min_after_ext = {min_after[31], min_after};
        max_after_ext = {max_after[31], max_after};
        input_span = max_after_ext - min_after_ext;
    end

    // Reciprocal factor = floor(255 * 2**(msb(span)+20) / span).
    // Normalize span to a 32-bit denominator in [2**31, 2**32).
    // The top 32 bits of the 60-bit numerator 255*2**51 equal 255*2**23
    // and are strictly below the denominator. Processing its 28 trailing
    // zero bits therefore yields the complete 28-bit quotient.
    typedef enum logic [2:0] {CALC_IDLE, CALC_RUN, CALC_ZP_MUL,
                               CALC_ZP_ESTIMATE, CALC_ZP_COMPARE,
                               CALC_ZP_REMAINDER, CALC_ZP_ROUND} calc_state_t;
    calc_state_t calc_state;
    logic calc_bank;
    logic [5:0] calc_index;
    logic [31:0] calc_denominator;
    logic [32:0] calc_remainder;
    logic [27:0] calc_quotient;
    logic [5:0] calc_shift;
    logic [39:0] scale_shift, scale_quotient;
    logic [7:0] scale_remainder;
    logic [59:0] zp_product;
    logic [40:0] zp_numerator, zp_compare;
    logic [8:0] zp_estimate;
    logic [8:0] zp_floor_reg;
    logic [40:0] zp_remainder_reg;

    wire calc_select_bank = calc_pending[read_bank] ? read_bank : ~read_bank;
    wire [5:0] calc_msb = msb32(span_bank[calc_select_bank]);
    wire [32:0] reciprocal_trial = {calc_remainder[31:0], 1'b0};
    wire reciprocal_bit = reciprocal_trial >= {1'b0, calc_denominator};
    wire [27:0] reciprocal_next = {calc_quotient[26:0], reciprocal_bit};
    wire [8:0] scale_trial = {scale_remainder, scale_shift[39]};
    wire scale_bit = scale_trial >= 9'd255;
    wire [39:0] scale_next = {scale_quotient[38:0], scale_bit};
    wire [31:0] min_magnitude = (~min_bank[calc_bank]) + 32'd1;
    wire [8:0] zp_estimate_next = zp_product >> shift_bank[calc_bank];
    wire zp_correction = (zp_compare <= zp_numerator);
    wire [8:0] zp_floor = zp_estimate + {8'd0, zp_correction};
    wire [40:0] zp_floor_product = zp_correction ? zp_compare :
                                   (zp_compare - {9'd0, span_bank[calc_bank]});
    wire [40:0] zp_remainder = zp_numerator - zp_floor_product;
    wire [41:0] zp_twice_remainder = {zp_remainder_reg, 1'b0};
    wire zp_round_up = (zp_twice_remainder > {10'd0, span_bank[calc_bank]}) ||
                       ((zp_twice_remainder == {10'd0, span_bank[calc_bank]}) && zp_floor_reg[0]);
    wire [9:0] zp_rounded = {1'b0, zp_floor_reg} + {9'd0, zp_round_up};

    integer bank;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_bank <= 1'b0;
            read_bank <= 1'b0;
            fill_active <= 1'b0;
            bank_busy <= 2'b00;
            calc_pending <= 2'b00;
            bank_ready <= 2'b00;
            write_index <= '0;
            read_index <= '0;
            write_channel <= '0;
            read_channel <= '0;
            frame_error <= 1'b0;
            calc_state <= CALC_IDLE;
            calc_bank <= 1'b0;
            calc_index <= '0;
            calc_denominator <= '0;
            calc_remainder <= '0;
            calc_quotient <= '0;
            calc_shift <= '0;
            scale_shift <= '0;
            scale_quotient <= '0;
            scale_remainder <= '0;
            zp_product <= '0;
            zp_numerator <= '0;
            zp_compare <= '0;
            zp_estimate <= '0;
            zp_floor_reg <= '0;
            zp_remainder_reg <= '0;
            for (bank = 0; bank < 2; bank = bank + 1) begin
                words_bank[bank] <= '0;
                min_bank[bank] <= '0;
                max_bank[bank] <= '0;
                span_bank[bank] <= '0;
                factor_bank[bank] <= '0;
                shift_bank[bank] <= '0;
                scale_bank[bank] <= '0;
                zero_bank[bank] <= '0;
            end
        end else begin
            if (input_fire) begin
                bank_busy[write_bank] <= 1'b1;
                fill_active <= 1'b1;
                min_bank[write_bank] <= min_after;
                max_bank[write_bank] <= max_after;
                if (write_channel == CHANNELS-1) write_channel <= '0;
                else write_channel <= write_channel + 9'd1;
                if (s_frame_last) begin
                    if (write_channel != CHANNELS-1) frame_error <= 1'b1;
                    words_bank[write_bank] <= write_index + 1'b1;
                    write_index <= '0;
                    write_channel <= '0;
                    fill_active <= 1'b0;
                    write_bank <= ~write_bank;
                    if (input_span == 0) begin
                        // All-zero frame: choose a harmless scale of 1.
                        span_bank[write_bank] <= ZERO_SPAN_SUBSTITUTE;
                        factor_bank[write_bank] <= '0;
                        shift_bank[write_bank] <= '0;
                        scale_bank[write_bank] <= 33'd16777216;
                        zero_bank[write_bank] <= '0;
                        bank_ready[write_bank] <= 1'b1;
                    end else begin
                        span_bank[write_bank] <= input_span[31:0];
                        calc_pending[write_bank] <= 1'b1;
                    end
                end else begin
                    write_index <= write_index + 1'b1;
                    if (write_index == MAX_WORDS-1) frame_error <= 1'b1;
                end
            end

            if (read_issue) begin
                read_index <= read_index + 1'b1;
                if (read_channel == CHANNELS-1) read_channel <= '0;
                else read_channel <= read_channel + 9'd1;
            end
            if (frame_done) begin
                bank_busy[read_bank] <= 1'b0;
                bank_ready[read_bank] <= 1'b0;
                read_bank <= ~read_bank;
                read_index <= '0;
                read_channel <= '0;
            end

            case (calc_state)
                CALC_IDLE: if (|calc_pending) begin
                    calc_bank <= calc_select_bank;
                    calc_pending[calc_select_bank] <= 1'b0;
                    calc_denominator <= span_bank[calc_select_bank] << (6'd31 - calc_msb);
                    calc_shift <= calc_msb + 6'd20;
                    calc_remainder <= 33'd2139095040; // 255 << 23
                    calc_quotient <= '0;
                    calc_index <= '0;
                    scale_shift <= ({span_bank[calc_select_bank], 8'b0} + 40'd127);
                    scale_quotient <= '0;
                    scale_remainder <= '0;
                    calc_state <= CALC_RUN;
                end
                CALC_RUN: begin
                    if (calc_index < 6'd28) begin
                        calc_remainder <= reciprocal_bit ?
                            reciprocal_trial - {1'b0, calc_denominator} : reciprocal_trial;
                        calc_quotient <= reciprocal_next;
                    end
                    scale_remainder <= scale_bit ? scale_trial - 9'd255 : scale_trial[7:0];
                    scale_quotient <= scale_next;
                    scale_shift <= {scale_shift[38:0], 1'b0};
                    if (calc_index == 6'd39) begin
                        factor_bank[calc_bank] <= calc_quotient;
                        shift_bank[calc_bank] <= calc_shift;
                        scale_bank[calc_bank] <= scale_next[32:0];
                        calc_state <= CALC_ZP_MUL;
                    end else calc_index <= calc_index + 6'd1;
                end
                CALC_ZP_MUL: begin
                    zp_product <= min_magnitude * factor_bank[calc_bank];
                    zp_numerator <= ({9'd0, min_magnitude} << 8) - {9'd0, min_magnitude};
                    calc_state <= CALC_ZP_ESTIMATE;
                end
                CALC_ZP_ESTIMATE: begin
                    zp_estimate <= zp_estimate_next;
                    calc_state <= CALC_ZP_COMPARE;
                end
                CALC_ZP_COMPARE: begin
                    zp_compare <= ({1'b0, zp_estimate} + 10'd1) * span_bank[calc_bank];
                    calc_state <= CALC_ZP_REMAINDER;
                end
                CALC_ZP_REMAINDER: begin
                    zp_floor_reg <= zp_floor;
                    zp_remainder_reg <= zp_remainder;
                    calc_state <= CALC_ZP_ROUND;
                end
                CALC_ZP_ROUND: begin
                    zero_bank[calc_bank] <= (zp_rounded > 10'd255) ? 8'd255 : zp_rounded[7:0];
                    bank_ready[calc_bank] <= 1'b1;
                    calc_state <= CALC_IDLE;
                end
                default: calc_state <= CALC_IDLE;
            endcase
        end
    end

    // Globally stalled streaming pipeline. Split the large 32x28 product
    // into four 16x14 DSP products and register its adder tree. The read data
    // and absolute-value stages also isolate the UltraRAM output from DSPs.
    // The floor reciprocal estimate is at most one below the exact quotient:
    // |x| <= span and the factor truncation error is less than 2^-19. One
    // product comparison then corrects it before exact ties-to-even rounding.
    typedef struct packed {
        logic token_last;
        logic frame_last;
        logic [31:0] span;
        logic [27:0] factor;
        logic [5:0] shift;
        logic [32:0] scale;
        logic [7:0] zero;
    } meta_t;

    meta_t r_meta, a_meta, b_meta, p_meta, u_meta, v_meta, e_meta, c_meta, d_meta;
    logic r_valid, a_valid, b_valid, p_valid, u_valid, v_valid;
    logic e_valid, c_valid, d_valid;
    logic r_bank;
    logic signed [31:0] a_value;
    logic [31:0] b_magnitude;
    logic a_sign, b_sign, p_sign, u_sign, v_sign, e_sign, c_sign, d_sign;
    logic [29:0] p00, p01, p10, p11;
    logic [43:0] u_low, u_high;
    logic [59:0] v_product;
    logic [40:0] p_numerator, u_numerator, v_numerator, e_numerator, c_numerator;
    logic [8:0] e_estimate, c_estimate, d_floor;
    logic [40:0] c_compare, d_remainder;

    wire signed [31:0] read_value = r_bank ? x_rd1 : x_rd0;
    wire [31:0] absolute_value = a_sign ? (~a_value + 32'd1) : a_value;
    wire [8:0] estimated_quotient = v_product >> v_meta.shift;
    wire sample_correction = c_compare <= c_numerator;
    wire [8:0] sample_floor = c_estimate + {8'd0, sample_correction};
    wire [40:0] sample_floor_product = sample_correction ? c_compare :
                                       (c_compare - {9'd0, c_meta.span});
    wire [40:0] sample_remainder = c_numerator - sample_floor_product;
    wire [41:0] sample_twice_remainder = {d_remainder, 1'b0};
    wire sample_round_up = (sample_twice_remainder > {10'd0, d_meta.span}) ||
                           ((sample_twice_remainder == {10'd0, d_meta.span}) && d_floor[0]);
    wire [9:0] sample_rounded = {1'b0, d_floor} + {9'd0, sample_round_up};
    wire signed [10:0] sample_signed = d_sign ? -$signed({1'b0, sample_rounded}) :
                                               $signed({1'b0, sample_rounded});
    wire signed [10:0] sample_with_zero = sample_signed + $signed({3'b000, d_meta.zero});

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_valid <= 1'b0;
            a_valid <= 1'b0;
            b_valid <= 1'b0;
            p_valid <= 1'b0;
            u_valid <= 1'b0;
            v_valid <= 1'b0;
            e_valid <= 1'b0;
            c_valid <= 1'b0;
            d_valid <= 1'b0;
            m_valid <= 1'b0;
        end else if (output_advance) begin
            r_valid <= read_issue;
            a_valid <= r_valid;
            b_valid <= a_valid;
            p_valid <= b_valid;
            u_valid <= p_valid;
            v_valid <= u_valid;
            e_valid <= v_valid;
            c_valid <= e_valid;
            d_valid <= c_valid;
            m_valid <= d_valid;
        end
    end

    // Data registers have no reset. Their contents are ignored unless the
    // corresponding valid bit is set, avoiding unnecessary reset routing.
    always_ff @(posedge clk) begin
        if (output_advance) begin
            if (read_issue) begin
                r_bank <= read_bank;
                r_meta.token_last <= (read_channel == CHANNELS-1);
                r_meta.frame_last <= (read_index == words_bank[read_bank]-1'b1);
                r_meta.span <= span_bank[read_bank];
                r_meta.factor <= factor_bank[read_bank];
                r_meta.shift <= shift_bank[read_bank];
                r_meta.scale <= scale_bank[read_bank];
                r_meta.zero <= zero_bank[read_bank];
            end
            if (r_valid) begin
                a_value <= read_value;
                a_sign <= read_value[31];
                a_meta <= r_meta;
            end
            if (a_valid) begin
                b_magnitude <= absolute_value;
                b_sign <= a_sign;
                b_meta <= a_meta;
            end
            if (b_valid) begin
                p00 <= b_magnitude[15:0] * b_meta.factor[13:0];
                p01 <= b_magnitude[15:0] * b_meta.factor[27:14];
                p10 <= b_magnitude[31:16] * b_meta.factor[13:0];
                p11 <= b_magnitude[31:16] * b_meta.factor[27:14];
                p_numerator <= ({9'd0, b_magnitude} << 8) - {9'd0, b_magnitude};
                p_sign <= b_sign;
                p_meta <= b_meta;
            end
            if (p_valid) begin
                u_low <= {14'd0, p00} + {p01, 14'd0};
                u_high <= {14'd0, p10} + {p11, 14'd0};
                u_numerator <= p_numerator;
                u_sign <= p_sign;
                u_meta <= p_meta;
            end
            if (u_valid) begin
                v_product <= {16'd0, u_low} + {u_high, 16'd0};
                v_numerator <= u_numerator;
                v_sign <= u_sign;
                v_meta <= u_meta;
            end
            if (v_valid) begin
                e_estimate <= estimated_quotient;
                e_numerator <= v_numerator;
                e_sign <= v_sign;
                e_meta <= v_meta;
            end
            if (e_valid) begin
                c_estimate <= e_estimate;
                c_compare <= ({1'b0, e_estimate} + 10'd1) * e_meta.span;
                c_numerator <= e_numerator;
                c_sign <= e_sign;
                c_meta <= e_meta;
            end
            if (c_valid) begin
                d_floor <= sample_floor;
                d_remainder <= sample_remainder;
                d_sign <= c_sign;
                d_meta <= c_meta;
            end
            if (d_valid) begin
                if (sample_with_zero < 0) m_data <= 8'd0;
                else if (sample_with_zero > 11'sd255) m_data <= 8'd255;
                else m_data <= sample_with_zero[7:0];
                m_token_last <= d_meta.token_last;
                m_frame_last <= d_meta.frame_last;
                m_scale_q24 <= d_meta.scale;
                m_zero_point <= d_meta.zero;
            end
        end
    end
endmodule
