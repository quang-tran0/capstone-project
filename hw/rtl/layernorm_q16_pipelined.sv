// Streaming Q16 LayerNorm for one 384-channel vector per transaction.
//
// Input, output, gamma and beta are signed Q16. The coefficient interface is
// a synchronous RAM read: coef_gamma/beta change just after a rising edge
// with coef_req=1. The table file is generated offline by
// gen_layernorm_pipelined_vectors.py. There is no floating-point hardware.
//
// A vector cannot be normalized until its final input has been accumulated.
// Two simple-dual-port x RAM banks overlap collection of vector k+1 with
// output of vector k. Within each stream, a beat can be accepted per clock.
// A few input clocks at every second vector boundary can still be blocked:
// two banks cannot sustain an indefinite vector-level II=384 when statistics
// become available only after the last input sample.
module layernorm_q16_pipelined #(
    parameter LUT_FILE = "rtl/transformer_qkv/invsqrt_q17_lut.mem"
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               s_valid,
    output logic               s_ready,
    input  logic signed [31:0] s_data,
    output logic               coef_req,
    output logic        [8:0]  coef_addr,
    input  logic signed [31:0] coef_gamma,
    input  logic signed [31:0] coef_beta,
    output logic               m_valid,
    input  logic               m_ready,
    output logic signed [31:0] m_data,
    output logic               m_last
);
    localparam logic [95:0] EPS_N2_Q48 = 96'd41435136; // 281 * 384 * 384

    // A balanced priority tree gives the exponent of a nonzero 96-bit D2.
    function automatic [6:0] msb96(input logic [95:0] value);
        logic [31:0] group32;
        logic [15:0] group16;
        logic [7:0] group8;
        logic [3:0] group4;
        logic [1:0] group2;
        logic [6:0] offset;
        begin
            if (|value[95:64]) begin group32 = value[95:64]; offset = 7'd64; end
            else if (|value[63:32]) begin group32 = value[63:32]; offset = 7'd32; end
            else begin group32 = value[31:0]; offset = 7'd0; end
            if (|group32[31:16]) begin group16 = group32[31:16]; offset = offset + 7'd16; end
            else group16 = group32[15:0];
            if (|group16[15:8]) begin group8 = group16[15:8]; offset = offset + 7'd8; end
            else group8 = group16[7:0];
            if (|group8[7:4]) begin group4 = group8[7:4]; offset = offset + 7'd4; end
            else group4 = group8[3:0];
            if (|group4[3:2]) begin group2 = group4[3:2]; offset = offset + 7'd2; end
            else group2 = group4[1:0];
            if (group2[1]) offset = offset + 7'd1;
            msb96 = offset;
        end
    endfunction

    logic [31:0] inverse_lut [0:511];
    initial $readmemh(LUT_FILE, inverse_lut);

    // Stage 1: two independent simple-dual-port RAMs. Neither array is reset,
    // which lets Vivado map each 384 x 32 store into block RAM. We never read
    // and write one bank on the same clock: bank_busy releases only after the
    // last value has entered the centered-value pipeline register.
    logic signed [31:0] x_mem0 [0:383];
    logic signed [31:0] x_mem1 [0:383];
    logic signed [31:0] x_rd0, x_rd1;
    logic wr_bank, rd_bank;
    logic [1:0] bank_busy;
    logic [8:0] wr_index, rd_index;
    logic [1:0] factor_valid;
    logic [17:0] factor_bank [0:1]; // unsigned Q17, at most 131072
    logic [5:0] shift_bank [0:1];   // 5..40 for all signed-32 inputs

    logic r_valid, r_last, r_bank;
    logic c_valid, c_last;
    logic p_valid, p_last;
    logic n_valid, n_last;
    logic a_valid, a_last;

    wire advance = !m_valid || m_ready;
    wire input_fire = s_valid && s_ready;
    wire launch = rst_n && advance && factor_valid[rd_bank];
    wire launch_last = launch && (rd_index == 9'd383);
    wire read_consumed_last = advance && r_valid && r_last;

    assign s_ready = rst_n && !bank_busy[wr_bank];
    assign coef_req = launch;
    assign coef_addr = rd_index;

    always_ff @(posedge clk) begin
        if (input_fire && !wr_bank) x_mem0[wr_index] <= s_data;
        if (launch && !rd_bank) x_rd0 <= x_mem0[rd_index];
    end
    always_ff @(posedge clk) begin
        if (input_fire && wr_bank) x_mem1[wr_index] <= s_data;
        if (launch && rd_bank) x_rd1 <= x_mem1[rd_index];
    end

    // x*x is registered before the 71-bit accumulation. This removes the
    // 32x32 multiplier from the long accumulator feedback path. sum_x and
    // sum_x2 have independent registers for the two banks.
    logic signed [40:0] sum_x_bank [0:1];
    logic [70:0] sum_x2_bank [0:1];
    logic [63:0] square_pipe;
    logic square_valid, square_first, square_last, square_bank;
    logic stats_start, stats_start_bank;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_bank <= 1'b0;
            rd_bank <= 1'b0;
            wr_index <= 9'd0;
            rd_index <= 9'd0;
            bank_busy <= 2'b00;
            sum_x_bank[0] <= '0;
            sum_x_bank[1] <= '0;
            sum_x2_bank[0] <= '0;
            sum_x2_bank[1] <= '0;
            square_valid <= 1'b0;
            square_first <= 1'b0;
            square_last <= 1'b0;
            square_bank <= 1'b0;
            square_pipe <= '0;
            stats_start <= 1'b0;
            stats_start_bank <= 1'b0;
        end else begin
            square_valid <= input_fire;
            if (input_fire) begin
                square_pipe <= $signed(s_data) * $signed(s_data);
                square_bank <= wr_bank;
                square_first <= (wr_index == 9'd0);
                square_last <= (wr_index == 9'd383);
                if (wr_index == 9'd0)
                    sum_x_bank[wr_bank] <= {{9{s_data[31]}}, s_data};
                else
                    sum_x_bank[wr_bank] <= sum_x_bank[wr_bank] + {{9{s_data[31]}}, s_data};
                if (wr_index == 9'd383) begin
                    bank_busy[wr_bank] <= 1'b1;
                    wr_bank <= ~wr_bank;
                    wr_index <= 9'd0;
                end else begin
                    wr_index <= wr_index + 9'd1;
                end
            end
            if (square_valid) begin
                if (square_first)
                    sum_x2_bank[square_bank] <= {7'd0, square_pipe};
                else
                    sum_x2_bank[square_bank] <= sum_x2_bank[square_bank] + {7'd0, square_pipe};
            end
            stats_start <= square_valid && square_last;
            if (square_valid && square_last) stats_start_bank <= square_bank;
            if (launch) begin
                if (rd_index == 9'd383) begin
                    rd_bank <= ~rd_bank;
                    rd_index <= 9'd0;
                end else begin
                    rd_index <= rd_index + 9'd1;
                end
            end
            // The centered stage has now consumed the last x and sum_x of
            // this bank. Reuse may start on the following edge, even while
            // the final output value remains in later pipeline registers.
            if (read_consumed_last) bank_busy[r_bank] <= 1'b0;
        end
    end

    // Stage 2: variance numerator. Multiplication by N=384 uses shifts.
    // One registered stage squares the signed 41-bit sum; the next subtracts.
    // These widths cover every signed 32-bit Q16 input, including INT32_MIN.
    logic stats1_valid, stats1_bank;
    logic [79:0] stats1_nsum2;
    logic signed [81:0] stats1_sum_square;
    logic var_valid, var_bank;
    logic [79:0] var_num;
    logic d2_valid, d2_bank;
    logic [95:0] d2_reg;
    wire [81:0] var_diff = {2'b00, stats1_nsum2} - $unsigned(stats1_sum_square);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stats1_valid <= 1'b0;
            var_valid <= 1'b0;
            d2_valid <= 1'b0;
        end else begin
            stats1_valid <= stats_start;
            var_valid <= stats1_valid;
            d2_valid <= var_valid;
        end
    end
    // Data registers need no reset: their matching valid bit resets to zero.
    // Keeping them in a non-reset process avoids ambiguous asynchronous-reset
    // inference and lets the synthesis tool use ordinary clock enables.
    always_ff @(posedge clk) begin
        if (rst_n) begin
            if (stats_start) begin
                stats1_bank <= stats_start_bank;
                stats1_nsum2 <= ({9'd0, sum_x2_bank[stats_start_bank]} << 8)
                              + ({9'd0, sum_x2_bank[stats_start_bank]} << 7);
                stats1_sum_square <= $signed(sum_x_bank[stats_start_bank])
                                   * $signed(sum_x_bank[stats_start_bank]);
            end
            if (stats1_valid) begin
                var_bank <= stats1_bank;
                var_num <= ({2'b00, stats1_nsum2} >= $unsigned(stats1_sum_square))
                         ? var_diff[79:0] : 80'd0;
            end
            if (var_valid) begin
                d2_bank <= var_bank;
                // D2 = (384*sum(x*x)-sum(x)^2)*2^16 + eps_Q48*384^2.
                d2_reg <= {var_num, 16'd0} + EPS_N2_Q48;
            end
        end
    end

    // Stage 3: inverse square root, four clocks from D2 to bank metadata.
    // D2 = m * 2^msb, with m in [1,2). Let e=floor(msb/2), p=msb%2:
    // sqrt(D2) = 2^e * sqrt(m * 2^p).
    // ROM[p, top 8 fraction bits] stores base=2^17/sqrt(t) (18 bits) and
    // delta=next_base-base (signed 14 bits). The lower 8 fraction bits linearly
    // interpolate. The ROM is 512 x 32 = 16 Kbits and has synchronous read.
    // The factor is used once per vector for all 384 channels. In Q16 units:
    // normalized = round_away_from_zero(centered * factor_q17 / 2^(e-7)).
    // This is the same scaling as centered*2^24/sqrt(D2) in the exact design.
    logic clz_valid, clz_bank;
    logic [95:0] clz_d2;
    logic [6:0] clz_msb;
    logic norm_valid, norm_bank;
    logic [8:0] norm_addr;
    logic [7:0] norm_frac;
    logic [5:0] norm_shift;
    logic rom_valid, rom_bank;
    logic [31:0] rom_word;
    logic [7:0] rom_frac;
    logic [5:0] rom_shift;

    wire [95:0] shifted_d2 = clz_d2 >> (clz_msb - 7'd16);
    wire [16:0] mantissa_q16 = shifted_d2[16:0];
    wire signed [13:0] rom_delta = rom_word[13:0];
    wire [17:0] rom_base = rom_word[31:14];
    wire signed [22:0] interpolation_product = rom_delta * $signed({1'b0, rom_frac});
    wire [22:0] interpolation_magnitude = interpolation_product[22]
                                          ? -interpolation_product : interpolation_product;
    wire [14:0] interpolation_rounded = (interpolation_magnitude + 23'd128) >> 8;
    wire signed [19:0] interpolated_factor = $signed({2'b00, rom_base})
                        + (interpolation_product[22] ? -$signed({5'd0, interpolation_rounded})
                                                       :  $signed({5'd0, interpolation_rounded}));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            clz_valid <= 1'b0;
            norm_valid <= 1'b0;
            rom_valid <= 1'b0;
            factor_valid <= 2'b00;
        end else begin
            clz_valid <= d2_valid;
            norm_valid <= clz_valid;
            rom_valid <= norm_valid;
            if (rom_valid) factor_valid[rom_bank] <= 1'b1;
            if (launch_last) factor_valid[rd_bank] <= 1'b0;
        end
    end
    always_ff @(posedge clk) begin
        if (rst_n) begin
            if (d2_valid) begin
                clz_bank <= d2_bank;
                clz_d2 <= d2_reg;
                clz_msb <= msb96(d2_reg);
            end
            if (clz_valid) begin
                norm_bank <= clz_bank;
                norm_addr <= {clz_msb[0], mantissa_q16[15:8]};
                norm_frac <= mantissa_q16[7:0];
                norm_shift <= {1'b0, clz_msb[6:1]} - 6'd7;
            end
            if (norm_valid) begin
                rom_bank <= norm_bank;
                rom_word <= inverse_lut[norm_addr];
                rom_frac <= norm_frac;
                rom_shift <= norm_shift;
            end
            if (rom_valid) begin
                factor_bank[rom_bank] <= interpolated_factor[17:0];
                shift_bank[rom_bank] <= rom_shift;
            end
        end
    end

    // Stage 4: an elastic seven-register output pipeline. advance freezes every
    // data/valid register and both RAM read ports during backpressure. The
    // 42x19 product, signed-magnitude preparation, normalization, and 22x19
    // gamma product each have their own clock stage. The extra magnitude
    // register breaks the old product -> abs -> 60-bit add -> barrel shift ->
    // sign restore -> gamma DSP path. It adds one clock of output latency
    // without changing the one-channel-per-clock steady-state throughput.
    // The 22-bit normalized value is sufficient for N=384: its magnitude is
    // bounded by approximately sqrt(383)*2^16 for any input vector.
    logic signed [41:0] centered_c;
    logic [17:0] factor_c;
    logic [5:0] shift_c;
    logic signed [18:0] gamma_c, gamma_p, gamma_prep, gamma_n;
    logic signed [31:0] beta_c, beta_p, beta_prep, beta_n, beta_a;
    logic signed [59:0] norm_product_p;
    logic [59:0] magnitude_prep;
    logic negative_prep;
    logic [5:0] shift_p, shift_prep;
    logic prep_valid, prep_last;
    logic signed [21:0] normalized_n;
    logic signed [40:0] affine_product_a;

    wire signed [31:0] x_selected = r_bank ? x_rd1 : x_rd0;
    wire signed [41:0] x_extended = {{10{x_selected[31]}}, x_selected};
    wire signed [41:0] centered_now = (x_extended <<< 8) + (x_extended <<< 7)
                            - {{1{sum_x_bank[r_bank][40]}}, sum_x_bank[r_bank]};
    wire [59:0] product_magnitude = norm_product_p[59]
                                   ? -norm_product_p : norm_product_p;
    // For magnitude = q*2^s + remainder, adding 2^(s-1) before >>s is
    // exactly q + remainder[s-1]. Valid shifts are 5..40; the old 60-bit
    // addition cannot overflow for any signed-32 input and Q17 factor.
    // Keeping the full shifted value preserves the original truncation point;
    // synthesis removes upper increment bits unused by normalized_now.
    wire [59:0] shifted_magnitude = magnitude_prep >> shift_prep;
    wire round_bit = magnitude_prep[shift_prep - 6'd1];
    wire [59:0] rounded_full = shifted_magnitude + {59'd0, round_bit};
    wire signed [21:0] normalized_now = negative_prep
                                         ? -$signed(rounded_full[21:0])
                                         :  $signed(rounded_full[21:0]);
    wire [41:0] affine_magnitude = affine_product_a[40]
                    ? -$signed({affine_product_a[40], affine_product_a})
                    :  $signed({affine_product_a[40], affine_product_a});
    wire [41:0] affine_rounded = (affine_magnitude + 42'd32768) >> 16;
    wire signed [32:0] affine_scaled = affine_product_a[40]
                                      ? -$signed({1'b0, affine_rounded[31:0]})
                                      :  $signed({1'b0, affine_rounded[31:0]});
    wire signed [32:0] affine_result = affine_scaled + {{1{beta_a[31]}}, beta_a};

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_valid <= 1'b0;
            c_valid <= 1'b0;
            p_valid <= 1'b0;
            prep_valid <= 1'b0;
            n_valid <= 1'b0;
            a_valid <= 1'b0;
            m_valid <= 1'b0;
        end else if (advance) begin
            r_valid <= launch;
            c_valid <= r_valid;
            p_valid <= c_valid;
            prep_valid <= p_valid;
            n_valid <= prep_valid;
            a_valid <= n_valid;
            m_valid <= a_valid;
        end
    end
    always_ff @(posedge clk) begin
        if (rst_n && advance) begin
            if (launch) begin
                r_bank <= rd_bank;
                r_last <= (rd_index == 9'd383);
            end
            if (r_valid) begin
                centered_c <= centered_now;
                factor_c <= factor_bank[r_bank];
                shift_c <= shift_bank[r_bank];
                gamma_c <= coef_gamma[18:0]; // model gamma fits signed 19 bits
                beta_c <= coef_beta;
                c_last <= r_last;
            end
            if (c_valid) begin
                norm_product_p <= centered_c * $signed({1'b0, factor_c});
                shift_p <= shift_c;
                gamma_p <= gamma_c;
                beta_p <= beta_c;
                p_last <= c_last;
            end
            if (p_valid) begin
                magnitude_prep <= product_magnitude;
                negative_prep <= norm_product_p[59];
                shift_prep <= shift_p;
                gamma_prep <= gamma_p;
                beta_prep <= beta_p;
                prep_last <= p_last;
            end
            if (prep_valid) begin
                normalized_n <= normalized_now;
                gamma_n <= gamma_prep;
                beta_n <= beta_prep;
                n_last <= prep_last;
            end
            if (n_valid) begin
                affine_product_a <= normalized_n * gamma_n;
                beta_a <= beta_n;
                a_last <= n_last;
            end
            if (a_valid) begin
                m_data <= affine_result[31:0];
                m_last <= a_last;
            end
        end
    end
endmodule
