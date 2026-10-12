// Q/K/V MatMulInteger + fixed-point dequantization for one UINT8 frame.
//
// The 16x16 array is output stationary.  One K sample enters each clock;
// after a tile drains, its 256 INT32 sums are serialized through a shared
// two-multiply rescale pipe.  There is no divider or floating-point RTL.
// All three ONNX branches share the same array and the same input frame.
// Configure weights, biases, and scales while no frame is being processed.
// Weight address = branch*(CHANNELS/16)*CHANNELS + col_tile*CHANNELS + k.
// Weight word byte lane c is signed W[k, col_tile*16+c].
module qkv_projection_q16 #(
    parameter integer CHANNELS = 384,
    parameter integer MAX_TOKENS = 128
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               s_valid,
    output logic               s_ready,
    input  logic        [7:0]  s_data,
    input  logic               s_token_last,
    input  logic               s_frame_last,
    input  logic        [32:0] s_scale_q24,
    input  logic        [7:0]  s_zero_point,
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
    output logic               m_valid,
    input  logic               m_ready,
    output logic signed [31:0] m_data_q16,
    output logic        [1:0]  m_branch,       // 0=Q, 1=K, 2=V
    output logic        [7:0]  m_token_index,
    output logic        [8:0]  m_channel_index,
    output logic               m_token_last,
    output logic               m_frame_last,
    output logic               frame_error
);
    localparam integer COL_TILES = CHANNELS / 16;
    localparam integer ROW_TILES = (MAX_TOKENS + 15) / 16;
    localparam integer BANK_DEPTH = ROW_TILES * CHANNELS;
    localparam integer WEIGHT_DEPTH = 3 * COL_TILES * CHANNELS;
    localparam integer BIAS_DEPTH = 3 * CHANNELS;

    typedef enum logic [2:0] {
        FILL, TILE_CLEAR, TILE_READ, TILE_DRAIN, TILE_OUT, TILE_NEXT
    } state_t;
    state_t state;

    // Stage 0: sixteen byte-wide activation RAMs permit all 16 token rows
    // to be read at the same k.  Input writes one bank each clock.  The
    // frame remains resident until Q, K, and V have all read it.
    logic [7:0] a_read [0:15];
    logic [7:0] a_write_lane;
    logic [15:0] a_write_addr, a_read_addr;
    logic input_fire, read_issue;
    logic [7:0] fill_token, tokens_in_frame;
    logic [8:0] fill_channel;
    logic [32:0] frame_scale_q24;
    logic [7:0] frame_zero_point;
    logic [1:0] branch;
    logic [3:0] row_tile;
    logic [5:0] col_tile;
    logic [8:0] k_issue;
    logic [5:0] drain_count;
    logic [8:0] issue_index;
    logic [7:0] rows_remaining;
    logic [8:0] tile_words;

    assign s_ready = rst_n && (state == FILL) && (fill_token < MAX_TOKENS);
    assign input_fire = s_valid && s_ready;
    assign a_write_lane = fill_token[3:0];
    assign a_write_addr = (fill_token >> 4) * CHANNELS + fill_channel;
    assign a_read_addr = row_tile * CHANNELS + k_issue;
    assign read_issue = state == TILE_READ;
    assign rows_remaining = tokens_in_frame - (row_tile * 16);
    assign tile_words = (rows_remaining >= 16) ? 9'd256 :
                        {rows_remaining[4:0], 4'b0000};

    for (genvar r = 0; r < 16; r = r + 1) begin : g_activation_ram
        // Each bank is only 3072 x 8 for MAX_TOKENS=128.  BRAM avoids
        // wasting an entire URAM primitive on each narrow bank.
        (* ram_style = "block" *) logic [7:0] mem [0:BANK_DEPTH-1];
        always_ff @(posedge clk) begin
            if (input_fire && (a_write_lane == r))
                mem[a_write_addr] <= s_data;
            if (read_issue)
                a_read[r] <= mem[a_read_addr];
        end
    end

    // Wide, K-major weight words make all 16 column weights available in
    // one read.  A one-cycle read pairs with the 16 activation RAM outputs.
    (* ram_style = "ultra" *) logic [127:0] weight_mem [0:WEIGHT_DEPTH-1];
    logic [127:0] weight_read;
    logic [14:0] weight_read_addr;
    assign weight_read_addr = branch * (COL_TILES * CHANNELS) +
                              col_tile * CHANNELS + k_issue;
    always_ff @(posedge clk) begin
        // Four 32-bit writes load each 128-bit word.  UltraRAM byte-write
        // enables map the selected quarter without a read/modify/write.
        if (cfg_weight_we)
            case (cfg_weight_lane)
                2'd0: weight_mem[cfg_weight_addr][31:0]   <= cfg_weight_data;
                2'd1: weight_mem[cfg_weight_addr][63:32]  <= cfg_weight_data;
                2'd2: weight_mem[cfg_weight_addr][95:64]  <= cfg_weight_data;
                2'd3: weight_mem[cfg_weight_addr][127:96] <= cfg_weight_data;
            endcase
        if (read_issue) weight_read <= weight_mem[weight_read_addr];
    end

    (* ram_style = "block" *) logic signed [31:0] bias_mem [0:BIAS_DEPTH-1];
    logic [31:0] weight_scale_q32 [0:2];
    always_ff @(posedge clk) begin
        if (cfg_bias_we) bias_mem[cfg_bias_addr] <= cfg_bias_q16;
        if (cfg_scale_we && cfg_scale_branch < 3)
            weight_scale_q32[cfg_scale_branch] <= cfg_scale_q32;
    end

    logic read_valid_d, read_valid_d2;
    logic [127:0] a_lanes, a_lanes_d, weight_read_d;
    logic [15:0] row_mask;
    logic [8191:0] accum_flat;
    for (genvar r = 0; r < 16; r = r + 1) begin : g_array_lane
        assign a_lanes[r*8 +: 8] = a_read[r];
        assign row_mask[r] = ((row_tile * 16 + r) < tokens_in_frame);
    end
    // Isolate the cascaded weight URAM output from the PE MAC path.  The
    // activation banks get the same register, keeping their k samples
    // aligned with the weight word and the delayed valid bit.
    always_ff @(posedge clk) begin
        a_lanes_d <= a_lanes;
        weight_read_d <= weight_read;
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) read_valid_d2 <= 1'b0;
        else read_valid_d2 <= read_valid_d;
    end
    tpu_systolic16x16_int8 array16 (
        .clk(clk), .rst_n(rst_n), .clear(state == TILE_CLEAR),
        .step_valid(read_valid_d2), .a_u8(a_lanes_d), .w_i8(weight_read_d),
        .a_zero_point(frame_zero_point), .row_mask(row_mask),
        .accum_flat(accum_flat)
    );

    // Stage 4: one result each clock while ready.  We keep the full INT32
    // accumulator; scale factors are unsigned Q24 and Q32 respectively.
    // Signed products are 32x34 -> 66 and 66x33 -> 99 bits.  Round half
    // away from zero at the final Q40 shift, then add the Q16 bias.
    logic p0_valid, p1_valid, p2_valid, p3_valid;
    logic signed [31:0] p0_acc, p0_bias;
    logic signed [65:0] p1_product;
    logic signed [98:0] p2_product;
    logic signed [59:0] p3_scaled_q16;
    logic signed [31:0] p1_bias, p2_bias, p3_bias;
    logic [1:0] p0_branch, p1_branch, p2_branch, p3_branch;
    logic [7:0] p0_token, p1_token, p2_token, p3_token;
    logic [8:0] p0_channel, p1_channel, p2_channel, p3_channel;
    logic p0_token_last, p1_token_last, p2_token_last, p3_token_last;
    logic p0_frame_last, p1_frame_last, p2_frame_last, p3_frame_last;
    logic [7:0] selected_token;
    logic [8:0] selected_channel;
    logic [10:0] bias_read_addr;
    logic output_advance, output_issue;
    assign output_advance = !m_valid || m_ready;
    assign output_issue = (state == TILE_OUT) && output_advance &&
                          (issue_index < tile_words);
    assign selected_token = row_tile * 16 + issue_index[7:4];
    assign selected_channel = col_tile * 16 + issue_index[3:0];
    assign bias_read_addr = branch * CHANNELS + selected_channel;

    // A signed arithmetic shift is floor(P/2^40).  The discarded 40 bits
    // form a nonnegative remainder.  Positive ties round up; negative ties
    // stay at the floor value.  This is exactly half-away-from-zero without
    // a 99-bit negate/absolute-value carry chain.
    function automatic logic signed [59:0] round_product_q16(
        input logic signed [98:0] product
    );
        logic signed [58:0] quotient;
        logic increment;
        begin
            quotient = product[98:40];
            increment = product[39] &&
                        (!product[98] || (|product[38:0]));
            round_product_q16 = $signed({quotient[58], quotient}) +
                                $signed({59'd0, increment});
        end
    endfunction

    function automatic logic signed [31:0] add_bias_saturate(
        input logic signed [59:0] scaled_q16,
        input logic signed [31:0] bias
    );
        logic signed [60:0] sum;
        begin
            sum = $signed({scaled_q16[59], scaled_q16}) +
                  $signed({{29{bias[31]}}, bias});
            if (sum[60:32] != {29{sum[31]}})
                add_bias_saturate = sum[60] ? 32'sh80000000 : 32'sh7fffffff;
            else
                add_bias_saturate = sum[31:0];
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= FILL;
            fill_token <= '0;
            fill_channel <= '0;
            tokens_in_frame <= '0;
            frame_scale_q24 <= '0;
            frame_zero_point <= '0;
            branch <= '0;
            row_tile <= '0;
            col_tile <= '0;
            k_issue <= '0;
            drain_count <= '0;
            issue_index <= '0;
            read_valid_d <= 1'b0;
            frame_error <= 1'b0;
        end else begin
            read_valid_d <= read_issue;
            case (state)
                FILL: if (input_fire) begin
                    if (fill_token == 0 && fill_channel == 0) begin
                        frame_scale_q24 <= s_scale_q24;
                        frame_zero_point <= s_zero_point;
                    end else if (s_scale_q24 != frame_scale_q24 ||
                                 s_zero_point != frame_zero_point)
                        frame_error <= 1'b1;
                    if (s_token_last != (fill_channel == CHANNELS-1))
                        frame_error <= 1'b1;
                    if (s_frame_last && fill_channel != CHANNELS-1)
                        frame_error <= 1'b1;
                    if (fill_channel == CHANNELS-1) begin
                        fill_channel <= '0;
                        if (s_frame_last) begin
                            tokens_in_frame <= fill_token + 1'b1;
                            branch <= '0;
                            row_tile <= '0;
                            col_tile <= '0;
                            state <= TILE_CLEAR;
                        end else if (fill_token == MAX_TOKENS-1) begin
                            frame_error <= 1'b1;
                            fill_token <= fill_token + 1'b1;
                        end else
                            fill_token <= fill_token + 1'b1;
                    end else
                        fill_channel <= fill_channel + 1'b1;
                end
                TILE_CLEAR: begin
                    k_issue <= '0;
                    state <= TILE_READ;
                end
                TILE_READ: begin
                    if (k_issue == CHANNELS-1) begin
                        k_issue <= k_issue + 1'b1;
                        drain_count <= '0;
                        state <= TILE_DRAIN;
                    end else
                        k_issue <= k_issue + 1'b1;
                end
                TILE_DRAIN: begin
                    // Last RAM response enters the extra register on the
                    // first drain edge and the array on the next edge.
                    // Bottom-right PE receives it 30 edges later.
                    if (drain_count == 6'd33) begin
                        issue_index <= '0;
                        state <= TILE_OUT;
                    end else
                        drain_count <= drain_count + 1'b1;
                end
                TILE_OUT: begin
                    if (output_issue) issue_index <= issue_index + 1'b1;
                    if (issue_index == tile_words && !p0_valid && !p1_valid &&
                        !p2_valid && !p3_valid && !m_valid)
                        state <= TILE_NEXT;
                end
                TILE_NEXT: begin
                    if (col_tile != COL_TILES-1) begin
                        col_tile <= col_tile + 1'b1;
                        state <= TILE_CLEAR;
                    end else if ((row_tile + 1) * 16 < tokens_in_frame) begin
                        col_tile <= '0;
                        row_tile <= row_tile + 1'b1;
                        state <= TILE_CLEAR;
                    end else if (branch != 2) begin
                        col_tile <= '0;
                        row_tile <= '0;
                        branch <= branch + 1'b1;
                        state <= TILE_CLEAR;
                    end else begin
                        // Previous m_frame_last has been accepted by now.
                        fill_token <= '0;
                        fill_channel <= '0;
                        state <= FILL;
                    end
                end
                default: state <= FILL;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p0_valid <= 0; p1_valid <= 0; p2_valid <= 0; p3_valid <= 0;
            m_valid <= 0;
            p0_acc <= '0; p0_bias <= '0;
            p1_product <= '0; p1_bias <= '0;
            p2_product <= '0; p2_bias <= '0;
            p3_scaled_q16 <= '0; p3_bias <= '0;
            p0_branch <= '0; p1_branch <= '0; p2_branch <= '0; p3_branch <= '0;
            p0_token <= '0; p1_token <= '0; p2_token <= '0; p3_token <= '0;
            p0_channel <= '0; p1_channel <= '0; p2_channel <= '0; p3_channel <= '0;
            p0_token_last <= 0; p1_token_last <= 0; p2_token_last <= 0; p3_token_last <= 0;
            p0_frame_last <= 0; p1_frame_last <= 0; p2_frame_last <= 0; p3_frame_last <= 0;
            m_data_q16 <= '0; m_branch <= '0;
            m_token_index <= '0; m_channel_index <= '0;
            m_token_last <= 0; m_frame_last <= 0;
        end else if (output_advance) begin
            p0_valid <= output_issue;
            if (output_issue) begin
                p0_acc <= $signed(accum_flat[
                    ((issue_index[7:4] * 16 + issue_index[3:0]) * 32) +: 32]);
                p0_bias <= bias_mem[bias_read_addr];
                p0_branch <= branch;
                p0_token <= selected_token;
                p0_channel <= selected_channel;
                p0_token_last <= selected_channel == CHANNELS-1;
                p0_frame_last <= (branch == 2) &&
                                 (selected_token == tokens_in_frame-1) &&
                                 (selected_channel == CHANNELS-1);
            end
            p1_valid <= p0_valid;
            if (p0_valid) begin
                p1_product <= $signed(p0_acc) *
                              $signed({1'b0, frame_scale_q24});
                p1_bias <= p0_bias;
                p1_branch <= p0_branch;
                p1_token <= p0_token;
                p1_channel <= p0_channel;
                p1_token_last <= p0_token_last;
                p1_frame_last <= p0_frame_last;
            end
            p2_valid <= p1_valid;
            if (p1_valid) begin
                p2_product <= p1_product *
                              $signed({1'b0, weight_scale_q32[p1_branch]});
                p2_bias <= p1_bias;
                p2_branch <= p1_branch;
                p2_token <= p1_token;
                p2_channel <= p1_channel;
                p2_token_last <= p1_token_last;
                p2_frame_last <= p1_frame_last;
            end
            p3_valid <= p2_valid;
            if (p2_valid) begin
                p3_scaled_q16 <= round_product_q16(p2_product);
                p3_bias <= p2_bias;
                p3_branch <= p2_branch;
                p3_token <= p2_token;
                p3_channel <= p2_channel;
                p3_token_last <= p2_token_last;
                p3_frame_last <= p2_frame_last;
            end
            m_valid <= p3_valid;
            if (p3_valid) begin
                m_data_q16 <= add_bias_saturate(p3_scaled_q16, p3_bias);
                m_branch <= p3_branch;
                m_token_index <= p3_token;
                m_channel_index <= p3_channel;
                m_token_last <= p3_token_last;
                m_frame_last <= p3_frame_last;
            end
        end
    end
endmodule
