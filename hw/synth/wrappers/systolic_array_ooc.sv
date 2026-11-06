`timescale 1ns/1ps

// Synthesis-only wrapper (not part of the RTL). Registers every array input,
// as the skew_buffer / controller will in gemm_tile, so the input-to-PE paths
// are timed register-to-register. acc_out already comes straight from PE
// registers and is left unregistered.
module systolic_array_ooc #(
    parameter int ROWS = 16,
    parameter int COLS = 16,
    parameter int A_WIDTH = 8,
    parameter int B_WIDTH = 8,
    parameter int ACC_WIDTH = 32
) (
    input  logic clk,
    input  logic rst_n,
    input  logic step_en,
    input  logic acc_clear,
    input  wire  signed [A_WIDTH-1:0]   a_in [ROWS],
    input  wire  signed [B_WIDTH-1:0]   b_in [COLS],
    input  logic [ROWS-1:0]             a_valid_in,
    input  logic [COLS-1:0]             b_valid_in,
    output wire  signed [ACC_WIDTH-1:0] acc_out [ROWS][COLS]
);
    logic rst_n_q, step_en_q, acc_clear_q;
    logic signed [A_WIDTH-1:0] a_q [ROWS];
    logic signed [B_WIDTH-1:0] b_q [COLS];
    logic [ROWS-1:0] av_q;
    logic [COLS-1:0] bv_q;

    always_ff @(posedge clk) begin
        rst_n_q     <= rst_n;
        step_en_q   <= step_en;
        acc_clear_q <= acc_clear;
        a_q         <= a_in;
        b_q         <= b_in;
        av_q        <= a_valid_in;
        bv_q        <= b_valid_in;
    end

    systolic_array #(
        .ROWS(ROWS), .COLS(COLS),
        .A_WIDTH(A_WIDTH), .B_WIDTH(B_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) u_array (
        .clk(clk), .rst_n(rst_n_q), .step_en(step_en_q), .acc_clear(acc_clear_q),
        .a_in(a_q), .b_in(b_q), .a_valid_in(av_q), .b_valid_in(bv_q),
        .acc_out(acc_out)
    );
endmodule
