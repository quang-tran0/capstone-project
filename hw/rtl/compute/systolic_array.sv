`timescale 1ns/1ps

// Output-stationary mesh. The caller supplies ALREADY SKEWED input streams:
// A[i,k] enters row i at step k+i; B[k,j] enters column j at step k+j.
// No input scheduler, output register or extra MAC pipeline is added here.
module systolic_array #(
    parameter int ROWS = 16,
    parameter int COLS = 16,
    parameter int A_WIDTH = 8,
    parameter int B_WIDTH = 8,
    parameter int ACC_WIDTH = 32,
    localparam int ROW_INDEX_WIDTH = (ROWS > 1) ? $clog2(ROWS) : 1
) (
    input logic clk,
    input logic rst_n,
    input logic step_en,
    input logic acc_clear,

    // Load COLS accumulators in the selected row; works with step_en=0.
    input logic acc_load,
    input logic [ROW_INDEX_WIDTH-1:0] acc_row,
    input wire signed [ACC_WIDTH-1:0] acc_init [COLS],

    input wire signed [A_WIDTH-1:0] a_in [ROWS],
    input wire signed [B_WIDTH-1:0] b_in [COLS],
    input logic [ROWS-1:0] a_valid_in,
    input logic [COLS-1:0] b_valid_in,
    output wire signed [ACC_WIDTH-1:0] acc_out [ROWS][COLS]
);
    // Link index denotes the boundary BEFORE a PE.
    // a_link[r][0] is the left edge; b_link[0][c] is the top edge.
    wire signed [A_WIDTH-1:0] a_link [ROWS][COLS+1];
    wire signed [B_WIDTH-1:0] b_link [ROWS+1][COLS];
    wire av_link [ROWS][COLS+1];
    wire bv_link [ROWS+1][COLS];

    initial begin
        if (ROWS < 1 || COLS < 1 || A_WIDTH < 1 || B_WIDTH < 1 ||
            ACC_WIDTH < A_WIDTH + B_WIDTH)
            $fatal(1, "Invalid systolic_array dimensions or arithmetic widths");
    end

    for (genvar c = 0; c < COLS; c++) begin : top_edge
        assign b_link[0][c] = b_in[c];
        assign bv_link[0][c] = b_valid_in[c];
    end

    for (genvar r = 0; r < ROWS; r++) begin : row
        wire load_row;
        assign load_row = acc_load && (acc_row == ROW_INDEX_WIDTH'(r));
        assign a_link[r][0] = a_in[r];
        assign av_link[r][0] = a_valid_in[r];

        for (genvar c = 0; c < COLS; c++) begin : col
            pe #(
                .A_WIDTH(A_WIDTH), .B_WIDTH(B_WIDTH), .ACC_WIDTH(ACC_WIDTH)
            ) u_pe (
                .clk(clk), .rst_n(rst_n), .step_en(step_en),
                .acc_clear(acc_clear), .acc_load(load_row), .acc_init(acc_init[c]),
                .a_in(a_link[r][c]), .a_valid_in(av_link[r][c]),
                .b_in(b_link[r][c]), .b_valid_in(bv_link[r][c]),

                // Each PE registers and forwards A right, B down.
                .a_out(a_link[r][c+1]), .a_valid_out(av_link[r][c+1]),
                .b_out(b_link[r+1][c]), .b_valid_out(bv_link[r+1][c]),
                .acc_out(acc_out[r][c])
            );
        end
    end
    // The right/bottom links are intentionally unused. Results stay local.
    // acc_clear/load do NOT flush valid links: drain the mesh or reset it
    // before starting an independent tile. Clear/load priority is in pe.sv.
endmodule
