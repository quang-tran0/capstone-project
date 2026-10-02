`timescale 1ns/1ps

(* use_dsp = "yes" *)
module pe #(
    parameter int A_WIDTH   = 8,
    parameter int B_WIDTH   = 8,
    parameter int ACC_WIDTH = 32
) (
    input  logic clk,
    input  logic rst_n,

    // Enable one systolic-array step
    input  logic step_en,

    // Clear local accumulator before starting a new output tile
    input  logic acc_clear,

    // Bias/partial sum in accumulator units. Load does not require step_en.
    // Priority: reset > clear > load > MAC.
    input  logic acc_load,
    input  logic signed [ACC_WIDTH-1:0] acc_init,

    // Operand A: propagates horizontally
    input  logic signed [A_WIDTH-1:0] a_in,
    input  logic                      a_valid_in,

    // Operand B: propagates vertically
    input  logic signed [B_WIDTH-1:0] b_in,
    input  logic                      b_valid_in,

    // Forwarded A
    output logic signed [A_WIDTH-1:0] a_out,
    output logic                      a_valid_out,

    // Forwarded B
    output logic signed [B_WIDTH-1:0] b_out,
    output logic                      b_valid_out,

    // Output-stationary accumulator
    output logic signed [ACC_WIDTH-1:0] acc_out
);
    localparam int PRODUCT_WIDTH = A_WIDTH + B_WIDTH;

    logic signed [PRODUCT_WIDTH-1:0] product;

    assign product = a_in * b_in;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            a_valid_out <= 1'b0;
            b_valid_out <= 1'b0;
        end
        else if (step_en) begin
            a_out       <= a_in;
            a_valid_out <= a_valid_in;

            b_out       <= b_in;
            b_valid_out <= b_valid_in;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            acc_out <= '0;
        end
        else if (acc_clear) begin
            acc_out <= '0;
        end
        else if (acc_load) begin
            acc_out <= acc_init;
        end
        else if (
            step_en   &&
            a_valid_in &&
            b_valid_in
        ) begin
            acc_out <= acc_out + product;
        end
    end


endmodule
