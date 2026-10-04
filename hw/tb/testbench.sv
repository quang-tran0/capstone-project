`timescale 1ns/1ps

module testbench;
    localparam int ARRAY_SIZE = 8;
    localparam int MATRIX_SIZE = 4;

    logic clk = 0;
    logic rst_n = 0;
    logic step_en = 0;
    logic signed [7:0] a_in [ARRAY_SIZE];
    logic signed [7:0] b_in [ARRAY_SIZE];
    logic [ARRAY_SIZE-1:0] a_valid_in = '0;
    logic [ARRAY_SIZE-1:0] b_valid_in = '0;
    logic signed [31:0] acc_init [ARRAY_SIZE];
    wire signed [31:0] acc_out [ARRAY_SIZE][ARRAY_SIZE];

    int matrix_a [MATRIX_SIZE][MATRIX_SIZE] = '{
        '{ 1,  2,  3,  4},
        '{ 5,  6,  7,  8},
        '{ 9, 10, 11, 12},
        '{13, 14, 15, 16}
    };
    int matrix_b [MATRIX_SIZE][MATRIX_SIZE] = '{
        '{16, 15, 14, 13},
        '{12, 11, 10,  9},
        '{ 8,  7,  6,  5},
        '{ 4,  3,  2,  1}
    };
    int expected [ARRAY_SIZE][ARRAY_SIZE];

    systolic_array #(
        .ROWS(ARRAY_SIZE),
        .COLS(ARRAY_SIZE),
        .A_WIDTH(8),
        .B_WIDTH(8),
        .ACC_WIDTH(32)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .step_en(step_en),
        .acc_clear(1'b0),
        .acc_load(1'b0),
        .acc_row(3'd0),
        .acc_init(acc_init),
        .a_in(a_in),
        .b_in(b_in),
        .a_valid_in(a_valid_in),
        .b_valid_in(b_valid_in),
        .acc_out(acc_out)
    );

    always #5 clk = ~clk;

    initial begin
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            a_in[i] = 0;
            b_in[i] = 0;
            acc_init[i] = 0;
            for (int j = 0; j < ARRAY_SIZE; j++)
                expected[i][j] = 0;
        end

        // Reference: ordinary matrix multiplication C[i,j] = sum(A[i,k]*B[k,j]).
        for (int i = 0; i < MATRIX_SIZE; i++)
            for (int j = 0; j < MATRIX_SIZE; j++)
                for (int k = 0; k < MATRIX_SIZE; k++)
                    expected[i][j] += matrix_a[i][k] * matrix_b[k][j];

        repeat (2) @(posedge clk);

        // Skew inputs: A[i,k] enters at step k+i, B[k,j] at step k+j.
        // The last product reaches PE[3][3] at step 3+3+3 = 9 (10 steps).
        for (int t = 0; t < 3*MATRIX_SIZE - 2; t++) begin
            @(negedge clk);
            rst_n = 1;
            step_en = 1;
            for (int lane = 0; lane < ARRAY_SIZE; lane++) begin
                a_in[lane] = 0;
                b_in[lane] = 0;
                a_valid_in[lane] = 0;
                b_valid_in[lane] = 0;
                // Unused rows/columns 4..7 always receive invalid inputs.
                if (lane < MATRIX_SIZE && t >= lane && t-lane < MATRIX_SIZE) begin
                    a_in[lane] = 8'(matrix_a[lane][t-lane]);
                    b_in[lane] = 8'(matrix_b[t-lane][lane]);
                    a_valid_in[lane] = 1;
                    b_valid_in[lane] = 1;
                end
            end
        end

        @(negedge clk);
        step_en = 0;
        a_valid_in = '0;
        b_valid_in = '0;

        // Check all 64 PEs, including the unused area which must remain zero.
        for (int i = 0; i < ARRAY_SIZE; i++)
            for (int j = 0; j < ARRAY_SIZE; j++)
                if (acc_out[i][j] !== expected[i][j])
                    $fatal(1, "FAIL: C[%0d][%0d] = %0d, expected %0d",
                           i, j, acc_out[i][j], expected[i][j]);

        $display("C = A * B (top-left 4x4 of the 8x8 array):");
        for (int i = 0; i < MATRIX_SIZE; i++) begin
            for (int j = 0; j < MATRIX_SIZE; j++)
                $write("%6d ", acc_out[i][j]);
            $write("\n");
        end
        $display("PASS: 4x4 matrix multiplication on an 8x8 systolic array");
        $finish;
    end
endmodule
