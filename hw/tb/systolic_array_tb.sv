`timescale 1ns/1ps

// Self-checking testbench for systolic_array (output-stationary mesh of pe).
// PE requirements R1-R9: docs/hw/pe.md. The reference is an ordinary matrix
// multiply, independent of the PE/mesh wiring.
// Inputs change on the falling edge; outputs are checked 1 ns after the rising edge.
module systolic_array_tb;
    parameter int ROWS = 16;
    parameter int COLS = 16;
    localparam int MAX_K = 1536;  // MiniLM FFN dense2, practical bound of guide 07

    logic clk = 0, rst_n = 0, step_en = 0, acc_clear = 0;
    logic signed [7:0] a_in [ROWS], b_in [COLS];
    logic [ROWS-1:0] a_valid_in = '0;
    logic [COLS-1:0] b_valid_in = '0;
    wire signed [31:0] acc_out [ROWS][COLS];

    integer matrix_a [ROWS][MAX_K], matrix_b [MAX_K][COLS];
    logic signed [31:0] expected [ROWS][COLS], snapshot [ROWS][COLS];
    logic [31:0] random_state = 32'h12345678;
    integer cases_passed = 0;

    systolic_array #(.ROWS(ROWS), .COLS(COLS)) dut (.*);
    always #5 clk = ~clk;
    initial begin
        #200_000_000;
        $fatal(1, "Timeout");
    end

    function automatic integer random_int8();
        random_state = random_state * 32'd1664525 + 32'd1013904223;
        return $signed(random_state[31:24]);
    endfunction

    task automatic check_result(input string label_text);
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++)
                if (acc_out[i][j] !== expected[i][j])
                    $fatal(1, "%s C[%0d,%0d]: got %0d expected %0d",
                           label_text, i, j, acc_out[i][j], expected[i][j]);
    endtask

    task automatic set_expected(input logic signed [31:0] value);
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++) expected[i][j] = value;
    endtask

    // Invalid lanes carry garbage on purpose: valid, not data, gates the MAC.
    task automatic idle_inputs;
        step_en = 0;
        a_valid_in = '0;
        b_valid_in = '0;
        for (int i = 0; i < ROWS; i++) a_in[i] = 8'(random_int8());
        for (int j = 0; j < COLS; j++) b_in[j] = 8'(random_int8());
    endtask

    task automatic reset_array;
        @(negedge clk);
        rst_n = 0;
        acc_clear = 0;
        idle_inputs();
        set_expected(0);
        @(posedge clk); #1;
        check_result("reset");
        @(negedge clk);
        rst_n = 1;
    endtask

    // Start a new tile with acc_clear only (no reset): R6.
    task automatic clear_array;
        @(negedge clk);
        acc_clear = 1;
        idle_inputs();
        set_expected(0);
        @(posedge clk); #1;
        check_result("acc_clear");
        @(negedge clk);
        acc_clear = 0;
    endtask

    // mode: 0=zeros, 1=identity B, 2=symmetric INT8 extremes (+-127, the codes the
    //       model uses), 3=random including -128, 4=random with paired bubbles.
    // use_reset=0 starts the tile with acc_clear, keeping the previous tile's sums
    // until then, so a disconnected clear is caught.
    task automatic run_matrix(input int k_count, mode, input bit stalls, use_reset);
        int k;
        if (use_reset) reset_array();
        else clear_array();
        for (int i = 0; i < ROWS; i++)
            for (int n = 0; n < k_count; n++)
                case (mode)
                    0: matrix_a[i][n] = 0;
                    1: matrix_a[i][n] = (i*7 + n*3) % 31 - 15;
                    2: matrix_a[i][n] = (i % 2) ? 127 : -127;
                    default: matrix_a[i][n] = random_int8();
                endcase
        for (int n = 0; n < k_count; n++)
            for (int j = 0; j < COLS; j++)
                case (mode)
                    0: matrix_b[n][j] = 0;
                    1: matrix_b[n][j] = (n == j) ? 1 : 0;
                    2: matrix_b[n][j] = (j % 2) ? -127 : 127;
                    default: matrix_b[n][j] = random_int8();
                endcase

        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++) begin
                expected[i][j] = 0;
                for (int n = 0; n < k_count; n++)
                    if (!(mode == 4 && n % 4 == 1))
                        expected[i][j] = expected[i][j] + matrix_a[i][n]*matrix_b[n][j];
            end

        // t counts ENABLED steps. At t=k+i+j, A[i,k] meets B[k,j].
        for (int t = 0; t < k_count + ROWS + COLS - 2; t++) begin
            @(negedge clk);
            for (int i = 0; i < ROWS; i++) begin
                k = t - i;
                a_valid_in[i] = (k >= 0 && k < k_count);
                a_in[i] = a_valid_in[i] ? 8'(matrix_a[i][k]) : 8'(random_int8());
                if (mode == 4 && k % 4 == 1) a_valid_in[i] = 0;
            end
            for (int j = 0; j < COLS; j++) begin
                k = t - j;
                b_valid_in[j] = (k >= 0 && k < k_count);
                b_in[j] = b_valid_in[j] ? 8'(matrix_b[k][j]) : 8'(random_int8());
                if (mode == 4 && k % 4 == 1) b_valid_in[j] = 0;
            end
            // Stall: the source holds its inputs; nothing may change (R3, R5).
            if (stalls && t % 5 == 2) begin
                step_en = 0;
                for (int i = 0; i < ROWS; i++)
                    for (int j = 0; j < COLS; j++) snapshot[i][j] = acc_out[i][j];
                repeat (2) begin
                    @(posedge clk); #1;
                    for (int i = 0; i < ROWS; i++)
                        for (int j = 0; j < COLS; j++)
                            if (acc_out[i][j] !== snapshot[i][j])
                                $fatal(1, "Stall changed C[%0d,%0d]", i, j);
                    @(negedge clk);
                end
            end
            step_en = 1;
            @(posedge clk); #1;
        end
        check_result("matrix multiply");
        @(negedge clk);
        idle_inputs();
        cases_passed++;
        $display("PASS %0dx%0d K=%0d mode=%0d stalls=%0b start=%s",
                 ROWS, COLS, k_count, mode, stalls, use_reset ? "reset" : "clear");
    endtask

    initial begin
        for (int i = 0; i < ROWS; i++) a_in[i] = 0;
        for (int j = 0; j < COLS; j++) b_in[j] = 0;

        run_matrix(16, 0, 0, 1);
        run_matrix(16, 1, 0, 0);
        run_matrix(16, 2, 1, 0);
        run_matrix(1, 3, 1, 0);
        run_matrix(16, 3, 0, 0);
        run_matrix(19, 3, 1, 0);
        run_matrix(19, 4, 1, 0);
        run_matrix(33, 3, 1, 1);
        // Practical bound: K = 1536 of +-127 codes, |acc| = 24 774 144 < 2^25.
        run_matrix(MAX_K, 2, 0, 0);
        run_matrix(MAX_K, 3, 0, 0);

        // Clear during a stall clears every PE (R6, even with step_en = 0).
        run_matrix(8, 3, 0, 0);
        @(negedge clk);
        acc_clear = 1;
        step_en = 0;
        set_expected(0);
        @(posedge clk); #1;
        check_result("clear while stalled");

        // Flush the mesh with invalid data so every internal valid is 0
        // (acc must not change: no valid pairs).
        @(negedge clk);
        acc_clear = 0;
        step_en = 1;
        repeat (ROWS + COLS) begin
            @(posedge clk); #1;
            check_result("flush with invalid data");
        end

        // Clear in the same cycle as a valid MAC discards that product (R6),
        // but forwarding continues (acc_clear does not touch the A/B path).
        @(negedge clk);
        acc_clear = 1;
        step_en = 1;
        a_valid_in = '1;
        b_valid_in = '1;
        for (int i = 0; i < ROWS; i++) a_in[i] = -2;
        for (int j = 0; j < COLS; j++) b_in[j] = 3;
        @(posedge clk); #1;
        check_result("clear wins over MAC");
        // Next step: A is valid in columns 0 (edge) and 1 (forwarded during the
        // clear cycle), B in rows 0 and 1, so exactly PE[0..1][0..1] MAC.
        @(negedge clk);
        acc_clear = 0;
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++)
                expected[i][j] = (i <= 1 && j <= 1) ? -6 : 0;
        @(posedge clk); #1;
        check_result("forwarding continues through clear");

        // Start an unfinished tile, then reset with clear released (R1, R7).
        @(negedge clk);
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 0;
        idle_inputs();
        set_expected(0);
        @(posedge clk); #1;
        check_result("reset wins during active tile");
        // Release reset; flush invalid data to make sure reset cleared the
        // internal valid bits, not just the sums.
        @(negedge clk);
        rst_n = 1;
        step_en = 1;
        repeat (ROWS + COLS) begin
            @(posedge clk); #1;
            check_result("reset cleared valid pipeline");
        end
        run_matrix(19, 3, 1, 0);
        $display("TEST PASSED: systolic array %0dx%0d, %0d matrix cases + controls",
                 ROWS, COLS, cases_passed);
        $finish;
    end
endmodule
