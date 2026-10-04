`timescale 1ns/1ps

module systolic_array_tb;
    parameter int ROWS = 16;
    parameter int COLS = 16;
    localparam int ROW_INDEX_WIDTH = (ROWS > 1) ? $clog2(ROWS) : 1;
    localparam int MAX_K = 19;

    logic clk = 0, rst_n = 0, step_en = 0;
    logic acc_clear = 0, acc_load = 0;
    logic [ROW_INDEX_WIDTH-1:0] acc_row = 0;
    logic signed [7:0] a_in [ROWS], b_in [COLS];
    logic [ROWS-1:0] a_valid_in = '0;
    logic [COLS-1:0] b_valid_in = '0;
    logic signed [31:0] acc_init [COLS];
    wire signed [31:0] acc_out [ROWS][COLS];

    // Reference uses ordinary matrix multiplication, not the PE/mesh wiring.
    integer matrix_a [ROWS][MAX_K], matrix_b [MAX_K][COLS];
    logic signed [31:0] expected [ROWS][COLS], snapshot [ROWS][COLS];
    logic [31:0] random_state = 32'h12345678;
    integer cases_passed = 0;

    systolic_array #(.ROWS(ROWS), .COLS(COLS)) dut (.*);
    always #5 clk = ~clk;
    initial begin
        #1000000;
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

    task automatic reset_array;
        @(negedge clk);
        rst_n = 0;
        step_en = 0;
        acc_load = 0;
        acc_clear = 0;
        a_valid_in = '0;
        b_valid_in = '0;
        for (int i = 0; i < ROWS; i++) begin
            a_in[i] = 0;
            for (int j = 0; j < COLS; j++) expected[i][j] = 0;
        end
        for (int j = 0; j < COLS; j++) begin
            b_in[j] = 0;
            acc_init[j] = 0;
        end
        @(posedge clk); #1;
        check_result("reset");
        @(negedge clk);
        rst_n = 1;
    endtask

    // mode: 0=zeros, 1=identity B, 2=int8 extremes + overflow,
    //       3=seeded random, 4=random with a paired invalid k (bubble).
    task automatic run_matrix(input int k_count, mode, input bit stalls);
        int k;
        reset_array();
        for (int i = 0; i < ROWS; i++)
            for (int n = 0; n < k_count; n++) begin
                case (mode)
                    0: matrix_a[i][n] = 0;
                    1: matrix_a[i][n] = (i*7 + n*3) % 31 - 15;
                    2: matrix_a[i][n] = (i % 2) ? 127 : -128;
                    default: matrix_a[i][n] = random_int8();
                endcase
            end
        for (int n = 0; n < k_count; n++)
            for (int j = 0; j < COLS; j++) begin
                case (mode)
                    0: matrix_b[n][j] = 0;
                    1: matrix_b[n][j] = (n == j) ? 1 : 0;
                    2: matrix_b[n][j] = (j % 2) ? -128 : 127;
                    default: matrix_b[n][j] = random_int8();
                endcase
            end

        // Load one row per clock while the entire mesh is stalled.
        // Checking after EVERY row detects accidentally loading other rows.
        for (int i = 0; i < ROWS; i++) begin
            @(negedge clk);
            acc_load = 1;
            acc_row = ROW_INDEX_WIDTH'(i);
            for (int j = 0; j < COLS; j++) begin
                acc_init[j] = (mode == 2) ? 32'sh7ffffff0 : i*100 - j*3 - 42;
                expected[i][j] = acc_init[j];
            end
            @(posedge clk); #1;
            check_result("row preload while stalled");
        end
        @(negedge clk);
        acc_load = 0;

        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++)
                for (int n = 0; n < k_count; n++)
                    if (!(mode == 4 && n % 4 == 1))
                        expected[i][j] = expected[i][j] + matrix_a[i][n]*matrix_b[n][j];

        // t counts ENABLED steps. At t=k+i+j, A[i,k] meets B[k,j].
        for (int t = 0; t < k_count + ROWS + COLS - 2; t++) begin
            @(negedge clk);
            for (int i = 0; i < ROWS; i++) begin
                k = t - i;
                a_valid_in[i] = (k >= 0 && k < k_count);
                a_in[i] = a_valid_in[i] ? matrix_a[i][k] : 0;
                if (mode == 4 && k % 4 == 1) a_valid_in[i] = 0;
            end
            for (int j = 0; j < COLS; j++) begin
                k = t - j;
                b_valid_in[j] = (k >= 0 && k < k_count);
                b_in[j] = b_valid_in[j] ? matrix_b[k][j] : 0;
                if (mode == 4 && k % 4 == 1) b_valid_in[j] = 0;
            end
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
        step_en = 0;
        a_valid_in = '0;
        b_valid_in = '0;
        cases_passed++;
        $display("PASS %0dx%0d K=%0d mode=%0d stalls=%0b", ROWS, COLS, k_count, mode, stalls);
    endtask

    initial begin
        run_matrix(16, 0, 0);
        run_matrix(16, 1, 0);
        run_matrix(16, 2, 1);
        run_matrix(1, 3, 1);
        run_matrix(16, 3, 0);
        run_matrix(19, 3, 1);
        run_matrix(19, 4, 1);

        reset_array();
        // Out-of-range row codes must not alias a real row (e.g. ROWS=3).
        if ((2**ROW_INDEX_WIDTH) > ROWS) begin
            @(negedge clk);
            acc_load = 1;
            acc_row = ROW_INDEX_WIDTH'(ROWS);
            for (int j = 0; j < COLS; j++) acc_init[j] = 99;
            @(posedge clk); #1;
            check_result("invalid row ignored");
        end
        // With step_en=1, only the loaded row skips MAC. Other rows advance.
        @(negedge clk);
        acc_load = 1;
        acc_row = ROW_INDEX_WIDTH'(ROWS-1);
        step_en = 1;
        a_valid_in = '1;
        b_valid_in = '1;
        for (int i = 0; i < ROWS; i++) a_in[i] = -2;
        for (int j = 0; j < COLS; j++) begin
            b_in[j] = 3;
            acc_init[j] = 123;
            expected[ROWS-1][j] = 123;
        end
        if (ROWS > 1) expected[0][0] = -6;
        @(posedge clk); #1;
        check_result("load selected row while other rows MAC");

        // Clear wins over row load even when stalled.
        @(negedge clk);
        step_en = 0;
        acc_clear = 1;
        acc_load = 1;
        acc_row = 0;
        for (int j = 0; j < COLS; j++) acc_init[j] = 123;
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++) expected[i][j] = 0;
        @(posedge clk); #1;
        check_result("clear wins over load");

        // Start an unfinished tile, then reset with load still asserted.
        @(negedge clk);
        acc_clear = 0;
        acc_load = 0;
        step_en = 1;
        a_valid_in = '1;
        b_valid_in = '1;
        for (int i = 0; i < ROWS; i++) a_in[i] = -2;
        for (int j = 0; j < COLS; j++) b_in[j] = 3;
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 0;
        acc_load = 1;
        @(posedge clk); #1;
        check_result("reset wins during active tile");
        // Release reset WITHOUT a second reset; flush invalid data to ensure
        // reset actually cleared the internal valid bits, not just the sums.
        @(negedge clk);
        rst_n = 1;
        acc_load = 0;
        a_valid_in = '0;
        b_valid_in = '0;
        repeat (ROWS + COLS) begin
            @(posedge clk); #1;
            check_result("reset cleared valid pipeline");
        end
        run_matrix(19, 3, 1);
        $display("TEST PASSED: systolic array %0dx%0d, %0d matrix cases + controls", ROWS, COLS, cases_passed);
        $finish;
    end
endmodule
