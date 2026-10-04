`timescale 1ns/1ps

module pe_tb;
    logic clk = 0;
    logic rst_n = 0;
    logic step_en = 0, acc_clear = 0, acc_load = 0;
    logic signed [31:0] acc_init = 0;
    logic signed [7:0] a_in = 0, a_out;
    logic signed [7:0] b_in = 0, b_out;
    logic a_valid_in = 0, b_valid_in = 0;
    logic a_valid_out, b_valid_out;
    logic signed [31:0] acc_out;

    logic signed [7:0] expected_a = 0;
    logic signed [7:0] expected_b = 0;
    logic expected_av = 0, expected_bv = 0;

    pe dut (.*);
    always #5 clk = ~clk;

    task automatic step(
        input bit en, clear_acc, load_acc,
        input logic signed [31:0] initial_acc,
        input integer a, b,
        input bit av, bv,
        input logic signed [31:0] expected_acc
    );
        @(negedge clk);
        step_en = en;
        acc_clear = clear_acc;
        acc_load = load_acc;
        acc_init = initial_acc;
        a_in = a;
        b_in = b;
        a_valid_in = av;
        b_valid_in = bv;
        if (en) begin
            expected_a = a;
            expected_b = b;
            expected_av = av;
            expected_bv = bv;
        end
        @(posedge clk);
        #1;
        if (acc_out !== expected_acc)
            $fatal(1, "Accumulator: got %0d, expected %0d", acc_out, expected_acc);
        if ({a_out, b_out, a_valid_out, b_valid_out} !==
            {expected_a, expected_b, expected_av, expected_bv})
            $fatal(1, "Operand/valid forwarding or stall mismatch");
    endtask

    initial begin
        @(posedge clk);
        #1;
        if ({acc_out, a_valid_out, b_valid_out} !== '0)
            $fatal(1, "Synchronous reset failed");
        @(negedge clk);
        rst_n = 1;

        step(1, 0, 0, 0,    2,    3, 1, 1,      6);
        step(1, 0, 0, 0,   -4,    5, 1, 1,    -14);
        step(1, 0, 0, 0,    1,   -2, 1, 1,    -16);
        step(1, 0, 0, 0,   10,   10, 0, 1,    -16);
        step(1, 0, 0, 0,   10,   10, 1, 0,    -16);
        step(1, 0, 0, 0,   10,   10, 0, 0,    -16);
        step(0, 0, 0, 0,   99,  -99, 1, 1,    -16);
        step(0, 1, 0, 0,   99,  -99, 1, 1,      0);
        step(1, 0, 0, 0, -128, -128, 1, 1,  16384);
        step(1, 0, 0, 0,  127,  127, 1, 1,  32513);
        step(1, 0, 0, 0, -128,  127, 1, 1,  16257);
        step(1, 0, 0, 0,  127, -128, 1, 1,      1);
        step(1, 1, 0, 0,    2,    3, 1, 1,      0);

        // Load works while stalled and overrides MAC, but not clear.
        step(0, 0, 1, 100, 2, 3, 1, 1, 100);
        step(1, 0, 0,   0, 2, 3, 1, 1, 106);
        step(1, 0, 1, -20, 2, 3, 1, 1, -20);
        step(1, 0, 0,   0, 2, 3, 1, 1, -14);
        step(1, 0, 1,   7, 2, 3, 0, 0,   7);
        step(0, 0, 1,  -8, 2, 3, 0, 0,  -8);
        step(0, 1, 1, 123, 2, 3, 1, 1,   0);
        step(1, 1, 1, 123, 2, 3, 1, 1,   0);

        // Signed 32-bit overflow wraps; no saturation.
        step(0, 0, 1, 32'sh7ffffffe, 0, 0, 0, 0, 32'sh7ffffffe);
        step(1, 0, 0, 0,  1, 2, 1, 1, 32'sh80000000);
        step(1, 0, 0, 0, -1, 1, 1, 1, 32'sh7fffffff);

        // Reset overrides load and MAC, without resetting operand data.
        @(negedge clk);
        rst_n = 0;
        acc_load = 1;
        acc_init = 123;
        a_in = 10;
        b_in = 20;
        @(posedge clk);
        #1;
        if ({acc_out, a_valid_out, b_valid_out} !== '0)
            $fatal(1, "Reset after accumulation failed");
        if ({a_out, b_out} !== {expected_a, expected_b})
            $fatal(1, "Reset should hold operand data");
        $display("TEST PASSED: MAC, forwarding, stall, preload, priority, signed wrap, synchronous reset");
        $finish;
    end
endmodule
