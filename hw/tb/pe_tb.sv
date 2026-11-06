`timescale 1ns/1ps

// Self-checking testbench for pe. Spec: docs/hw/pe.md (requirements R1-R9).
// Inputs change on the falling edge; outputs are checked 1 ns after the rising edge.
module pe_tb;
    localparam int A_WIDTH   = 8;
    localparam int B_WIDTH   = 8;
    localparam int ACC_WIDTH = 32;

    // ---------------- DUT ----------------
    logic clk = 0, rst_n = 0, step_en = 0, acc_clear = 0;
    logic signed [A_WIDTH-1:0]   a_in = 0;
    logic signed [B_WIDTH-1:0]   b_in = 0;
    logic                        a_valid_in = 0, b_valid_in = 0;
    logic signed [A_WIDTH-1:0]   a_out;
    logic signed [B_WIDTH-1:0]   b_out;
    logic                        a_valid_out, b_valid_out;
    logic signed [ACC_WIDTH-1:0] acc_out;

    pe #(
        .A_WIDTH(A_WIDTH), .B_WIDTH(B_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n), .step_en(step_en), .acc_clear(acc_clear),
        .a_in(a_in), .a_valid_in(a_valid_in),
        .b_in(b_in), .b_valid_in(b_valid_in),
        .a_out(a_out), .a_valid_out(a_valid_out),
        .b_out(b_out), .b_valid_out(b_valid_out),
        .acc_out(acc_out)
    );

    // ---------------- Clock and watchdog ----------------
    always #5 clk = ~clk;

    initial begin
        #10_000_000;
        $fatal(1, "Timeout");
    end

    // ---------------- Reference model ----------------
    // What the PE must hold after each rising edge.
    logic signed [ACC_WIDTH-1:0] ref_acc = 0;
    logic signed [A_WIDTH-1:0]   ref_a = 0;
    logic signed [B_WIDTH-1:0]   ref_b = 0;
    logic                        ref_av = 0, ref_bv = 0;
    bit                          data_known = 0;  // a_out/b_out are not reset (R1)
    int                          errors = 0;

    task automatic fail(input string msg);
        errors++;
        if (errors <= 10)
            $display("ERROR at %0t ns: %s", $time, msg);
    endtask

    task automatic check();
        if (acc_out !== ref_acc)
            fail($sformatf("acc_out = %0d, expected %0d", acc_out, ref_acc));
        if (a_valid_out !== ref_av || b_valid_out !== ref_bv)
            fail($sformatf("valid_out = %b%b, expected %b%b",
                           a_valid_out, b_valid_out, ref_av, ref_bv));
        if (data_known && (a_out !== ref_a || b_out !== ref_b))
            fail($sformatf("a_out/b_out = %0d/%0d, expected %0d/%0d",
                           a_out, b_out, ref_a, ref_b));
    endtask

    // One clock cycle: drive the inputs, update the model, check after the edge.
    // rst = 1 asserts reset (rst_n = 0) for this cycle.
    task automatic step(input bit rst, en, clr, input int a, b, input bit av, bv);
        logic signed [A_WIDTH-1:0] a_s;
        logic signed [B_WIDTH-1:0] b_s;
        a_s = a;
        b_s = b;

        @(negedge clk);
        rst_n      = !rst;
        step_en    = en;
        acc_clear  = clr;
        a_in       = a_s;
        b_in       = b_s;
        a_valid_in = av;
        b_valid_in = bv;

        // Accumulator: reset > clear > MAC > hold (R1, R7, R6, R2, R3).
        // The product is signed and the sum wraps at ACC_WIDTH bits (R8, R9).
        if (rst)                 ref_acc = 0;
        else if (clr)            ref_acc = 0;
        else if (en && av && bv) ref_acc = ref_acc + a_s * b_s;

        // Forwarding: valids reset, data held; moves only when step_en (R1, R4, R5).
        if (rst) begin
            ref_av = 0;
            ref_bv = 0;
        end else if (en) begin
            ref_a      = a_s;
            ref_b      = b_s;
            ref_av     = av;
            ref_bv     = bv;
            data_known = 1;
        end

        @(posedge clk);
        #1;
        check();
    endtask

    // Random INT8 operand; one time in five an edge value.
    function automatic int pick();
        int r;
        if ($urandom_range(0, 4) == 0) begin
            case ($urandom_range(0, 5))
                0: return -128;
                1: return -127;
                2: return -1;
                3: return 0;
                4: return 1;
                default: return 127;
            endcase
        end
        r = $urandom_range(0, 255);
        return r - 128;
    endfunction

    // ---------------- Tests ----------------
    //            rst en clr    a     b   av bv
    initial begin
        // R1: reset clears acc and valids
        step(1, 0, 0,    0,    0, 0, 0);
        // R7: reset wins over a valid MAC
        step(1, 1, 0,    5,    7, 1, 1);

        // R2: directed MACs: 2*3 + (-4)*5 = -14
        step(0, 1, 0,    2,    3, 1, 1);
        step(0, 1, 0,   -4,    5, 1, 1);

        // R3: no MAC without a_valid, b_valid or step_en
        step(0, 1, 0,    9,    9, 0, 1);
        step(0, 1, 0,    9,    9, 1, 0);
        step(0, 0, 0,    9,    9, 1, 1);

        // R5: stall while the inputs keep changing
        repeat (5) step(0, 0, 0, pick(), pick(), 1, 1);

        // R6: clear during a stall; clear in the same cycle as a valid MAC
        step(0, 0, 1,    3,    3, 1, 1);
        step(0, 1, 0,    3,    3, 1, 1);
        step(0, 1, 1,    3,    3, 1, 1);

        // R8: signed extremes
        step(0, 1, 0, -128, -128, 1, 1);
        step(0, 1, 0, -128,  127, 1, 1);
        step(0, 1, 0,  127, -128, 1, 1);
        step(0, 1, 0,  127,  127, 1, 1);
        step(0, 1, 0,   -1,   -1, 1, 1);
        step(0, 1, 0,    0, -128, 1, 1);

        // R7: reset in the middle of a tile, together with clear and a valid MAC
        step(0, 1, 0,  100,  100, 1, 1);
        step(1, 1, 1,  100,  100, 1, 1);

        // R9: the MiniLM bound K = 1536 does not overflow ...
        step(0, 0, 1,    0,    0, 0, 0);
        repeat (1536) step(0, 1, 0, -128, -128, 1, 1);
        if (acc_out !== 32'sd25165824)
            fail($sformatf("K = 1536 bound: acc_out = %0d", acc_out));
        // ... and 2^17 products of 2^14 wrap to -2^31
        repeat (131072 - 1536) step(0, 1, 0, -128, -128, 1, 1);
        if (acc_out !== 32'sh80000000)
            fail($sformatf("wrap: acc_out = %0d", acc_out));

        // All requirements: random cycles with rare reset and clear
        repeat (20000)
            step($urandom_range(0, 999) == 0,   // 0.1% reset
                 $urandom_range(0, 9) < 8,      // 80% step_en
                 $urandom_range(0, 49) == 0,    // 2% clear
                 pick(), pick(),
                 $urandom_range(0, 9) < 7,      // 70% a_valid
                 $urandom_range(0, 9) < 7);     // 70% b_valid

        if (errors == 0) $display("TEST PASSED: pe");
        else             $display("TEST FAILED: pe, %0d errors", errors);
        $finish;
    end
endmodule
