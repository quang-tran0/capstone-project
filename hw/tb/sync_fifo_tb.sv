`timescale 1ns/1ps

module sync_fifo_tb;
    parameter int DATA_WIDTH = 256;
    parameter int DEPTH = 2;
    localparam int LEVEL_WIDTH = (DEPTH > 1) ? $clog2(DEPTH + 1) : 1;

    logic clk = 0;
    logic rst_n = 0;
    logic wr_en = 0, rd_en = 0;
    logic [DATA_WIDTH-1:0] wr_data = '0;
    wire [DATA_WIDTH-1:0] rd_data;
    wire full, empty;
    wire [LEVEL_WIDTH-1:0] level;
    logic [DATA_WIDTH-1:0] reference_queue [$];
    logic [31:0] random_state = 32'h1badf00d;
    int cycles_checked = 0;

    sync_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(DEPTH)) dut (.*);
    always #5 clk = ~clk;

    // Independent queue checks ordering, flags, wraparound and rejected requests.
    task automatic check_state;
        if (int'(level) != reference_queue.size() ||
            empty !== (reference_queue.size() == 0) ||
            full !== (reference_queue.size() == DEPTH))
            $fatal(1, "Flags/count mismatch: level=%0d expected=%0d full=%b empty=%b",
                   level, reference_queue.size(), full, empty);
        if (reference_queue.size() != 0 && rd_data !== reference_queue[0])
            $fatal(1, "Head mismatch: got %h expected %h", rd_data, reference_queue[0]);
    endtask

    task automatic cycle(input bit write_req, read_req,
                         input logic [DATA_WIDTH-1:0] data,
                         input bit reset_n = 1);
        logic [DATA_WIDTH-1:0] removed;
        @(negedge clk);
        rst_n = reset_n;
        wr_en = write_req;
        rd_en = read_req;
        wr_data = data;
        // The old head is consumed before this edge's write is appended.
        @(posedge clk);
        if (!reset_n) begin
            reference_queue.delete();
        end else begin
            check_state();
            if (read_req && reference_queue.size() != 0)
                removed = reference_queue.pop_front();
            if (write_req && reference_queue.size() < DEPTH)
                reference_queue.push_back(data);
        end
        #1;
        check_state();
        cycles_checked++;
    endtask

    function automatic logic [31:0] next_random;
        random_state = random_state * 32'd1664525 + 32'd1013904223;
        return random_state;
    endfunction

    initial begin
        logic [DATA_WIDTH-1:0] data;
        logic [31:0] requests;
        cycle(1, 1, '1, 0);             // Reset wins over both requests.
        cycle(0, 1, '0);                // Empty read is ignored.
        cycle(1, 1, '1);                // Empty: write retained, no bypass pop.
        repeat (4) cycle(0, 0, '0);     // Unconsumed head remains stable.
        cycle(0, 1, '0);

        for (int i = 0; i < DEPTH; i++)
            cycle(1, 0, DATA_WIDTH'(i + 1));
        cycle(1, 0, '1);                // Full write must not overwrite data.
        // Full simultaneous pop/push sustains one item/cycle across pointer wraps.
        for (int i = 0; i < 3*DEPTH; i++)
            cycle(1, 1, DATA_WIDTH'(i + 100));
        repeat (DEPTH) cycle(0, 1, '0);
        cycle(0, 1, '0);

        cycle(1, 0, '1);
        cycle(1, 1, '0, 0);             // Reset discards pending contents.
        cycle(0, 1, '0);
        for (int i = 0; i < 1000; i++) begin
            requests = next_random();
            // Exercise every data bit, including the high lanes of wide words.
            for (int b = 0; b < DATA_WIDTH; b++) begin
                random_state = next_random();
                data[b] = random_state[31];
            end
            cycle(requests[30], requests[31], data, i % 97 != 0);
        end
        repeat (DEPTH) cycle(0, 1, '0);
        $display("TEST PASSED: sync_fifo width=%0d depth=%0d cycles=%0d",
                 DATA_WIDTH, DEPTH, cycles_checked);
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "Timeout");
    end
endmodule
