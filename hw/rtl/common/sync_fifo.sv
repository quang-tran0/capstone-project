`timescale 1ns/1ps

// Single-clock FIFO. Default capacity: 2 x 256 bits = 64 bytes.
// Show-ahead read: rd_data is the oldest word whenever empty == 0.
// Sample it at the rising edge with rd_en && !empty to remove that word.
// Writes when full are ignored UNLESS a read is accepted on the same edge.
// Reads when empty are ignored, even if a write occurs on that edge (no bypass).
// Async memory read suits a small LUT-RAM FIFO; a large BRAM FIFO needs a
// synchronous read/prefetch implementation. Memory contents are not reset.
module sync_fifo #(
    parameter int DATA_WIDTH = 256,
    parameter int DEPTH = 2,
    localparam int PTR_WIDTH = (DEPTH > 1) ? $clog2(DEPTH) : 1,
    localparam int LEVEL_WIDTH = (DEPTH > 1) ? $clog2(DEPTH + 1) : 1
) (
    input  logic clk,
    input  logic rst_n,                 // Synchronous active-low reset.
    input  logic wr_en,
    input  logic [DATA_WIDTH-1:0] wr_data,
    input  logic rd_en,
    output wire [DATA_WIDTH-1:0] rd_data, // Undefined when empty.
    output wire full,
    output wire empty,
    output logic [LEVEL_WIDTH-1:0] level
);
    logic [DATA_WIDTH-1:0] mem [DEPTH];
    logic [PTR_WIDTH-1:0] wr_ptr, rd_ptr;
    wire push, pop;

    assign empty = (level == 0);
    assign full = (level == LEVEL_WIDTH'(DEPTH));
    assign pop = rd_en && !empty;
    assign push = wr_en && (!full || pop);
    assign rd_data = mem[rd_ptr];

    // No reset on the storage array, allowing distributed RAM inference.
    always_ff @(posedge clk) begin
        if (rst_n && push)
            mem[wr_ptr] <= wr_data;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
            level <= '0;
        end else begin
            if (push)
                wr_ptr <= (wr_ptr == PTR_WIDTH'(DEPTH-1)) ? '0 : wr_ptr + 1'b1;
            if (pop)
                rd_ptr <= (rd_ptr == PTR_WIDTH'(DEPTH-1)) ? '0 : rd_ptr + 1'b1;
            case ({push, pop})
                2'b10: level <= level + 1'b1;
                2'b01: level <= level - 1'b1;
                default: ; // Simultaneous read/write keeps the occupancy.
            endcase
        end
    end

    // synthesis translate_off
    initial begin
        if (DATA_WIDTH < 1 || DEPTH < 1)
            $fatal(1, "sync_fifo requires positive DATA_WIDTH and DEPTH");
    end
    // synthesis translate_on
endmodule
