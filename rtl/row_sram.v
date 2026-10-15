// SRAM model with one write port and one synchronous read port.


`timescale 1ns / 1ps

module row_sram #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 8
) (
    input                  clk,
    input                  we,
    input [ADDR_WIDTH-1:0] waddr,
    input [ DATA_WIDTH-1:0] wdata,
    input [ADDR_WIDTH-1:0] raddr,

    output reg [DATA_WIDTH-1:0] rdata
);
  localparam DEPTH = 1 << ADDR_WIDTH;

  reg [DATA_WIDTH-1:0] mem[0:DEPTH-1];

  always @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    rdata <= mem[raddr];
  end

endmodule
