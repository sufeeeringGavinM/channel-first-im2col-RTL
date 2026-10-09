// One output channel's kernel memory with synchronous reads.
`timescale 1ns / 1ps
module kernel_bank #(parameter BIT_WIDTH = 8, ADDR_WIDTH = 8) (
    input clk, we, read_enable, read_active,
    input [ADDR_WIDTH-1:0] waddr, raddr,
    input [BIT_WIDTH-1:0] wdata,
    output reg [BIT_WIDTH-1:0] rdata
);
  reg [BIT_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];
  always @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    if (read_enable) rdata <= read_active ? mem[raddr] : {BIT_WIDTH{1'b0}};
  end
endmodule
