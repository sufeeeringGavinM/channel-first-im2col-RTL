// Align padding validity with the synchronous activation read.
`timescale 1ns / 1ps
module activation_feeder #(parameter BIT_WIDTH = 8, ACC_WIDTH = 32) (
    input clk, feed_active, stream_enable, feed_en, rd_valid,
    input [ACC_WIDTH-1:0] sram_rdata,
    output [BIT_WIDTH-1:0] activation_out
);
  reg rd_valid_q;
  always @(posedge clk) rd_valid_q <= rd_valid;
  assign activation_out = (feed_active && stream_enable && feed_en && rd_valid_q)
                          ? sram_rdata[BIT_WIDTH-1:0] : {BIT_WIDTH{1'b0}};
endmodule
