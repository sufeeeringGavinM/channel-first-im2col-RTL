// Separate input/output SRAM arrays, activation feeding, and output accumulation.
// Bank index selects input channel c and output channel k independently.
// The activation feeder delays the shared request by this row's position.

`timescale 1ns / 1ps

module channel_bank #(
    parameter CHANNEL_INDEX    = 0,
    parameter BIT_WIDTH    = 8,
    parameter ACC_WIDTH    = 32,
    parameter ADDR_WIDTH   = 8,
    parameter WB_CNT_WIDTH = 8,
    parameter OUT_LATENCY  = 5,
    parameter CFG_WIDTH = 32
) (
    input feed_active,
    input writeback_active,
    input [CFG_WIDTH-1:0] output_pixels,
    input clk,
    input clear,
    input stream_enable,
    input accumulate,   // 0: writeback overwrites output (first tap). 1: output += result (later taps). Hold stable for the pass.

    // Shared request on clear; otherwise each row receives the preceding row's bundle.
    input [ADDR_WIDTH+1:0] feed_in, clear_feed,
    output [ADDR_WIDTH+1:0] feed_out,

    input                  sram_we_ext,
    input [ADDR_WIDTH-1:0] sram_waddr_ext,
    input [ BIT_WIDTH-1:0] sram_wdata_ext,

    input [ACC_WIDTH-1:0] result_in,

    output [BIT_WIDTH-1:0] activation_out
);

  wire [BIT_WIDTH-1:0] sram_rdata;
  wire [ACC_WIDTH-1:0] old_out, wb_data;
  wire [ADDR_WIDTH-1:0] wb_ptr, wb_ptr_next;
  wire wb_valid;
  wire [ADDR_WIDTH-1:0] sram_raddr;

  activation_feeder #(
      .CHANNEL_INDEX(CHANNEL_INDEX), .BIT_WIDTH(BIT_WIDTH),
      .ADDR_WIDTH(ADDR_WIDTH)
  ) u_activation (
      .clk(clk), .clear(clear), .feed_active(feed_active), .stream_enable(stream_enable),
      .feed_in(feed_in), .clear_feed(clear_feed), .feed_out(feed_out),
      .sram_raddr(sram_raddr), .sram_rdata(sram_rdata), .activation_out(activation_out)
  );
  output_accumulator #(
      .CHANNEL_INDEX(CHANNEL_INDEX), .ACC_WIDTH(ACC_WIDTH), .ADDR_WIDTH(ADDR_WIDTH),
      .WB_CNT_WIDTH(WB_CNT_WIDTH), .OUT_LATENCY(OUT_LATENCY), .CFG_WIDTH(CFG_WIDTH)
  ) u_output (
      .clk(clk), .clear(clear), .writeback_active(writeback_active),
      .stream_enable(stream_enable), .accumulate(accumulate),
      .output_pixels(output_pixels),
      .old_out(old_out), .result_in(result_in),
      .wb_valid(wb_valid), .wb_ptr(wb_ptr), .wb_ptr_next(wb_ptr_next), .wb_data(wb_data)
  );

  // External writes preload activations; computation writes only output sums.
  row_sram #(.DATA_WIDTH(BIT_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) u_input_sram (
      .clk(clk), .we(sram_we_ext), .waddr(sram_waddr_ext), .wdata(sram_wdata_ext),
      .raddr(sram_raddr), .rdata(sram_rdata)
  );
  row_sram #(.DATA_WIDTH(ACC_WIDTH), .ADDR_WIDTH(ADDR_WIDTH)) u_output_sram (
      .clk(clk), .we(wb_valid), .waddr(wb_ptr), .wdata(wb_data),
      .raddr(wb_ptr_next), .rdata(old_out)
  );
endmodule
