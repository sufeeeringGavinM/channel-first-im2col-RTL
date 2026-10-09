// Shared input/output memory, activation feeding, and output accumulation.
// Bank index selects input channel c and output channel k independently.
// MMU supplies address/valid/feed_en delayed by CHANNEL_INDEX cycles for systolic skew.

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
    input [ADDR_WIDTH-1:0] output_base,
    input [CFG_WIDTH-1:0] output_pixels,
    input clk,
    input clear,
    input stream_enable,
    input accumulate,   // 0: writeback overwrites output (first tap). 1: output += result (later taps). Hold stable for the pass.

    input [WB_CNT_WIDTH-1:0] writeback_cnt,

    // Skewed feed signals from the shared address generator (already delayed by CHANNEL_INDEX)
    input                  feed_en,    // this row may feed
    input                  rd_valid,   // 0 when the pixel falls in the padding
    input [ADDR_WIDTH-1:0] rd_addr,    // SRAM read address (already the prefetched address)

    input                  sram_we_ext,
    input [ADDR_WIDTH-1:0] sram_waddr_ext,
    input [ BIT_WIDTH-1:0] sram_wdata_ext,

    input [ACC_WIDTH-1:0] result_in,

    output [BIT_WIDTH-1:0] activation_out
);

  wire [ACC_WIDTH-1:0] sram_rdata, old_out, wb_data;
  wire [ADDR_WIDTH-1:0] wb_ptr, wb_ptr_next;
  wire wb_valid;

  activation_feeder #(.BIT_WIDTH(BIT_WIDTH), .ACC_WIDTH(ACC_WIDTH)) u_activation (
      .clk(clk), .feed_active(feed_active), .stream_enable(stream_enable),
      .feed_en(feed_en), .rd_valid(rd_valid), .sram_rdata(sram_rdata),
      .activation_out(activation_out)
  );
  output_accumulator #(
      .CHANNEL_INDEX(CHANNEL_INDEX), .ACC_WIDTH(ACC_WIDTH), .ADDR_WIDTH(ADDR_WIDTH),
      .WB_CNT_WIDTH(WB_CNT_WIDTH), .OUT_LATENCY(OUT_LATENCY), .CFG_WIDTH(CFG_WIDTH)
  ) u_output (
      .clk(clk), .clear(clear), .writeback_active(writeback_active),
      .stream_enable(stream_enable), .accumulate(accumulate),
      .output_base(output_base), .output_pixels(output_pixels),
      .writeback_cnt(writeback_cnt), .old_out(old_out), .result_in(result_in),
      .wb_valid(wb_valid), .wb_ptr(wb_ptr), .wb_ptr_next(wb_ptr_next), .wb_data(wb_data)
  );

  wire                  sram_we = sram_we_ext | wb_valid;

  // External writes take priority over result writeback.
  wire [ADDR_WIDTH-1:0] sram_waddr = sram_we_ext ? sram_waddr_ext : wb_ptr;
  wire [ ACC_WIDTH-1:0] sram_wdata = sram_we_ext ? {{(ACC_WIDTH-BIT_WIDTH){sram_wdata_ext[BIT_WIDTH-1]}}, sram_wdata_ext} : wb_data;

  // Signed external activations are sign-extended into full-width words.
  row_sram #(
      .DATA_WIDTH(ACC_WIDTH),
      .ADDR_WIDTH(ADDR_WIDTH)
  ) u_sram (
      .clk  (clk),
      .we   (sram_we),
      .waddr(sram_waddr),
      .wdata(sram_wdata),
      .raddr(rd_addr),
      .raddr2(wb_ptr_next),
      .rdata(sram_rdata),
      .rdata2(old_out)
  );

endmodule
