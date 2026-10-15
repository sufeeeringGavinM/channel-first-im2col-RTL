// Address and data generation for per-tap output writeback.
`timescale 1ns / 1ps
module output_accumulator #(
    parameter CHANNEL_INDEX = 0, ACC_WIDTH = 32, ADDR_WIDTH = 8,
              WB_CNT_WIDTH = 8, OUT_LATENCY = 5, CFG_WIDTH = 32
) (
    input clk, clear, writeback_active, stream_enable, accumulate,
    input [CFG_WIDTH-1:0] output_pixels,
    input [ACC_WIDTH-1:0] old_out, result_in,
    output wb_valid,
    output reg [ADDR_WIDTH-1:0] wb_ptr,
    output [ADDR_WIDTH-1:0] wb_ptr_next,
    output [ACC_WIDTH-1:0] wb_data
);
  localparam WB_START_CYCLE = CHANNEL_INDEX + OUT_LATENCY;
  reg [WB_CNT_WIDTH-1:0] writeback_cnt;
  always @(posedge clk) begin
    if (clear) writeback_cnt <= 0;
    else writeback_cnt <= writeback_cnt + 1'b1;
  end

  // The output SRAM starts at address zero.
  assign wb_valid = writeback_active && stream_enable && (writeback_cnt >= WB_START_CYCLE) &&
                  (writeback_cnt - WB_START_CYCLE < output_pixels);

  // Advance the output address on each enabled writeback cycle.
  always @(posedge clk) begin
    if (clear) wb_ptr <= 0;
    else if (wb_valid) wb_ptr <= wb_ptr + 1'b1;
  end

  // Across-taps accumulation: read-modify-write on the output plane.
  // The SRAM has one-cycle read latency, so we read the address wb_ptr will have on the next cycle;
  // the old value then shows up on old_out in the same cycle result_in is written.
  // Each output address is written once per pass and read the cycle before, so there is
  // no read-after-write hazard within a pass.
  assign wb_ptr_next = clear ? {ADDR_WIDTH{1'b0}} : (wb_valid ? wb_ptr + 1'b1 : wb_ptr);

  assign wb_data = accumulate ? old_out + result_in : result_in;
endmodule
