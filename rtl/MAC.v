`timescale 1ns / 1ps

// Signed PE with two stationary weights: compute from one while loading the other.
module MAC #(
    parameter BIT_WIDTH = 8,
    parameter ACC_WIDTH = 32
) (
    input clk,
    input nrst,
    input clear_acc,
    input load_weights,
    input weight_write_buffer,
    input weight_read_buffer,
    input signed [ACC_WIDTH-1:0] acc_in,
    input signed [BIT_WIDTH-1:0] data_in,
    input signed [BIT_WIDTH-1:0] weight_in,
    output reg signed [ACC_WIDTH-1:0] acc_out,
    output reg signed [BIT_WIDTH-1:0] data_out,
    output reg signed [BIT_WIDTH-1:0] weight_out
);
  reg signed [BIT_WIDTH-1:0] stored_weight [0:1];
  wire signed [ACC_WIDTH-1:0] product =
      $signed(data_in) * $signed(stored_weight[weight_read_buffer]);

  always @(posedge clk) begin
    if (!nrst) begin
      acc_out <= 0;
      data_out <= 0;
      weight_out <= 0;
      stored_weight[0] <= 0;
      stored_weight[1] <= 0;
    end else begin
      // Weight shifting is independent of activation and partial-sum advancement.
      if (load_weights) begin
        stored_weight[weight_write_buffer] <= weight_in;
        weight_out <= weight_in;
      end
      if (clear_acc) begin
        acc_out <= 0;
        data_out <= 0;
      end else begin
        data_out <= data_in;
        acc_out <= acc_in + product;
      end
    end
  end
endmodule
