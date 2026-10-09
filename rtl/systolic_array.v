// PE grid, stationary weight buffers, and registered column outputs.
`timescale 1ns / 1ps
module systolic_array #(
    parameter ARRAY_SIZE = 4, BIT_WIDTH = 8, ACC_WIDTH = 32, CFG_WIDTH = 32
) (
    input clk, nrst, clear_pass, load_weights,
    input weight_write_buffer, weight_read_buffer,
    input [CFG_WIDTH-1:0] cfg_C, cfg_K,
    input [ARRAY_SIZE-1:0][BIT_WIDTH-1:0] activation_rows, weight_row_in,
    output reg [ARRAY_SIZE-1:0][ACC_WIDTH-1:0] acc_col_out
);
  // Each PE connection is a separate word-sized net.
  wire [BIT_WIDTH-1:0] mac_data_in [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1], mac_data_out [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];     // Buses between columns
  wire [BIT_WIDTH-1:0] mac_weight_in [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1], mac_weight_out [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]; // Buses between rows
  wire [ACC_WIDTH-1:0] mac_acc_in [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1], mac_acc_out [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];       // Buses between rows

  genvar c, k;
  generate
    for (c = 0; c < ARRAY_SIZE; c = c + 1) begin : g_c
      for (k = 0; k < ARRAY_SIZE; k = k + 1) begin : g_k

        // Column k=0 receives the SRAM activation; later columns receive it from the left PE.
        if (k == 0) begin
          assign mac_data_in[c][k] = activation_rows[c];
        end else begin
          assign mac_data_in[c][k] = mac_data_out[c][k-1];
        end

        // Input channel c=0 starts the sum; later channels add the preceding PE result.
        if (c == 0) begin
          assign mac_acc_in[c][k]    = 0;
          assign mac_weight_in[c][k] = (load_weights && k < cfg_K) ? weight_row_in[k] : 0;
        end else begin
          assign mac_acc_in[c][k]    = mac_acc_out[c-1][k];
          assign mac_weight_in[c][k] = (load_weights && c < cfg_C && k < cfg_K) ? mac_weight_out[c-1][k] : 0;
        end

        MAC #(
            .BIT_WIDTH(BIT_WIDTH),
            .ACC_WIDTH(ACC_WIDTH)
        ) u_MAC (
            .nrst        (nrst),
            .clk         (clk),
            .clear_acc(clear_pass),
            .load_weights(load_weights),
            .weight_write_buffer(weight_write_buffer),
            .weight_read_buffer(weight_read_buffer),
            .data_in     (mac_data_in[c][k]),
            .weight_in   (mac_weight_in[c][k]),
            .acc_in      (mac_acc_in[c][k]),
            .data_out    (mac_data_out[c][k]),
            .weight_out  (mac_weight_out[c][k]),
            .acc_out     (mac_acc_out[c][k])
        );

      end
    end
  endgenerate

  generate
    for (k = 0; k < ARRAY_SIZE; k = k + 1) begin : g_output
      always @(posedge clk) begin
        if (!nrst) acc_col_out[k] <= 0;
        else acc_col_out[k] <= mac_acc_out[ARRAY_SIZE-1][k];
      end
    end
  endgenerate

endmodule
