`timescale 1ns / 1ps

// Runtime configuration capture and wiring for the convolution controller and datapath.
// Images share weights and stream in spatial-position-major, image-minor order.
// Configuration is captured on start and remains fixed throughout the run.
module conv_core #(
    parameter ARRAY_SIZE = 4,
    parameter BIT_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter ADDR_WIDTH = 8,
    parameter KERNEL_ADDR_WIDTH = 8,
    parameter CFG_WIDTH = 32
) (
    input clk,
    input nrst,
    input start,
    output busy,
    output done,
    input [CFG_WIDTH-1:0] cfg_N,
    input [CFG_WIDTH-1:0] cfg_C,
    input [CFG_WIDTH-1:0] cfg_H,
    input [CFG_WIDTH-1:0] cfg_W,
    input [CFG_WIDTH-1:0] cfg_K,
    input [CFG_WIDTH-1:0] cfg_R,
    input [CFG_WIDTH-1:0] cfg_S,
    input [CFG_WIDTH-1:0] cfg_P,
    input [CFG_WIDTH-1:0] cfg_Q,
    input [CFG_WIDTH-1:0] cfg_pad_i,
    input [CFG_WIDTH-1:0] cfg_pad_j,
    input [CFG_WIDTH-1:0] cfg_stride_i,
    input [CFG_WIDTH-1:0] cfg_stride_j,
    input [CFG_WIDTH-1:0] cfg_dilation_i,
    input [CFG_WIDTH-1:0] cfg_dilation_j,

    input [ARRAY_SIZE-1:0] sram_we_ext,
    input [ARRAY_SIZE-1:0][ADDR_WIDTH-1:0] sram_waddr_ext,
    input [ARRAY_SIZE-1:0][BIT_WIDTH-1:0] sram_wdata_ext,
    input [ARRAY_SIZE-1:0] kernel_we_ext,
    input [ARRAY_SIZE-1:0][KERNEL_ADDR_WIDTH-1:0] kernel_waddr_ext,
    input [ARRAY_SIZE-1:0][BIT_WIDTH-1:0] kernel_wdata_ext,
    output [ARRAY_SIZE-1:0][ACC_WIDTH-1:0] acc_col_out
);
  reg [CFG_WIDTH-1:0] N, C, H, W, K, R, S, P, Q, pad_i, pad_j, stride_i, stride_j, dilation_i, dilation_j;
  wire [CFG_WIDTH-1:0] r, s, next_r, next_s;
  wire clear_pass, prime, initial_load, running, preload_next, restart_weights;
  wire accumulate, active_weight_buffer;
  wire load_weights, weight_write_buffer;
  wire [ARRAY_SIZE-1:0] input_we = sram_we_ext & {ARRAY_SIZE{nrst && !busy}};

  wire initial_done, kernel_read_enable;
  wire [CFG_WIDTH-1:0] c_read;
  wire [KERNEL_ADDR_WIDTH-1:0] kernel_raddr;
  wire [ARRAY_SIZE-1:0][BIT_WIDTH-1:0] kernel_rdata, weight_row_in;
  weight_loader #(
      .ARRAY_SIZE(ARRAY_SIZE), .BIT_WIDTH(BIT_WIDTH),
      .ADDR_WIDTH(KERNEL_ADDR_WIDTH), .CFG_WIDTH(CFG_WIDTH)
  ) u_weight_loader (
      .clk(clk), .nrst(nrst), .restart(restart_weights), .prime(prime),
      .initial_load(initial_load), .preload_next(preload_next),
      .running(running), .active_weight_buffer(active_weight_buffer),
      .R(R), .S(S), .r(r), .s(s), .next_r(next_r), .next_s(next_s),
      .kernel_rdata(kernel_rdata), .load_weights(load_weights),
      .weight_write_buffer(weight_write_buffer), .initial_done(initial_done),
      .kernel_read_enable(kernel_read_enable), .kernel_raddr(kernel_raddr),
      .c_read(c_read), .weight_row_in(weight_row_in)
  );

  genvar k;
  generate
    for (k = 0; k < ARRAY_SIZE; k = k + 1) begin : g_kernel
      kernel_bank #(.BIT_WIDTH(BIT_WIDTH), .ADDR_WIDTH(KERNEL_ADDR_WIDTH)) u_bank (
          .clk(clk), .we(nrst && !busy && kernel_we_ext[k]),
          .waddr(kernel_waddr_ext[k]), .wdata(kernel_wdata_ext[k]),
          .read_enable(nrst && kernel_read_enable), .read_active(k < K && c_read < C),
          .raddr(kernel_raddr), .rdata(kernel_rdata[k])
      );
    end
  endgenerate

  always @(posedge clk) begin
    if (!nrst) begin
      N <= 1;
      C <= 0;
      H <= 0;
      W <= 0;
      K <= 0;
      R <= 0;
      S <= 0;
      P <= 0;
      Q <= 0;
      pad_i <= 0;
      pad_j <= 0;
      stride_i <= 0;
      stride_j <= 0;
      dilation_i <= 0;
      dilation_j <= 0;
    end else if (start && !busy) begin
      N <= cfg_N;
      C <= cfg_C;
      H <= cfg_H;
      W <= cfg_W;
      K <= cfg_K;
      R <= cfg_R;
      S <= cfg_S;
      P <= cfg_P;
      Q <= cfg_Q;
      pad_i <= cfg_pad_i;
      pad_j <= cfg_pad_j;
      stride_i <= cfg_stride_i;
      stride_j <= cfg_stride_j;
      dilation_i <= cfg_dilation_i;
      dilation_j <= cfg_dilation_j;
    end
  end

  conv_controller #(.ARRAY_SIZE(ARRAY_SIZE), .CFG_WIDTH(CFG_WIDTH)) u_controller (
      .clk(clk), .nrst(nrst), .start(start), .initial_done(initial_done),
      .N(N), .P(P), .Q(Q), .R(R), .S(S), .busy(busy), .done(done),
      .clear_pass(clear_pass), .prime(prime), .initial_load(initial_load),
      .running(running), .preload_next(preload_next), .restart_weights(restart_weights),
      .accumulate(accumulate), .active_weight_buffer(active_weight_buffer),
      .r(r), .s(s), .next_r(next_r), .next_s(next_s)
  );

  MMU #(
      .ARRAY_SIZE(ARRAY_SIZE), .BIT_WIDTH(BIT_WIDTH), .ACC_WIDTH(ACC_WIDTH),
      .ADDR_WIDTH(ADDR_WIDTH), .CFG_WIDTH(CFG_WIDTH)
  ) u_mmu (
      .cfg_N(N),
      .cfg_C(C),
      .cfg_K(K),
      .cfg_H(H),
      .cfg_W(W),
      .cfg_P(P),
      .cfg_Q(Q),
      .cfg_pad_i(pad_i),
      .cfg_pad_j(pad_j),
      .cfg_stride_i(stride_i),
      .cfg_stride_j(stride_j),
      .cfg_dilation_i(dilation_i),
      .cfg_dilation_j(dilation_j),
      .r(r), .s(s), .clk(clk), .nrst(nrst),
      .clear_pass(clear_pass), .load_weights(load_weights),
      .weight_write_buffer(weight_write_buffer), .weight_read_buffer(active_weight_buffer),
      .accumulate(accumulate),
      .sram_we_ext(input_we), .sram_waddr_ext(sram_waddr_ext),
      .sram_wdata_ext(sram_wdata_ext), .weight_row_in(weight_row_in),
      .acc_col_out(acc_col_out)
  );
endmodule
