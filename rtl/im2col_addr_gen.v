`timescale 1ns / 1ps

// Combinational input SRAM address generation for a convolution tap.
// Shapes: input (N,C,H,W), kernel (K,C,R,S), output (N,K,P,Q).
// position = (p*Q+q)*N+n: images vary fastest within each output position.
// addr = (h*W+w)*N+n within the SRAM array for input channel c.
module im2col_addr_gen #(
    parameter CFG_WIDTH = 32,
    parameter POSITION_WIDTH  = 8,
    parameter ADDR_WIDTH = 8
) (
    input [CFG_WIDTH-1:0] cfg_N,
    input [CFG_WIDTH-1:0] cfg_H,
    input [CFG_WIDTH-1:0] cfg_W,
    input [CFG_WIDTH-1:0] cfg_Q,
    input [CFG_WIDTH-1:0] cfg_pad_i,
    input [CFG_WIDTH-1:0] cfg_pad_j,
    input [CFG_WIDTH-1:0] cfg_stride_i,
    input [CFG_WIDTH-1:0] cfg_stride_j,
    input [CFG_WIDTH-1:0] cfg_dilation_i,
    input [CFG_WIDTH-1:0] cfg_dilation_j,
    input  [POSITION_WIDTH-1:0] position,
    input  [CFG_WIDTH-1:0]  r,
    input  [CFG_WIDTH-1:0]  s,
    output [ADDR_WIDTH-1:0] addr,
    output                 valid
);

  // Decode image and spatial coordinates from the interleaved stream.
  wire [POSITION_WIDTH-1:0] n = position % cfg_N;
  wire [POSITION_WIDTH-1:0] pix = position / cfg_N;
  wire [POSITION_WIDTH-1:0] p = pix / cfg_Q;
  wire [POSITION_WIDTH-1:0] q = pix % cfg_Q;

  // Zero-extend unsigned coordinates before signed arithmetic. Negative h/w
  // represent padding above/left of the image. Wide intermediates preserve
  // the products for configurable pixel widths and convolution geometry.
  wire signed [63:0] p_signed = $signed({1'b0, p});
  wire signed [63:0] q_signed = $signed({1'b0, q});
  wire signed [63:0] r_signed = $signed({1'b0, r});
  wire signed [63:0] s_signed = $signed({1'b0, s});
  wire signed [63:0] h = p_signed * cfg_stride_i + r_signed * cfg_dilation_i - cfg_pad_i;
  wire signed [63:0] w = q_signed * cfg_stride_j + s_signed * cfg_dilation_j - cfg_pad_j;

  // The caller bounds position and tap coordinates; padding checks use h/w.
  assign valid = (h >= 0) && (h < cfg_H) && (w >= 0) && (w < cfg_W);
  wire signed [63:0] flat = h * cfg_W + w;
  assign addr = valid ? ADDR_WIDTH'(flat * cfg_N + n) : {ADDR_WIDTH{1'b0}};
endmodule
