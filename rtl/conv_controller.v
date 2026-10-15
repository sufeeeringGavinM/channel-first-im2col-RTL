// Tap sequencing, pass duration, and active PE weight-buffer selection.
`timescale 1ns / 1ps
module conv_controller #(parameter ARRAY_SIZE = 4, CFG_WIDTH = 32) (
    input clk, nrst, start, initial_done,
    input [CFG_WIDTH-1:0] N, P, Q, R, S,
    output busy,
    output reg done,
    output clear_pass, prime, initial_load, running, preload_next, restart_weights, accumulate,
    output reg active_weight_buffer,
    output reg [CFG_WIDTH-1:0] r, s,
    output [CFG_WIDTH-1:0] next_r, next_s
);
  localparam IDLE = 3'd0, PRIME = 3'd1, LOAD_WEIGHTS = 3'd2, RUN = 3'd3, SWITCH_TAP = 3'd4;
  reg [2:0] state;
  reg [CFG_WIDTH:0] run_count;
  wire [CFG_WIDTH:0] run_cycles = (CFG_WIDTH+1)'(N) * P * Q + 2*ARRAY_SIZE;
  assign busy = state != IDLE;
  assign clear_pass = !nrst || state != RUN;
  wire has_next_tap = r != R-1 || s != S-1;
  assign next_r = s == S-1 ? r + 1'b1 : r;
  assign next_s = s == S-1 ? {CFG_WIDTH{1'b0}} : s + 1'b1;
  assign preload_next = state == RUN && has_next_tap;
  assign prime = state == PRIME;
  assign initial_load = state == LOAD_WEIGHTS;
  assign running = state == RUN;
  assign accumulate = r != 0 || s != 0;
  assign restart_weights = (state == IDLE && start) ||
                           (state == RUN && run_count == run_cycles-1 && has_next_tap);

  always @(posedge clk) begin
    if (!nrst) begin
      state <= IDLE;
      done <= 0;
      active_weight_buffer <= 0;
      r <= 0; s <= 0;
      run_count <= 0;
    end else begin
      done <= 0;
      case (state)
        IDLE: if (start) begin
          active_weight_buffer <= 0;
          r <= 0; s <= 0;
          run_count <= 0;
          state <= PRIME;
        end
        PRIME: state <= LOAD_WEIGHTS;
        LOAD_WEIGHTS: begin
          if (initial_done) begin
            run_count <= 0;
            state <= RUN;
          end
        end
        SWITCH_TAP: begin
          // One cleared cycle resets the stream and prefetches pixel zero for the new tap.
          run_count <= 0;
          state <= RUN;
        end
        RUN: begin
          if (run_count == run_cycles-1) begin
            // Every logical output is committed before changing tap.
            run_count <= 0;
            if (s == S-1 && r == R-1) begin
              state <= IDLE;
              done <= 1;
            end else begin
              // The next weights are ready before the current physical array finishes draining.
              r <= next_r;
              s <= next_s;
              active_weight_buffer <= !active_weight_buffer;
              state <= SWITCH_TAP;
            end
          end else run_count <= run_count + 1'b1;
        end
        default: state <= IDLE;
      endcase
    end
  end
endmodule
