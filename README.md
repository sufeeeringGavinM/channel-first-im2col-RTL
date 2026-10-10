# im2col address generator and convolution core

This project implements signed convolution (cross-correlation) using a square
systolic array. A shared im2col address generator reads input samples for each
kernel tap without storing an expanded im2col matrix. `conv_core` runs the MMU
for each tap and accumulates results in SRAM. Each PE
has two stationary weight registers: one supplies the current tap while the
other receives the next tap from the kernel SRAM.

## Logical tensor shapes

These shapes describe convolution dimensions and coordinates independently of
how values are stored in tensor files or SRAM.

| Tensor | Logical shape | Coordinates |
|---|---|---|
| Input | `(N,C,H,W)` | `(n,c,h,w)` |
| Kernel | `(K,C,R,S)` | `(k,c,r,s)` |
| Output | `(N,K,P,Q)` | `(n,k,p,q)` |

The core processes `N` images per start, all using the same kernel. Images
are interleaved within each tap. Logical channel counts satisfy
`1 <= C,K <= ARRAY_SIZE`; `N` is positive and limited by SRAM capacity.

## Runtime configuration

| Setting | Interface |
|---|---|
| Runtime dimensions | `cfg_N`, `cfg_C`, `cfg_K`, `cfg_H`, `cfg_W`, `cfg_R`, `cfg_S`, `cfg_P`, `cfg_Q` |
| Runtime padding | `cfg_pad_i`, `cfg_pad_j` |
| Runtime stride | `cfg_stride_i`, `cfg_stride_j` |
| Runtime dilation | `cfg_dilation_i`, `cfg_dilation_j` |
| Fixed hardware parameters | `ARRAY_SIZE`, `BIT_WIDTH`, `ACC_WIDTH`, `ADDR_WIDTH`, `KERNEL_ADDR_WIDTH`, `CFG_WIDTH` |

Configuration ports are unsigned `CFG_WIDTH`-bit values, defaulting to 32 bits.
The core captures all configuration into registers on the rising edge that
accepts `start` while idle. Changing the ports during a run does not affect
that run. They are direct RTL ports; the design has no register-bus interface.

`i` is the height axis; `j` is the width axis. Padding is symmetric within each
axis. Software supplies `P,Q` consistent with the geometry:

```text
extent_i = dilation_i*(R-1) + 1
extent_j = dilation_j*(S-1) + 1
P = floor((H + 2*pad_i - extent_i) / stride_i) + 1
Q = floor((W + 2*pad_j - extent_j) / stride_j) + 1
```

Dimensions, stride, and dilation are positive; padding is nonnegative. The
dilated kernel fits within the padded input. Configuration and tensor contents
must fit the fixed memory capacities. Kernel coordinates use `CFG_WIDTH` bits.

## Tensor-file and SRAM storage layouts

| Tensor | Tensor-file order | SRAM bank | Within-bank order | SRAM address |
|---|---|---|---|---|
| Input | NCHW | `c` | HWN | `(h*W+w)*N+n` |
| Kernel | KCRS | `k` | CRS | `(c*R+r)*S+s` |
| Output | NKPQ | `k` | PQN | `OUT_BASE+(p*Q+q)*N+n` |

In each storage order, the rightmost coordinate varies fastest. Tensor files
contain hexadecimal values in the logical dimension order. The testbench
rearranges inputs into the banked SRAM layout and exports outputs in NKPQ order.
Kernel-file values map to bank `k`, with `(c,r,s)` varying within that bank.

Within each input/output bank, images are interleaved with `n` varying fastest.
For `N=2`, input bank `c` starts with logical `(n,c,h,w)` coordinates
`(0,c,0,0)`, `(1,c,0,0)`, `(0,c,0,1)`, `(1,c,0,1)`, and so on.
Outputs use the same image-minor ordering, starting at `OUT_BASE=N*H*W` after
the input region. Each SRAM word contains one value.

The physical storage has fixed bank counts and depths. The logical tensor
shapes are runtime values; they determine which banks and addresses are used.

## Modules

| Module | Responsibility |
|---|---|
| `im2col_addr_gen` | Combinational input-plane address and padding validity from runtime geometry. |
| `MAC` | Registered signed multiply-accumulate with two stationary weight registers. |
| `MMU` | Shared address generation, channel skew, pass timing, and wiring banks to the PE array. |
| `channel_bank` | Owns one shared input/output SRAM and selects external or output writes. |
| `activation_feeder` | Aligns read validity and gates activations for padding and inactive rows. |
| `output_accumulator` | Output writeback timing, addresses, and accumulation across taps. |
| `systolic_array` | PE grid, weight forwarding, and registered column outputs. |
| `kernel_bank` | One output channel’s kernel SRAM and synchronous read interface. |
| `weight_loader` | Kernel read addresses and initial/background weight shifting. |
| `row_sram` | SRAM model with one write port and two synchronous read ports. |
| `conv_core` | Configuration registers, tap sequencing, start/busy/done, and module wiring. |

The hardware hierarchy is:

```text
conv_core                       Configuration and tap sequencing
├── weight_loader               Initial and background weight loading
├── kernel_bank × ARRAY_SIZE    Kernel storage
└── MMU                         Address generation, skew, and pass timing
    ├── im2col_addr_gen
    ├── channel_bank × ARRAY_SIZE
    │   ├── activation_feeder
    │   ├── output_accumulator
    │   └── row_sram             Shared input/output storage
    └── systolic_array
        └── MAC × ARRAY_SIZE²
```

Each channel bank owns one memory. Activation feeding and output accumulation
share its read ports; the bank selects external writes or result writeback.
The weight loader owns the descending row counter and prefetch/shift phases.
The top level owns tap coordinates, run timing, and active weight-buffer selection.

## Address generation and array operation

### Activation addresses

For tap `(r,s)`, the stream visits all images at each output spatial position.
The flattened stream position decodes as:

```text
n = position % N
pix = position / N
```

The spatial address generator then computes:

```text
p = pix / Q
q = pix % Q
h = p*stride_i + r*dilation_i - pad_i
w = q*stride_j + s*dilation_j - pad_j
spatial_addr = h*W + w
sram_addr = spatial_addr*N + n
```

The address generator returns `spatial_addr` within one input plane. The MMU
converts it to the interleaved SRAM address. `valid` is true when
`0 <= h < H` and `0 <= w < W`; otherwise the SRAM read address is zero and the feeder supplies
zero. The caller supplies `pix < P*Q`, `r < R`, and `s < S`; the generator
checks input bounds. Runtime division and remainder by `Q` are part of the
combinational address path.

### Kernel addresses and weight loading

`weight_loader` generates the shared kernel SRAM read address:

```text
kernel_raddr = (c_read*R + kernel_r)*S + kernel_s
```

Every `kernel_bank` instance receives the same address. Instance `k` supplies
weight `(k,c_read,kernel_r,kernel_s)` from its independent SRAM array; `k` is
selected by the instance, not encoded in the address. Initial loading uses the
current tap `(r,s)`. Background loading during RUN uses `(next_r,next_s)` and
writes the inactive PE weight registers while the current tap computes.

Weights enter the top PE row and shift downward. The loader supplies channel
coordinates in descending physical-row order, from `ARRAY_SIZE-1` to zero,
so each weight reaches its corresponding row. Kernel reads are synchronous:
one PRIME or PREFETCH cycle fetches the first row before shifting begins.
During each shift, `c_load` identifies the row being loaded, while the SRAM
read requests the following row:

```text
c_read = (load_weights && c_load != 0) ? c_load-1 : c_load
```

The returned data is available for the next clock's weight shift. Reads for
`c_read >= C` or kernel instances `k >= K` return zero. The loader performs
`ARRAY_SIZE` shifts even when fewer input channels are active. There is no
background preload after the final tap.

### Array operation and writeback

PE row `c` consumes input channel `c`; PE column `k` produces output channel
`k`. Activations move right and partial sums move down. The shared generator
prefetches for the one-cycle SRAM read latency; row `c` receives its feed
bundle delayed by `c` cycles. The bottom physical row supplies registered
column results. Rows `c >= C` receive zero activations and weights; columns
`k >= K` receive zero weights and have writeback disabled. The physical array
is `ARRAY_SIZE` by `ARRAY_SIZE`. Internal activation, weight, and partial-sum
connections are unpacked arrays of separate word-sized PE nets. Each column
has a registered output from the bottom physical row.

A direct MMU caller keeps its configuration, `(r,s)`, and `accumulate` stable
throughout a pass, loads weights across all physical rows in descending row
order, and runs the pass for `N*P*Q+2*ARRAY_SIZE` cycles. The first tap overwrites
outputs; later taps use `accumulate=1`. Writeback writes exactly `N*P*Q` words
per active column. Feed enables stop after the configured output positions,
while the physical pipeline drains. The caller controls pass duration and
asserts `clear_pass` between passes to clear counters and partial sums.
`load_weights` shifts weights into the register selected by `weight_write_buffer`;
`weight_read_buffer` selects the register used for multiplication. Loading and
computation are independent. A manual caller can hold both buffer selectors at
zero and assert `clear_pass` during loading. `conv_core` handles these steps internally.

## Memory layout and precision

There are `ARRAY_SIZE` input/output SRAM banks and `ARRAY_SIZE` kernel banks:
`2*ARRAY_SIZE` modeled memories in total. Runtime `C,K` select active channels;
they do not change the physical bank count. Each PE has two `BIT_WIDTH` stationary weight registers, totaling
`2*ARRAY_SIZE*ARRAY_SIZE` weight words. These registers are separate from the
kernel SRAM and the registered weight-forwarding chain.
These memories are RTL arrays, without instantiated physical SRAM macros.

Input/output SRAM words are `ACC_WIDTH` bits; kernel SRAM words are
`BIT_WIDTH` bits. Addresses and tensor-file orders are listed in
[Tensor-file and SRAM storage layouts](#tensor-file-and-sram-storage-layouts).

Each input/output bank has `2**ADDR_WIDTH` words and must hold `N*(H*W+P*Q)` words.
Each kernel bank has `2**KERNEL_ADDR_WIDTH` words and must hold `C*R*S` words.
Output bases follow the configured batch size and input shape. Every job uses the kernels
preloaded for its configured shape.

Activations and weights are signed `BIT_WIDTH` values. External input
writes sign-extend into SRAM; activation reads select the low `BIT_WIDTH`
bits. Products, partial sums, output writes, and cross-tap accumulation use
`ACC_WIDTH` bits and wrap at that width. Use `ACC_WIDTH >= BIT_WIDTH`.
The second input/output SRAM read port prefetches stored output values for
read-modify-write accumulation across taps.

## Running the core

1. Assert `nrst=0` across a rising clock edge, then release reset.
2. While `busy=0`, preload inputs using `sram_we_ext`, `sram_waddr_ext`,
   and `sram_wdata_ext`.
3. Preload kernels using `kernel_we_ext`, `kernel_waddr_ext`, and
   `kernel_wdata_ext`.
4. Drive all `cfg_*` ports, including the derived `P,Q`, and pulse `start`
   for one clock cycle.
5. Wait for the one-cycle `done` pulse, then read results from the output planes.

External buses have `ARRAY_SIZE` lanes. Input writes use the first `C` lanes;
kernel writes and logical outputs use the first `K` lanes. External write
lanes are usable while idle before the configuration is captured.

The controller visits `s` within `r`. Before the first tap, one PRIME cycle
fetches a synchronous kernel read, then `ARRAY_SIZE` cycles shift its weights
in descending row order into PE buffer zero. Each tap runs for
`N*P*Q+2*ARRAY_SIZE` cycles. During RUN, one read-prefetch cycle followed by
`ARRAY_SIZE` weight shifts fills the inactive PE registers with the next tap,
while activations and partial sums continue advancing with the active weights.
The remaining RUN cycles drain the current tap's physical pipeline.

After a tap finishes, the controller selects the prepared weight buffer and
spends one SWITCH_TAP cycle clearing the pipeline and prefetching the new tap's
first activation. The final tap has no next-tap preload or switch cycle.
With `T=R*S`, completion occurs this many edges after the accepted start edge:

```text
cycles = 1 + ARRAY_SIZE + T*(N*P*Q + 2*ARRAY_SIZE) + (T-1)
```

For the supplied ten-by-ten example (`N=1`, `T=9`, `P*Q=16`), this is 343 cycles.
At completion, the physical array drain has ended, all logical outputs are
committed, `busy` drops, and `done` pulses.

Starts and external input/kernel writes are ignored while busy. Holding
`start` high launches another run when idle. Synchronous active-low reset
aborts a run and clears control state while retaining memory contents. Each
run overwrites outputs on its first tap. Jobs can use different configurations
without a reset or a new hardware build. `acc_col_out` exposes live pipeline
results; final results reside in SRAM. The testbench reads SRAM through hierarchy;
the core has no external output readback port.

## Fixtures and simulation

The project has one testbench, `sim/tb_conv_core.v`, run through `run.sh`.
The Python runner supplies the same `N,C,H,W,K,R,S`, padding, stride, and
dilation arguments to the generator and testbench. Precision remains fixed in
`gen_matrices.py` and the testbench. Set physical `ARRAY_SIZE` independently,
with `ARRAY_SIZE >= max(C,K)`. The test assumes sane configuration and complete
input and kernel files.

```bash
python3 run_case.py --seed 0
```

`run_case.py` generates tensors, then runs the simulation. It shares its argument
names and defaults with `gen_matrices.py`; every geometry value is passed to
both programs. For example:

```bash
python3 run_case.py --N 2 --C 2 --H 5 --W 6 --K 3 --R 2 --S 3 \
  --pad_i 1 --pad_j 2 --stride_i 2 --stride_j 1 --dilation_i 2 --dilation_j 1
```

Both Python scripts accept `--N`, `--C`, `--H`, `--W`, `--K`, `--R`, `--S`,
`--pad_i`, `--pad_j`, `--stride_i`, `--stride_j`, `--dilation_i`, `--dilation_j`,
`--seed`, and `--out-dir`. Their defaults are declared at the top of
`gen_matrices.py`. The runner puts tensors, simulated outputs, and the SRAM
report in `--out-dir` (default `sim`). Use `--wave-file path` for a waveform.
Run either script with `--help` to list arguments.

The generator can also be used independently:

```bash
python3 gen_matrices.py --N 2 --seed 0
bash run.sh +N=2
```

The generator uses `conv_golden.py` and writes `sim/input_tensor.txt`,
`sim/kernel_tensor.txt`, and the Python reference `sim/output_tensor.txt`.
Files use NCHW/KCRS/NKPQ order and two's-complement hexadecimal values.

The testbench reads the input and kernel files, preloads interleaved input words
and kernel SRAM, drives configuration, and pulses `start`. The core controls
weight loading, tap sequencing, double buffering, and accumulation until `done`.
The testbench checks every output against its own signed convolution reference.

| Result | Destination |
|---|---|
| Start-to-done clock cycles | Terminal; excludes reset and external preloading |
| Useful PE count and percentage | Terminal; `C*K` out of `ARRAY_SIZE*ARRAY_SIZE` |
| Output tensor in NKPQ order | `sim/output_tensor_core.txt` |
| Every address in input/output and kernel SRAM banks | `sim/sram_layout.txt` |
| Output comparison result | Terminal |

The SRAM report includes hexadecimal values, signed decimal values, tensor
coordinates, unused locations, and inactive banks. Uninitialized words appear
as `x`. It shows the actual SRAM contents after completion.

Shapes and geometry can be overridden with `+N=`, `+C=`, `+K=`, `+H=`, `+W=`,
`+R=`, `+S=`, `+pad_i=`, `+pad_j=`, `+stride_i=`, `+stride_j=`,
`+dilation_i=`, and `+dilation_j=`. The testbench computes `P,Q` automatically.
Its defaults are declared at the top of the testbench. For example, with
matching tensor files:

```bash
bash run.sh +N=2
```

File paths can be overridden with `+INPUT_FILE=path`, `+KERNEL_FILE=path`,
`+OUTPUT_FILE=path`, and `+SRAM_LAYOUT_FILE=path`. A waveform is generated when
`+WAVE_FILE=path` is supplied. To reuse a compiled simulation, pass these same
arguments to `vvp sim/tb_conv_core.vvp`.

## Case sweep

`test_cases.py` runs directed boundary cases and 100 seeded random configurations
through `run_case.py`. It reads the physical array size and SRAM capacities from
the testbench and generates configurations that fit. Cases cover batches, unequal
channel counts, padding, stride, dilation, rectangular kernels, and memory limits.

```bash
python3 test_cases.py
python3 test_cases.py --cases 500 --seed 42
python3 test_cases.py --cases 0 --limit 4
```

Each case compares the simulated tensor to the generated Python golden tensor,
checks its length and cycle count, and requires an SRAM report. Results go to
`sim/case_sweep`: each case retains configuration, tensors, SRAM contents and
`run.log`. `summary.json` lists failures and checked outputs. The script continues
after failures and exits unsuccessfully if any case fails. A case times out after
120 seconds; override with `--timeout`. Use `--out-dir` to select a results folder
or `--random-only` to skip directed cases. Cases run sequentially; each invokes
the existing compilation and simulation runner.

The sweep prints each case's channel utilization from the testbench and saves
it in `summary.json` alongside cycle counts and useful/total PE counts. This is
`100*C*K/ARRAY_SIZE**2`, the fraction of PEs assigned useful channels. It
measures channel occupancy, not activity over time; loading, pipeline fill/drain,
and padded input positions do not reduce this percentage.

## TODO

1. Currently, it is assumed that input tensors and kernel tensors fit in the SRAMs.
In reality, they probably will not. So a next step is tiling. We'll have to cover:
    a. the case for when the number of channels exceed the number of PE rows,
    b. the case for when the number of kernels exceed the number of PE columns,
    c. the case for when input/output data for an input/output channel don't fit in its SRAM bank,
    d. the case for when kernels don't fit in the kernel SRAM banks
    e. proper accumulation across those cases
3. Other AI accelerator things such as:
    a. host control interface
    b. off-chip memory where entire input tensors & kernel tensors live & where final outputs go
       - need a hardware interface for this as well, a DMA.
    c. layer chaining
    d. bias, requantization, nonlinear activation, pooling, etc.

And deciding which of #2 we offload to the host

## Credits

https://github.com/kaggar11/systolic_4x4arr 
    
- Code was initially based on this. It's pretty much a Ship of Theseus thing now at this point since the only recognizable parts are the PE and the PE array buses but still worth acknowledging.

"Characterizing and Demystifying the Implicit Convolution Algorithm on Commercial Matrix-Multiplication Accelerators" by
Zhou et al.
- https://doi.org/10.48550/arXiv.2110.03901 
- Channel-first implicit im2col algorithm and TPU architecture taken from here.

# Note on AI-use

Codex and GPT-6.1 Sol was used to assist in the creation of this project.
