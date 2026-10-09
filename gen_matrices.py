"""Generate signed convolution fixtures in (N,C,H,W), (K,C,R,S), (N,K,P,Q) order."""
import argparse
from pathlib import Path
import random

from conv_golden import conv2d, conv_output_size

# C and K must each fit within ARRAY_SIZE; all N images share one kernel.
N, C, H, W = 1, 4, 6, 6
K, R, S = 2, 3, 3
pad_i, pad_j = 0, 0
stride_i, stride_j = 1,1
dilation_i, dilation_j = 1, 1
BIT_WIDTH = 8
ACC_WIDTH = 32


# Both command-line entry points use these names and defaults.
TENSOR_ARGUMENTS = (
    ("N", N), ("C", C), ("H", H), ("W", W),
    ("K", K), ("R", R), ("S", S),
    ("pad_i", pad_i), ("pad_j", pad_j),
    ("stride_i", stride_i), ("stride_j", stride_j),
    ("dilation_i", dilation_i), ("dilation_j", dilation_j),
)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in TENSOR_ARGUMENTS:
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "sim")
    return parser


def output_shape(config):
    if min(config.N, config.C, config.H, config.W, config.K, config.R, config.S) <= 0:
        raise ValueError("Shapes must be positive")
    if min(config.dilation_i, config.dilation_j) < 1:
        raise ValueError("Dilation must contain positive integers")
    extent_i = config.dilation_i * (config.R - 1) + 1
    extent_j = config.dilation_j * (config.S - 1) + 1
    P = conv_output_size(config.H, extent_i, config.pad_i, config.stride_i)
    Q = conv_output_size(config.W, extent_j, config.pad_j, config.stride_j)
    return P, Q


def convolve(inputs, kernels, config):
    """Use the class golden model, then wrap results to the hardware width."""
    outputs, _ = conv2d(
        inputs, (config.N, config.C, config.H, config.W),
        kernels, (config.K, config.C, config.R, config.S),
        stride=(config.stride_i, config.stride_j),
        pad=(config.pad_i, config.pad_j),
        dilation=(config.dilation_i, config.dilation_j),
    )
    return [value & ((1 << ACC_WIDTH) - 1) for value in outputs]


def write_hex_tensor(path, shape, values, bit_width):
    """Write a flattened tensor with the last coordinate varying fastest."""
    hex_digits = (bit_width + 3) // 4
    with path.open("w") as handle:
        handle.write(f"// shape={shape}, last coordinate varies fastest, hex\n")
        for start in range(0, len(values), shape[-1]):
            handle.write(" ".join(f"{v & ((1 << bit_width) - 1):0{hex_digits}x}" for v in
                                  values[start:start + shape[-1]]) + "\n")


def main():
    args = create_parser().parse_args()
    P, Q = output_shape(args)
    rng = random.Random(args.seed)
    elem_min = -(1 << (BIT_WIDTH - 1))
    elem_max = (1 << (BIT_WIDTH - 1)) - 1
    input_shape = (args.N, args.C, args.H, args.W)
    kernel_shape = (args.K, args.C, args.R, args.S)
    output_dims = (args.N, args.K, P, Q)
    inputs = [rng.randint(elem_min, elem_max)
              for _ in range(args.N * args.C * args.H * args.W)]
    kernels = [rng.randint(elem_min, elem_max)
               for _ in range(args.K * args.C * args.R * args.S)]
    outputs = convolve(inputs, kernels, args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_hex_tensor(args.out_dir / "input_tensor.txt", input_shape, inputs, BIT_WIDTH)
    write_hex_tensor(args.out_dir / "kernel_tensor.txt", kernel_shape, kernels, BIT_WIDTH)
    write_hex_tensor(args.out_dir / "output_tensor.txt", output_dims, outputs, ACC_WIDTH)
    print(f"Wrote input {input_shape}, kernel {kernel_shape}, "
          f"output {output_dims} to {args.out_dir}")


if __name__ == "__main__":
    main()
