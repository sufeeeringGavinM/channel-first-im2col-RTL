"""Generate tensors and simulate the convolution with the same configuration."""
from pathlib import Path
import subprocess
import sys

from gen_matrices import TENSOR_ARGUMENTS, create_parser


def main():
    parser = create_parser()
    parser.description = __doc__
    parser.add_argument("--wave-file", type=Path,
                        help="Optional waveform output path")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    output_dir = args.out_dir.resolve()

    generator_args = []
    simulation_args = []
    for name, _ in TENSOR_ARGUMENTS:
        value = str(getattr(args, name))
        generator_args.extend([f"--{name}", value])
        simulation_args.append(f"+{name}={value}")

    subprocess.run(
        [sys.executable, str(project / "gen_matrices.py"), *generator_args,
         "--seed", str(args.seed), "--out-dir", str(output_dir)],
        cwd=project, check=True,
    )
    simulation_args.extend([
        f"+INPUT_FILE={output_dir / 'input_tensor.txt'}",
        f"+KERNEL_FILE={output_dir / 'kernel_tensor.txt'}",
        f"+OUTPUT_FILE={output_dir / 'output_tensor_core.txt'}",
        f"+SRAM_LAYOUT_FILE={output_dir / 'sram_layout.txt'}",
    ])
    if args.wave_file is not None:
        simulation_args.append(f"+WAVE_FILE={args.wave_file.resolve()}")
    subprocess.run(["bash", str(project / "run.sh"), *simulation_args],
                   cwd=project, check=True)


if __name__ == "__main__":
    main()
