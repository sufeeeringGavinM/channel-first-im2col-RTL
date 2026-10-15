"""Sweep valid convolution configurations through run_case.py."""
import argparse
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys

from gen_matrices import TENSOR_ARGUMENTS

PROJECT = Path(__file__).resolve().parent
NAMES = [name for name, _ in TENSOR_ARGUMENTS]


def hardware_limits():
    source = (PROJECT / 'sim/tb_conv_core.v').read_text()
    def parameter(name):
        return int(re.search(r'\bparameter\s+' + name + r'\s*=\s*(\d+)\s*;', source)[1])
    return parameter('ARRAY_SIZE'), 1 << parameter('ADDR_WIDTH'), 1 << parameter('KERNEL_ADDR_WIDTH')


def output_shape(c):
    P = (c['H'] + 2*c['pad_i'] - c['dilation_i']*(c['R']-1) - 1) // c['stride_i'] + 1
    Q = (c['W'] + 2*c['pad_j'] - c['dilation_j']*(c['S']-1) - 1) // c['stride_j'] + 1
    return P, Q


def fits(c, limits):
    array, words, kernel_words = limits
    P, Q = output_shape(c)
    return (P > 0 and Q > 0 and c['C'] <= array and c['K'] <= array
            and c['N']*c['H']*c['W'] <= words and c['N']*P*Q <= words
            and c['C']*c['R']*c['S'] <= kernel_words)


def make_cases(count, seed, random_only=False):
    limits = hardware_limits()
    defaults = dict(TENSOR_ARGUMENTS)
    # Batches, independent axes, long kernels, idle PEs and memory boundaries.
    directed = [defaults, *[dict(zip(NAMES, values)) for values in [
        (2,2,5,6,3,2,3,1,2,2,1,2,1),
        (2,1,8,8,2,1,1,0,0,1,1,1,1),
        (1,1,17,2,4,17,1,0,0,1,1,1,1),
        (3,3,3,4,2,1,2,0,1,1,2,1,2),
        (4,1,1,1,1,1,1,0,0,1,1,1,1),
        (1,1,1,1,1,3,3,1,1,1,1,1,1),
        (1,1,8,16,2,1,1,0,0,1,1,1,1),
        # Input and output arrays can each use their entire capacity.
        (2,2,8,16,3,1,1,0,0,1,1,1,1),
        (1,1,1,1,2,2,2,8,8,1,1,1,1),
        (1,4,8,8,2,8,8,0,0,1,1,1,1),
        (1,10,8,8,10,5,5,0,0,1,1,1,1),
        (1,1,6,7,10,2,3,1,2,2,3,2,1),
        (2,10,4,5,1,2,2,1,0,1,2,1,2),
        (1,2,3,7,3,1,3,0,2,1,2,1,3),
        (1,2,7,3,3,3,1,2,0,2,1,3,1),
    ]]]
    cases = [] if random_only else [c for c in directed if fits(c, limits)]
    rng = random.Random(seed)
    for _ in range(count):
        while True:
            c = dict(zip(NAMES, [rng.randint(1,4), rng.randint(1,limits[0]),
                rng.randint(1,12), rng.randint(1,12), rng.randint(1,limits[0]),
                rng.randint(1,5), rng.randint(1,5), rng.randint(0,4), rng.randint(0,4),
                rng.randint(1,4), rng.randint(1,4), rng.randint(1,3), rng.randint(1,3)]))
            if fits(c, limits):
                cases.append(c)
                break
    return cases


def read_tensor(path):
    return [int(word, 16) for line in path.read_text().splitlines()
            for word in line.split('//', 1)[0].split()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=100, help='Random cases in addition to directed cases')
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--timeout', type=float, default=120, help='Maximum seconds per case')
    parser.add_argument('--random-only', action='store_true')
    parser.add_argument('--limit', type=int, help='Run only the first few cases')
    parser.add_argument('--out-dir', type=Path, default=PROJECT / 'sim/case_sweep')
    args = parser.parse_args()
    cases = make_cases(args.cases, args.seed, args.random_only)
    if args.limit is not None:
        cases = cases[:args.limit]
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    results = []
    total_outputs = 0
    array = hardware_limits()[0]
    for index, config in enumerate(cases, 1):
        folder = output_dir / f'case_{index:04d}'
        folder.mkdir(exist_ok=True)
        case_seed = args.seed + index
        (folder / 'config.json').write_text(json.dumps({**config, 'seed': case_seed}, indent=2) + '\n')
        command = [sys.executable, str(PROJECT / 'run_case.py'), '--seed', str(case_seed), '--out-dir', str(folder)]
        for name in NAMES:
            command.extend([f'--{name}', str(config[name])])
        try:
            with (folder / 'run.log').open('w') as log:
                with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True) as process:
                    try:
                        status = process.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        raise
                    if status:
                        raise subprocess.CalledProcessError(status, command)
            expected = read_tensor(folder / 'output_tensor.txt')
            actual = read_tensor(folder / 'output_tensor_core.txt')
            P, Q = output_shape(config)
            if actual != expected or len(actual) != config['N']*config['K']*P*Q:
                raise ValueError('Simulated output differs from Python golden tensor')
            text = (folder / 'run.log').read_text()
            cycles = int(re.search(r'Run cycles .*?: (\d+)', text)[1])
            utilization = re.search(
                r'Useful MAC utilization \(channels only\): ([\d.]+)% \((\d+)/(\d+) PEs', text)
            utilization_percent = float(utilization[1])
            useful_pes, total_pes = int(utilization[2]), int(utilization[3])
            taps = config['R']*config['S']
            if cycles != 1 + array + taps*(config['N']*P*Q + 2*array) + taps-1:
                raise ValueError('Unexpected cycle count')
            if not (folder / 'sram_layout.txt').is_file():
                raise ValueError('SRAM report missing')
            results.append({'case': index, 'config': config, 'outputs': len(actual),
                            'cycles': cycles, 'useful_pes': useful_pes, 'total_pes': total_pes,
                            'channel_utilization_percent': utilization_percent})
            total_outputs += len(actual)
            print(f'PASS {index}/{len(cases)}: N={config["N"]} C={config["C"]} K={config["K"]}, '
                  f'{len(actual)} outputs, {cycles} cycles, '
                  f'utilization {utilization_percent:.2f}% ({useful_pes}/{total_pes} PEs)', flush=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, ValueError, TypeError) as error:
            failures.append({'case': index, 'error': str(error), 'directory': str(folder)})
            print(f'FAIL {index}/{len(cases)}: {error}; see {folder / "run.log"}', flush=True)
    summary = {'cases': len(cases), 'passed': len(cases)-len(failures),
               'outputs_checked': total_outputs, 'results': results, 'failures': failures}
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'{summary["passed"]}/{len(cases)} passed; {total_outputs} outputs checked. Results: {output_dir}')
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
