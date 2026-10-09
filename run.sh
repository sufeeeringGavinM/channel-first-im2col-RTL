#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
iverilog -g2012 -Wall -s tb_conv_core -o sim/tb_conv_core.vvp sim/tb_conv_core.v rtl/*.v
vvp sim/tb_conv_core.vvp "$@"
