#!/usr/bin/env bash
# Compile a testbench with the Vivado simulator (xsim) and open its waveform.
#
#   hw/sim/xsim_wave.sh [systolic_array_tb|pe_tb|testbench] [--batch]
#
# Default: systolic_array_tb, opened in the Vivado GUI with the main signals
# already added. --batch runs without GUI and only writes the .wdb, which can be
# opened later with:  xsim --gui hw/sim/xsim/<tb>.wdb
set -euo pipefail

TB=${1:-systolic_array_tb}
MODE=${2:-}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HW=$(cd "$HERE/.." && pwd)
VIVADO_BIN=${VIVADO_BIN:-$(dirname "$(command -v xvlog 2>/dev/null || echo "$HOME/Source/2025.1/Vivado/bin/xvlog")")}

case $TB in
    systolic_array_tb) SRC=("$HW/rtl/compute/pe.sv" "$HW/rtl/compute/systolic_array.sv" "$HW/tb/systolic_array_tb.sv") ;;
    pe_tb)             SRC=("$HW/rtl/compute/pe.sv" "$HW/tb/pe_tb.sv") ;;
    testbench)         SRC=("$HW/rtl/compute/pe.sv" "$HW/rtl/compute/systolic_array.sv" "$HW/tb/testbench.sv") ;;
    *) echo "Unknown testbench '$TB'" >&2; exit 1 ;;
esac

WORK="$HERE/xsim"
mkdir -p "$WORK"
cd "$WORK"

"$VIVADO_BIN/xvlog" -sv "${SRC[@]}" > xvlog_$TB.log
"$VIVADO_BIN/xelab" -debug typical "$TB" -s "$TB" > xelab_$TB.log

if [[ $MODE == --batch ]]; then
    cat > "run_$TB.tcl" <<TCL
log_wave -recursive *
run all
quit
TCL
    "$VIVADO_BIN/xsim" "$TB" -wdb "$TB.wdb" -tclbatch "run_$TB.tcl" -log "xsim_$TB.log"
else
    "$VIVADO_BIN/xsim" "$TB" -gui -wdb "$TB.wdb" -tclbatch "$HERE/wave_$TB.tcl" -log "xsim_$TB.log" &
    echo "Opened xsim GUI for $TB (log: $WORK/xsim_$TB.log)"
fi
