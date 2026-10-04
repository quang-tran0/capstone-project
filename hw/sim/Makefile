.DEFAULT_GOAL := all
HW_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
BUILD_DIR ?= $(HW_DIR)/../temp/rtl-tests
IVERILOG ?= iverilog
VVP ?= vvp
PE := $(HW_DIR)/rtl/compute/pe.sv
ARRAY := $(HW_DIR)/rtl/compute/systolic_array.sv

VLIB ?= vlib
VLOG ?= vlog
VSIM ?= vsim

.PHONY: all build run wave clean
all: run

build:
	@test -d work || $(VLIB) work
	$(VLOG) -sv -work work $(PE) $(ARRAY) $(HW_DIR)/tb/testbench.sv

run: build
	$(VSIM) -c -onfinish exit -voptargs=+acc -l run.log -wlf vsim.wlf \
		-do 'log -r /*; if {[catch {run -all}]} {quit -code 1}; quit -f' work.testbench

wave:
	$(VSIM) -view vsim.wlf -do 'add wave -r /*'

clean:
	rm -rf work
	rm -f transcript run.log vsim.wlf

# RTL regression: Icarus for PE/array, QuestaSim for FIFO. Run make test.
.PHONY: test test-pe test-array test-fifo
test: test-pe test-array test-fifo

$(BUILD_DIR):
	mkdir -p "$@"

test-pe: | $(BUILD_DIR)
	$(IVERILOG) -g2012 -Wall -s pe_tb -o "$(BUILD_DIR)/pe.vvp" $(PE) $(HW_DIR)/tb/pe_tb.sv
	$(VVP) "$(BUILD_DIR)/pe.vvp"

# Same tests run at several sizes, including the required 16x16 array.
test-array: | $(BUILD_DIR)
	@set -e; for shape in 1x1 2x2 3x5 16x16; do \
		rows=$${shape%x*}; cols=$${shape#*x}; \
		$(IVERILOG) -g2012 -Wall -s systolic_array_tb \
			-P systolic_array_tb.ROWS=$$rows -P systolic_array_tb.COLS=$$cols \
			-o "$(BUILD_DIR)/array-$$shape.vvp" $(PE) $(ARRAY) $(HW_DIR)/tb/systolic_array_tb.sv; \
		$(VVP) "$(BUILD_DIR)/array-$$shape.vvp"; \
	done

# FIFO checks: one entry, non-power-of-two depth, and stream configurations.
test-fifo:
	@mkdir -p log
	@test -d work || $(VLIB) work
	$(VLOG) -sv -work work $(HW_DIR)/rtl/common/sync_fifo.sv $(HW_DIR)/tb/sync_fifo_tb.sv
	@set -e; for shape in 1x1 8x3 128x16 256x2; do \
		width=$${shape%x*}; depth=$${shape#*x}; \
		$(VSIM) -c -onfinish exit -gDATA_WIDTH=$$width -gDEPTH=$$depth \
			-l log/fifo-$$shape.log -wlf log/fifo-$$shape.wlf \
			-do 'if {[catch {run -all}]} {quit -code 1}; quit -f' work.sync_fifo_tb; \
		grep -q 'TEST PASSED: sync_fifo' log/fifo-$$shape.log; \
	done
