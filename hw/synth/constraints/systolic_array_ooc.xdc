# Out-of-context timing constraints for systolic_array (wrapped by systolic_array_ooc).
# CLK_PERIOD_NS is set by run_ooc.tcl; the target clock is 200 MHz (5.000 ns).
create_clock -name clk -period $::CLK_PERIOD_NS [get_ports clk]
set_property HD.CLK_SRC BUFGCE_X0Y0 [get_ports clk]
