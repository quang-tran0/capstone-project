# Out-of-context synthesis + place-and-route for one RTL block on the KV260.
#
#   vivado -mode batch -source hw/synth/run_ooc.tcl -tclargs <block> [clock_mhz]
#
# <block>: systolic_array (more blocks are added to the table below).
# Checkpoints/logs go to hw/synth/build/<block>/ (ignored by git); the reports
# required by AGENTS.md go to hw/synth/reports/<block>/:
#   timing_target.rpt   report_timing_summary (setup, hold, pulse width)
#   utilization.rpt     report_utilization (flat and hierarchical)

set script_dir [file dirname [file normalize [info script]]]
set hw_dir     [file normalize "$script_dir/.."]
set part       xck26-sfvc784-2LV-c

set block [lindex $argv 0]
set mhz   [expr {[llength $argv] > 1 ? [lindex $argv 1] : 200.0}]
set ::CLK_PERIOD_NS [format %.3f [expr {1000.0 / $mhz}]]

# block -> {top sources constraints}
set blocks [dict create \
    systolic_array [list systolic_array_ooc \
        [list $hw_dir/rtl/compute/pe.sv \
              $hw_dir/rtl/compute/systolic_array.sv \
              $script_dir/wrappers/systolic_array_ooc.sv] \
        [list $script_dir/constraints/systolic_array_ooc.xdc]] \
]
if {![dict exists $blocks $block]} {
    error "Unknown block '$block'. Known: [dict keys $blocks]"
}
lassign [dict get $blocks $block] top sources xdcs

set build_dir  "$script_dir/build/$block"
set report_dir "$script_dir/reports/$block"
file mkdir $build_dir $report_dir

set_part $part
read_verilog -sv $sources
foreach x $xdcs { read_xdc -mode out_of_context $x }

synth_design -top $top -part $part -mode out_of_context
write_checkpoint -force $build_dir/post_synth.dcp
report_utilization -file $build_dir/utilization_synth.rpt

opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force $build_dir/post_route.dcp

report_timing_summary -max_paths 10 -report_unconstrained -warn_on_violation \
    -file $report_dir/timing_target.rpt
report_utilization -file $report_dir/utilization.rpt
report_utilization -hierarchical -hierarchical_depth 3 -file $report_dir/utilization_hier.rpt
report_methodology -file $build_dir/methodology.rpt

puts "DONE block=$block top=$top clock=${mhz}MHz period=${::CLK_PERIOD_NS}ns"
