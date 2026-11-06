# Waveform layout for systolic_array_tb in the Vivado simulator GUI.
# Every signal is logged, so anything else (e.g. a single PE under dut) can be
# dragged in from the Scope panel.
log_wave -recursive *

set tb /systolic_array_tb
add_wave_divider {Control}
add_wave $tb/clk $tb/rst_n $tb/step_en $tb/acc_clear
add_wave_divider {Array inputs (already skewed)}
add_wave -radix dec $tb/a_in
add_wave -radix bin $tb/a_valid_in
add_wave -radix dec $tb/b_in
add_wave -radix bin $tb/b_valid_in
add_wave_divider {Mesh links: a_link[r][c] enters PE(r,c) from the left, b_link[r][c] from above}
add_wave -radix dec $tb/dut/a_link
add_wave -radix bin $tb/dut/av_link
add_wave -radix dec $tb/dut/b_link
add_wave -radix bin $tb/dut/bv_link
add_wave_divider {Results: acc_out[r][c] vs expected}
add_wave -radix dec $tb/acc_out $tb/expected

run all
