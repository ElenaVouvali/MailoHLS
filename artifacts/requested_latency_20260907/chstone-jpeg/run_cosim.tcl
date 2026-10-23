open_project hls_project
add_files -tb tb_jpeg.c
open_solution solution1
cosim_design -rtl verilog -tool xsim -trace_level none
exit
