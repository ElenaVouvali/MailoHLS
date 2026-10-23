open_project -reset project
set_top chstone_aes
add_files {kernel.c} -cflags {-I. -D__gmp_const=const}
add_files -tb {tb.c} -cflags {-I. -D__gmp_const=const}
open_solution -reset solution1 -flow_target vivado
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 10 -name default
config_array_partition -complete_threshold 0 -throughput_driven off
config_compile -pipeline_loops 0
#csim_design
csynth_design
cosim_design -rtl verilog -tool xsim -trace_level none
exit
