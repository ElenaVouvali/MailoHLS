open_project -reset hls_project
set_top forward_propagation
add_files {network.cpp} -cflags {-I.}
open_solution -reset solution1 -flow_target vivado
set_part {xcu200-fsgd2104-2-e}
create_clock -period 3.33 -name default
config_array_partition -complete_threshold 0 -throughput_driven off
config_compile -pipeline_loops 0
csynth_design
exit
