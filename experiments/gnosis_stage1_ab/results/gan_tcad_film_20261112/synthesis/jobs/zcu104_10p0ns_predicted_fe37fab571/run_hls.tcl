open_project -reset hls_project
set_top forward_propagation
add_files -cflags {-I/opt/Xilinx/Vitis_HLS/2021.1/include} {source.cpp}
open_solution -reset solution1 -flow_target vivado
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 10.0 -name default
config_array_partition -complete_threshold 0 -throughput_driven off
config_compile -pipeline_loops 0
csynth_design
exit
