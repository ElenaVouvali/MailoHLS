open_project -reset project_trmm
set_top kernel_trmm
add_files {kernel.cpp}
open_solution -reset solution1 -flow_target vivado
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 10 -name default
config_array_partition -complete_threshold 0 -throughput_driven off
config_compile -pipeline_loops 0
csynth_design
exit
