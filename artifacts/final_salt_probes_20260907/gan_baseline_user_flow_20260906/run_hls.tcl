open_project -reset project_gan
set_top forward_propagation
add_files Application/GAN/network.cpp

open_solution -reset solution1 -flow_target vivado
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 10 -name default

config_array_partition -complete_threshold 0 -throughput_driven off
config_compile -pipeline_loops 0

csynth_design
exit
