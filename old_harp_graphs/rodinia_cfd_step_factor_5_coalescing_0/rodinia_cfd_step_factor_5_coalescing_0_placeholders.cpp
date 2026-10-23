#include"cfd_step_factor.h"

#define LARGE_BUS 512
#include "mc.h"

extern "C" {
inline void compute_velocity(float& density, float3& momentum, float3& velocity)
{
    velocity.x = (float)momentum.x / (float)density;
    velocity.y = (float)momentum.y / (float)density;
    velocity.z = (float)momentum.z / (float)density;
}

inline float compute_speed_sqd(float3& velocity)
{
    return (float)velocity.x*(float)velocity.x + (float)velocity.y*(float)velocity.y + (float)velocity.z*(float)velocity.z;
}

inline float compute_pressure(float& density, float& density_energy, float& speed_sqd)
{
    return (float(GAMMA) - float(1.0f))*(float)((float)density_energy - float(0.5f)*(float)((float)density*(float)speed_sqd));
}

inline float compute_speed_of_sound(float& density, float& pressure)
{
    return (float)sqrt((float)(float(GAMMA)*(float)pressure / (float)density));
}

void cfd_step_factor(float result[TILE_ROWS], float variables[TILE_ROWS * NVAR], float areas[TILE_ROWS])
{
L1:    for (int i = 0; i < TILE_ROWS / PARA_FACTOR; i++) {
#pragma HLS pipeline II=auto{_PIPE_L1}
#pragma HLS unroll factor=auto{_UNROLL_L1}


L2:        float density[PARA_FACTOR];
#pragma HLS array_partition variable=density type=auto{_ARRAY_T_L2} factor=auto{_ARRAY_F_L2} dim=auto{_ARRAY_D_L2}
L3:        float3 momentum[PARA_FACTOR];
#pragma HLS array_partition variable=momentum type=auto{_ARRAY_T_L3} factor=auto{_ARRAY_F_L3} dim=auto{_ARRAY_D_L3}
L4:        float density_energy[PARA_FACTOR];
#pragma HLS array_partition variable=density_energy type=auto{_ARRAY_T_L4} factor=auto{_ARRAY_F_L4} dim=auto{_ARRAY_D_L4}

        int iii = i * PARA_FACTOR;

        density[0] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 0 ]; 
        momentum[0].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 0)];
        momentum[0].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 0)];
        momentum[0].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 0)];
        density_energy[0] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 0 ];

        density[1] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 5 ]; 
        momentum[1].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 5)];
        momentum[1].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 5)];
        momentum[1].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 5)];
        density_energy[1] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 5 ];


        density[2] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 10 ]; 
        momentum[2].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 10)];
        momentum[2].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 10)];
        momentum[2].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 10)];
        density_energy[2] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 10 ];


        density[3] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 15 ]; 
        momentum[3].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 15)];
        momentum[3].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 15)];
        momentum[3].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 15)];
        density_energy[3] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 15 ];


        density[4] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 20 ]; 
        momentum[4].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 20)];
        momentum[4].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 20)];
        momentum[4].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 20)];
        density_energy[4] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 20 ];


        density[5] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 25 ]; 
        momentum[5].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 25)];
        momentum[5].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 25)];
        momentum[5].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 25)];
        density_energy[5] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 25 ];


        density[6] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 30 ]; 
        momentum[6].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 30)];
        momentum[6].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 30)];
        momentum[6].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 30)];
        density_energy[6] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 30 ];


        density[7] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 35 ]; 
        momentum[7].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 35)];
        momentum[7].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 35)];
        momentum[7].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 35)];
        density_energy[7] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 35 ];


        density[8] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 40 ]; 
        momentum[8].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 40)];
        momentum[8].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 40)];
        momentum[8].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 40)];
        density_energy[8] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 40 ];

        density [9] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 45 ]; 
        momentum[9].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 45)];
        momentum[9].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 45)];
        momentum[9].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 45)];
        density_energy[9] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 45 ];


        density[10] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 50 ]; 
        momentum[10].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 50)];
        momentum[10].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 50)];
        momentum[10].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 50)];
        density_energy[10] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 50 ];


        density[11] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 55 ]; 
        momentum[11].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 55)];
        momentum[11].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 55)];
        momentum[11].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 55)];
        density_energy[11] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 55 ];


        density[12] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 60 ]; 
        momentum[12].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 60)];
        momentum[12].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 60)];
        momentum[12].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 60)];
        density_energy[12] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 60 ];


        density[13] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 65 ]; 
        momentum[13].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 65)];
        momentum[13].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 65)];
        momentum[13].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 65)];
        density_energy[13] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 65 ];


        density[14] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 70 ]; 
        momentum[14].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 70)];
        momentum[14].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 70)];
        momentum[14].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 70)];
        density_energy[14] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 70 ];


        density[15] = variables      [NVAR*iii + VAR_DENSITY     +   0 + 75 ]; 
        momentum[15].x = variables   [NVAR*iii + (VAR_MOMENTUM   +   0 + 75)];
        momentum[15].y = variables[NVAR*iii + (VAR_MOMENTUM      +   1 + 75)];
        momentum[15].z = variables[NVAR*iii + (VAR_MOMENTUM      +   2 + 75)];
        density_energy[15] = variables[NVAR*iii + VAR_DENSITY_ENERGY   + 75 ];


L5:        for (int ii = 0; ii < PARA_FACTOR; ii++) {
#pragma HLS pipeline II=auto{_PIPE_L5}
#pragma HLS unroll factor=auto{_UNROLL_L5}
            int iii = i * PARA_FACTOR + ii;
            float3 velocity;       compute_velocity(density[ii], momentum[ii], velocity);
            float speed_sqd = compute_speed_sqd(velocity);
            float pressure = compute_pressure(density[ii], density_energy[ii], speed_sqd);
            float speed_of_sound = compute_speed_of_sound(density[ii], pressure);
    
            result[iii] = float(0.5f) / (float)((float)sqrt((float)areas[iii]) * (float)((float)sqrt((float)speed_sqd) + (float)speed_of_sound));
        }
        
    }
}


void buffer_load_variables(int flag, int k, float variables_inner[TILE_ROWS * NVAR], class ap_uint<LARGE_BUS> * variables)
{
#pragma HLS inline off
    if (flag) memcpy_wide_bus_read_float(variables_inner, variables + (k * TILE_ROWS * NVAR) / (LARGE_BUS / 32), 0, sizeof(float) * TILE_ROWS * NVAR);
    return;
}

void buffer_load_areas(int flag, int k, float areas_inner[TILE_ROWS], class ap_uint<LARGE_BUS> * areas)
{
#pragma HLS inline off
    if (flag) memcpy_wide_bus_read_float(areas_inner, areas + (k * TILE_ROWS) / (LARGE_BUS / 32), 0, sizeof(float) * TILE_ROWS);
    return;
}

void buffer_compute(int flag, float result_inner[TILE_ROWS], float variables_inner[TILE_ROWS * NVAR], float areas_inner[TILE_ROWS])
{
#pragma HLS inline off
    if (flag) cfd_step_factor(result_inner, variables_inner, areas_inner);
    return;
}

void buffer_store(int flag, int k, class ap_uint<LARGE_BUS> * result, float result_inner[TILE_ROWS])
{
#pragma HLS inline off
    if (flag) memcpy_wide_bus_write_float(result + (k * TILE_ROWS) / (LARGE_BUS / 32), result_inner, 0, sizeof(float) * TILE_ROWS);
    return;
}

void workload(class ap_uint<LARGE_BUS> * result,  class ap_uint<LARGE_BUS> * variables, class ap_uint<LARGE_BUS> * areas)
{
    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=result1
    #pragma HLS INTERFACE m_axi port=variables offset=slave bundle=variables1
    #pragma HLS INTERFACE m_axi port=areas offset=slave bundle=areas1
    
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=variables bundle=control
    #pragma HLS INTERFACE s_axilite port=areas bundle=control
    
    #pragma HLS INTERFACE s_axilite port=return bundle=control

L6:    float result_inner_0      [TILE_ROWS];
#pragma HLS array_partition variable=result_inner_0 type=auto{_ARRAY_T_L6} factor=auto{_ARRAY_F_L6} dim=auto{_ARRAY_D_L6}
L7:    float variables_inner_0   [TILE_ROWS * NVAR];
#pragma HLS array_partition variable=variables_inner_0 type=auto{_ARRAY_T_L7} factor=auto{_ARRAY_F_L7} dim=auto{_ARRAY_D_L7}
L8:    float areas_inner_0       [TILE_ROWS];
#pragma HLS array_partition variable=areas_inner_0 type=auto{_ARRAY_T_L8} factor=auto{_ARRAY_F_L8} dim=auto{_ARRAY_D_L8}

L9:    float result_inner_1      [TILE_ROWS];
#pragma HLS array_partition variable=result_inner_1 type=auto{_ARRAY_T_L9} factor=auto{_ARRAY_F_L9} dim=auto{_ARRAY_D_L9}
L10:    float variables_inner_1   [TILE_ROWS * NVAR];
#pragma HLS array_partition variable=variables_inner_1 type=auto{_ARRAY_T_L10} factor=auto{_ARRAY_F_L10} dim=auto{_ARRAY_D_L10}
L11:    float areas_inner_1       [TILE_ROWS];
#pragma HLS array_partition variable=areas_inner_1 type=auto{_ARRAY_T_L11} factor=auto{_ARRAY_F_L11} dim=auto{_ARRAY_D_L11}

L12:    float result_inner_2      [TILE_ROWS];
#pragma HLS array_partition variable=result_inner_2 type=auto{_ARRAY_T_L12} factor=auto{_ARRAY_F_L12} dim=auto{_ARRAY_D_L12}
L13:    float variables_inner_2   [TILE_ROWS * NVAR];
#pragma HLS array_partition variable=variables_inner_2 type=auto{_ARRAY_T_L13} factor=auto{_ARRAY_F_L13} dim=auto{_ARRAY_D_L13}
L14:    float areas_inner_2       [TILE_ROWS];
#pragma HLS array_partition variable=areas_inner_2 type=auto{_ARRAY_T_L14} factor=auto{_ARRAY_F_L14} dim=auto{_ARRAY_D_L14}

L15:    float result_inner_3      [TILE_ROWS];
#pragma HLS array_partition variable=result_inner_3 type=auto{_ARRAY_T_L15} factor=auto{_ARRAY_F_L15} dim=auto{_ARRAY_D_L15}
L16:    float variables_inner_3   [TILE_ROWS * NVAR];
#pragma HLS array_partition variable=variables_inner_3 type=auto{_ARRAY_T_L16} factor=auto{_ARRAY_F_L16} dim=auto{_ARRAY_D_L16}
L17:    float areas_inner_3       [TILE_ROWS];
#pragma HLS array_partition variable=areas_inner_3 type=auto{_ARRAY_T_L17} factor=auto{_ARRAY_F_L17} dim=auto{_ARRAY_D_L17}

L18:    for (int k = 0; k < SIZE / TILE_ROWS + 3; k++) {
#pragma HLS pipeline II=auto{_PIPE_L18}
#pragma HLS unroll factor=auto{_UNROLL_L18}

        int load_variables_flag = k >= 0 && k < (SIZE / TILE_ROWS);
        int load_areas_flag = k >= 1 && k < (SIZE / TILE_ROWS) + 1;
        int compute_flag = k >= 2 && k < (SIZE / TILE_ROWS) + 2;
        int store_flag = k >= 3 && k < (SIZE / TILE_ROWS) + 3;
        
        if (k % 4 == 0) {
            buffer_load_variables(load_variables_flag, k, variables_inner_0, variables);
            buffer_load_areas(load_areas_flag, k - 1, areas_inner_3, areas);
            buffer_compute(compute_flag, result_inner_2, variables_inner_2, areas_inner_2);
            buffer_store(store_flag, k - 3, result, result_inner_1);
        }
    
        else if (k % 4 == 1) {
            buffer_load_variables(load_variables_flag, k, variables_inner_1, variables);
            buffer_load_areas(load_areas_flag, k - 1, areas_inner_0, areas);
            buffer_compute(compute_flag, result_inner_3, variables_inner_3, areas_inner_3);
            buffer_store(store_flag, k - 3, result, result_inner_2);
        }
        
        else if (k % 4 == 2){
            buffer_load_variables(load_variables_flag, k, variables_inner_2, variables);
            buffer_load_areas(load_areas_flag, k - 1, areas_inner_1, areas);
            buffer_compute(compute_flag, result_inner_0, variables_inner_0, areas_inner_0);
            buffer_store(store_flag, k - 3, result, result_inner_3);
        }
        else {
            buffer_load_variables(load_variables_flag, k, variables_inner_3, variables);
            buffer_load_areas(load_areas_flag, k - 1, areas_inner_2, areas);
            buffer_compute(compute_flag, result_inner_1, variables_inner_1, areas_inner_1);
            buffer_store(store_flag, k - 3, result, result_inner_0);
        }

    } 
    return;
}

}

