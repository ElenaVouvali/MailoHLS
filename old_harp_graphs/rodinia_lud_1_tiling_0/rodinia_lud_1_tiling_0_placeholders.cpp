#include"lud.h"
#include <iostream>

//Elment with the block BSIZE, diagonal
#define AA(i,j) result[(offset + i) * matrix_dim + j + offset]
//Elment with global index
#define BB(i,j) result[i * matrix_dim + j]

using namespace std;
extern "C"{
void diagonal_load(float* result, float* buffer, int offset){
    int i, j;
L1:    for(i = 0; i < BSIZE; i++){
#pragma HLS pipeline II=auto{_PIPE_L1}
#pragma HLS unroll factor=auto{_UNROLL_L1}
L2:        for (j = 0; j < BSIZE; j++){
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
            buffer[i * BSIZE + j] = AA(i, j);
        }
    }
}

void diagonal_store(float* result, float* buffer, int offset){
    int i, j;
L3:    for(i = 0; i < BSIZE; i++){
#pragma HLS pipeline II=auto{_PIPE_L3}
#pragma HLS unroll factor=auto{_UNROLL_L3}
L4:        for (j = 0; j < BSIZE; j++){
#pragma HLS pipeline II=auto{_PIPE_L4}
#pragma HLS unroll factor=auto{_UNROLL_L4}
            AA(i, j) = buffer[i * BSIZE + j];
        }
    }
}
void lud_diagonal(float* result, 
                  int offset)
{ 
    int i, j, k;

L5:    float buffer[BSIZE * BSIZE];
#pragma HLS array_partition variable=buffer type=auto{_ARRAY_T_L5} factor=auto{_ARRAY_F_L5} dim=auto{_ARRAY_D_L5}

    diagonal_load(result, buffer, offset);

L6:    for (i = 0; i < BSIZE; i++){
#pragma HLS pipeline II=auto{_PIPE_L6}
#pragma HLS unroll factor=auto{_UNROLL_L6}
L7:        top:for (j = i; j < BSIZE; j++){
#pragma HLS pipeline II=auto{_PIPE_L7}
#pragma HLS unroll factor=auto{_UNROLL_L7}
L8:            for (k = 0; k < BSIZE; k++){
#pragma HLS pipeline II=auto{_PIPE_L8}
#pragma HLS unroll factor=auto{_UNROLL_L8}
                buffer[i * BSIZE + j] = buffer[i * BSIZE + j] - 
                buffer[i * BSIZE + k]* buffer[k * BSIZE + j];
		if(k==i) break;
            }
        }
           
           float temp = 1.f / buffer[i * BSIZE + i];

L9:           left:for (j = i + 1; j < BSIZE; j++){
#pragma HLS pipeline II=auto{_PIPE_L9}
#pragma HLS unroll factor=auto{_UNROLL_L9}
L10:               for (k = 0; k < BSIZE; k++){
#pragma HLS pipeline II=auto{_PIPE_L10}
#pragma HLS unroll factor=auto{_UNROLL_L10}
                   buffer[j * BSIZE + i]= buffer[j * BSIZE + i]- buffer[j * BSIZE + k] * buffer[k * BSIZE + i];
               	   if(k==i) break;
	       }
               buffer[j * BSIZE + i] = buffer[j * BSIZE + i] * temp;
           }
    }

    diagonal_store(result, buffer, offset);
}

void perimeter_load(float* result, float* top, float* left, int offset, int chunk_idx){
    int i, j;

    int i_top = offset; 
    int j_top = offset + BSIZE * (chunk_idx + 1);

    int i_left = offset + BSIZE * (chunk_idx + 1);
    int j_left = offset;

L11:    for(i = 0; i < BSIZE; i++){
#pragma HLS pipeline II=auto{_PIPE_L11}
#pragma HLS unroll factor=auto{_UNROLL_L11}
L12:        for(j = 0; j < BSIZE; j++){
#pragma HLS pipeline II=auto{_PIPE_L12}
#pragma HLS unroll factor=auto{_UNROLL_L12}
            top[i * BSIZE + j] = BB((i_top + i), (j_top + j));
        }
    }

L13:    for(i = 0; i < BSIZE; i++){
#pragma HLS pipeline II=auto{_PIPE_L13}
#pragma HLS unroll factor=auto{_UNROLL_L13}
L14:        for(j = 0; j < BSIZE; j++){
#pragma HLS pipeline II=auto{_PIPE_L14}
#pragma HLS unroll factor=auto{_UNROLL_L14}
            left[i * BSIZE + j] = BB((i_left + i), (j_left + j));
        }
    }
}

void perimeter_store(float* result, float* top, float* left, int offset, int chunk_idx){
    int i, j;

    int i_top = offset; 
    int j_top = offset + BSIZE * (chunk_idx + 1);

    int i_left = offset + BSIZE * (chunk_idx + 1);
    int j_left = offset;

L15:    for(i = 0; i < BSIZE; i++){
#pragma HLS pipeline II=auto{_PIPE_L15}
#pragma HLS unroll factor=auto{_UNROLL_L15}
L16:        for(j = 0; j < BSIZE; j++){
#pragma HLS pipeline II=auto{_PIPE_L16}
#pragma HLS unroll factor=auto{_UNROLL_L16}
            BB((i_top + i), (j_top + j)) = top[i * BSIZE + j];
        }
    }

L17:    for(i = 0; i < BSIZE; i++){
#pragma HLS pipeline II=auto{_PIPE_L17}
#pragma HLS unroll factor=auto{_UNROLL_L17}
L18:        for(j = 0; j < BSIZE; j++){
#pragma HLS pipeline II=auto{_PIPE_L18}
#pragma HLS unroll factor=auto{_UNROLL_L18}
            BB((i_left + i), (j_left + j)) = left[i * BSIZE + j];
        }
    }
}

void lud_perimeter(float* result, 
                   int    offset) {
L19:    float diagonal_buffer[BSIZE * BSIZE];
#pragma HLS array_partition variable=diagonal_buffer type=auto{_ARRAY_T_L19} factor=auto{_ARRAY_F_L19} dim=auto{_ARRAY_D_L19}
L20:    float top_buffer[BSIZE * BSIZE];
#pragma HLS array_partition variable=top_buffer type=auto{_ARRAY_T_L20} factor=auto{_ARRAY_F_L20} dim=auto{_ARRAY_D_L20}
L21:    float left_buffer[BSIZE * BSIZE];
#pragma HLS array_partition variable=left_buffer type=auto{_ARRAY_T_L21} factor=auto{_ARRAY_F_L21} dim=auto{_ARRAY_D_L21}

    int i, j, k;

L22:    diagonal:for (i = 0; i < BSIZE; i++){
#pragma HLS pipeline II=auto{_PIPE_L22}
#pragma HLS unroll factor=auto{_UNROLL_L22}
L23:        for (j = 0; j < BSIZE; j++){
#pragma HLS pipeline II=auto{_PIPE_L23}
#pragma HLS unroll factor=auto{_UNROLL_L23}
            diagonal_buffer[i * BSIZE + j] = AA(i, j);
        }
    }

    int chunk_idx, chunk_num;

    //OURS
    offset = 0;

    chunk_num = ((matrix_dim - offset) / BSIZE) - 1;

L24:    for (chunk_idx = 0; chunk_idx < 100 /*chunk_num*/; chunk_idx++){
#pragma HLS pipeline II=auto{_PIPE_L24}
#pragma HLS unroll factor=auto{_UNROLL_L24}
        perimeter_load(result, top_buffer, left_buffer, offset, chunk_idx);
                
                        float sum; 
                        // processing top perimeter
L25:                        for (j = 0; j < BSIZE; j++){
#pragma HLS pipeline II=auto{_PIPE_L25}
#pragma HLS unroll factor=auto{_UNROLL_L25}
L26:                            for (i = 0; i < BSIZE; i++){ 
#pragma HLS pipeline II=auto{_PIPE_L26}
#pragma HLS unroll factor=auto{_UNROLL_L26}
                                sum = 0.0f; 
L27:                                for (k = 0; k < BSIZE; k++){
#pragma HLS pipeline II=auto{_PIPE_L27}
#pragma HLS unroll factor=auto{_UNROLL_L27}
                                    sum += diagonal_buffer[BSIZE * i + k] * top_buffer[k * BSIZE + j];
                                    if(k==j) break;
				}

                                top_buffer[i * BSIZE + j] = top_buffer[i * BSIZE + j] - sum;
                            }
                        }


                        // processing left perimeter
L28:                        for (i = 0; i < BSIZE; i++){
#pragma HLS pipeline II=auto{_PIPE_L28}
#pragma HLS unroll factor=auto{_UNROLL_L28}
L29:                            for (j = 0; j < BSIZE; j++){
#pragma HLS pipeline II=auto{_PIPE_L29}
#pragma HLS unroll factor=auto{_UNROLL_L29}
                                 sum = 0.0f;
L30:                                 for (k = 0; k < BSIZE; k++){
#pragma HLS pipeline II=auto{_PIPE_L30}
#pragma HLS unroll factor=auto{_UNROLL_L30}
                                     sum += left_buffer[i * BSIZE + k] * diagonal_buffer[BSIZE * k + j];
                                     if(k==j) break;
				 }

                                 left_buffer[i * BSIZE + j] = (left_buffer[i * BSIZE + j] - sum) / diagonal_buffer[j * BSIZE + j];
                            }
                        }

                        perimeter_store(result, top_buffer, left_buffer, offset, chunk_idx);

    }
    cout << "success here perimeter" << endl;
}

void internal_load(float* result, float* top, float* left, float* inner, int offset, int chunk_idx, int chunk_num){
    int i, j;
    int i_global, j_global;

    i_global = offset + BSIZE * (1 + chunk_idx / chunk_num);
    j_global = offset + BSIZE * (1 + chunk_idx % chunk_num);

L31:    for(i = 0; i < BSIZE; i++){
#pragma HLS pipeline II=auto{_PIPE_L31}
#pragma HLS unroll factor=auto{_UNROLL_L31}
L32:        for(j = 0; j < BSIZE; j++){
#pragma HLS pipeline II=auto{_PIPE_L32}
#pragma HLS unroll factor=auto{_UNROLL_L32}
            top[i * BSIZE + j]  = result[matrix_dim * (i + offset) + j + j_global];
        }
    }
L33:    for(i = 0; i < BSIZE; i++){
#pragma HLS pipeline II=auto{_PIPE_L33}
#pragma HLS unroll factor=auto{_UNROLL_L33}
L34:        for(j = 0; j < BSIZE; j++){
#pragma HLS pipeline II=auto{_PIPE_L34}
#pragma HLS unroll factor=auto{_UNROLL_L34}
            left[i * BSIZE + j] = result[matrix_dim * (i + i_global) + offset + j];
        }
    }

L35:    for(i = 0; i < BSIZE; i++){
#pragma HLS pipeline II=auto{_PIPE_L35}
#pragma HLS unroll factor=auto{_UNROLL_L35}
L36:        for(j = 0; j < BSIZE; j++){
#pragma HLS pipeline II=auto{_PIPE_L36}
#pragma HLS unroll factor=auto{_UNROLL_L36}
            inner[i * BSIZE + j] = result[matrix_dim * (i + i_global) + j + j_global];
        }
    }
}

void internal_store(float* result, float* inner, int offset, int chunk_idx, int chunk_num){
    int i, j;
    int i_global, j_global;

    i_global = offset + BSIZE * (1 + chunk_idx / chunk_num);
    j_global = offset + BSIZE * (1 + chunk_idx % chunk_num);

L37:    for(i = 0; i < BSIZE; i++){
#pragma HLS pipeline II=auto{_PIPE_L37}
#pragma HLS unroll factor=auto{_UNROLL_L37}
L38:        for(j = 0; j < BSIZE; j++){
#pragma HLS pipeline II=auto{_PIPE_L38}
#pragma HLS unroll factor=auto{_UNROLL_L38}
            result[matrix_dim * (i + i_global) + j + j_global] = inner[i * BSIZE + j];
        }
    }
}

void lud_internal( float* result, 
                    int  offset) {
    int chunk_idx, chunk_num;
    
    //OURS
    offset = 0;	
    chunk_num = ((matrix_dim - offset) / BSIZE) - 1;

L39:    float top_buffer[BSIZE * BSIZE];
#pragma HLS array_partition variable=top_buffer type=auto{_ARRAY_T_L39} factor=auto{_ARRAY_F_L39} dim=auto{_ARRAY_D_L39}
L40:    float left_buffer[BSIZE * BSIZE];
#pragma HLS array_partition variable=left_buffer type=auto{_ARRAY_T_L40} factor=auto{_ARRAY_F_L40} dim=auto{_ARRAY_D_L40}
L41:    float inner_buffer[BSIZE * BSIZE];
#pragma HLS array_partition variable=inner_buffer type=auto{_ARRAY_T_L41} factor=auto{_ARRAY_F_L41} dim=auto{_ARRAY_D_L41}

    int i, j, k, i_global, j_global;

L42:    for  (chunk_idx = 0; chunk_idx < 100 /*chunk_num * chunk_num*/; chunk_idx++){
#pragma HLS pipeline II=auto{_PIPE_L42}
#pragma HLS unroll factor=auto{_UNROLL_L42}
        internal_load(result, top_buffer, left_buffer, inner_buffer, offset, chunk_idx, chunk_num);

L43:        for (i = 0; i < BSIZE; i++){
#pragma HLS pipeline II=auto{_PIPE_L43}
#pragma HLS unroll factor=auto{_UNROLL_L43}
L44:            for (j = 0; j < BSIZE; j++){
#pragma HLS pipeline II=auto{_PIPE_L44}
#pragma HLS unroll factor=auto{_UNROLL_L44}
                float sum = 0.0f;
                //#pragma HLS unsafemath
L45:                for (k = 0; k < BSIZE; k++){
#pragma HLS pipeline II=auto{_PIPE_L45}
#pragma HLS unroll factor=auto{_UNROLL_L45}
                    sum += left_buffer[BSIZE * i + k] * top_buffer[BSIZE * k + j];
                }
                inner_buffer[i * BSIZE + j] -= sum;
            }
        }

        internal_store(result, inner_buffer, offset, chunk_idx, chunk_num);
    }

    cout << "success internal" << endl;
}

void workload(float result[GRID_ROWS * GRID_COLS]){
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=result bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

L46:    for(int i = 0; i < matrix_dim - BSIZE; i += BSIZE){
#pragma HLS pipeline II=auto{_PIPE_L46}
#pragma HLS unroll factor=auto{_UNROLL_L46}
        lud_diagonal(result, i);
        lud_perimeter(result, i);
        lud_internal(result, i);
    }

    int i = matrix_dim - BSIZE;
    lud_diagonal(result, i);
    return;
}

}
