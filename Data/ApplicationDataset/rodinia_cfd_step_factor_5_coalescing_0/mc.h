#ifndef MC_H
#define MC_H

#ifndef LARGE_BUS
#define LARGE_BUS 512
#endif 

#ifndef SIZE_1
#define SIZE_1 1
#endif 

#ifndef SIZE_2
#define SIZE_2 1
#endif 

#ifndef SIZE_3
#define SIZE_3 1
#endif 

// #define MARS_WIDE_BUS_TYPE ap_uint<LARGE_BUS>

// #include "ap_int.h"

// #include "mars_wide_bus.h"
// #include "mars_wide_bus_2d.h"

// #include "mars_wide_bus_4d.h"

#ifndef __VIVADO_HLS__
#include <stdint.h>

template<int W>
struct ap_uint {
    uint32_t words[(W + 31) / 32];
};

void memcpy_wide_bus_read_float(
    float *dst,
    const ap_uint<LARGE_BUS> *src,
    int offset,
    int size);

void memcpy_wide_bus_write_float(
    ap_uint<LARGE_BUS> *dst,
    const float *src,
    int offset,
    int size);

#endif  

#endif  


