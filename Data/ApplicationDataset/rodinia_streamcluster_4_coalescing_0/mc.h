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

#include <cstddef>
#include <cstring>

template<int W>
struct ap_uint {
    unsigned char data[(W + 7) / 8];
};

inline void memcpy_wide_bus_read_float(
    float* dst,
    const ap_uint<LARGE_BUS>* src,
    std::size_t offset,
    std::size_t num_bytes)
{
    std::memcpy(
        dst,
        reinterpret_cast<const unsigned char*>(src) + offset,
        num_bytes);
}

inline void memcpy_wide_bus_read_int(
    int* dst,
    const ap_uint<LARGE_BUS>* src,
    std::size_t offset,
    std::size_t num_bytes)
{
    std::memcpy(
        dst,
        reinterpret_cast<const unsigned char*>(src) + offset,
        num_bytes);
}

inline void memcpy_wide_bus_read_char(
    char* dst,
    const ap_uint<LARGE_BUS>* src,
    std::size_t offset,
    std::size_t num_bytes)
{
    std::memcpy(
        dst,
        reinterpret_cast<const unsigned char*>(src) + offset,
        num_bytes);
}

inline void memcpy_wide_bus_write_float(
    ap_uint<LARGE_BUS>* dst,
    const float* src,
    std::size_t offset,
    std::size_t num_bytes)
{
    std::memcpy(
        reinterpret_cast<unsigned char*>(dst) + offset,
        src,
        num_bytes);
}

inline void memcpy_wide_bus_write_char(
    ap_uint<LARGE_BUS>* dst,
    const char* src,
    std::size_t offset,
    std::size_t num_bytes)
{
    std::memcpy(
        reinterpret_cast<unsigned char*>(dst) + offset,
        src,
        num_bytes);
}


// #define MARS_WIDE_BUS_TYPE ap_uint<LARGE_BUS>

// #include "ap_int.h"

// #include "mars_wide_bus.h"
// #include "mars_wide_bus_2d.h"

// #include "mars_wide_bus_4d.h"

#endif
