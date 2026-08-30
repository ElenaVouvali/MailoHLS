#pragma once

#include <cstdint>

template<int W>
struct ap_uint {

    std::uint64_t v;

    ap_uint(std::uint64_t x = 0) : v(x) {}

    ap_uint& operator=(std::uint64_t x) {
        v = x;
        return *this;
    }

    operator std::uint64_t() const { return v; }

    struct range_proxy {
        std::uint64_t &ref;
        range_proxy(std::uint64_t &r) : ref(r) {}

        range_proxy& operator=(std::uint64_t x) {
            ref = x;
            return *this;
        }

        operator std::uint64_t() const { return ref; }
    };

    // non-const: acts as an lvalue
    range_proxy range(int /*hi*/, int /*lo*/) {
        return range_proxy(v);
    }

    // const: acts as rvalue
    std::uint64_t range(int /*hi*/, int /*lo*/) const {
        return v;
    }
};

// if something uses ap_int, alias it to ap_uint
template<int W>
using ap_int = ap_uint<W>;

