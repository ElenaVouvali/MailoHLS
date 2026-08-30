#ifndef AP_FIXED_STUB_H
#define AP_FIXED_STUB_H

#include <cstdint>
#include <cmath>

// ------------------------------------------------------------
// Minimal HLS enum stub
// ------------------------------------------------------------
enum { AP_RND = 0 };

// Forward declarations
template<int W> struct ap_int;
template<int W> struct ap_uint;

// ------------------------------------------------------------
// Small compile-time max helper
// ------------------------------------------------------------
template<int A, int B>
struct ap_max {
    static const int value = (A > B) ? A : B;
};

// ------------------------------------------------------------
// ap_uint
// ------------------------------------------------------------
template<int W>
struct ap_uint {
    std::uint64_t v;

    ap_uint(std::uint64_t x = 0) : v(x) {}

    template<int W2>
    ap_uint(const ap_uint<W2>& x) : v(static_cast<std::uint64_t>(x)) {}

    template<int W2>
    ap_uint(const ap_int<W2>& x);

    ap_uint& operator=(std::uint64_t x) {
        v = x;
        return *this;
    }

    template<int W2>
    ap_uint& operator=(const ap_uint<W2>& x) {
        v = static_cast<std::uint64_t>(x);
        return *this;
    }

    template<int W2>
    ap_uint& operator=(const ap_int<W2>& x);

    operator std::uint64_t() const { return v; }

    int to_int() const { return static_cast<int>(v); }
    unsigned to_uint() const { return static_cast<unsigned>(v); }

    struct range_proxy {
        std::uint64_t& ref;

        range_proxy(std::uint64_t& r) : ref(r) {}

        range_proxy& operator=(std::uint64_t x) {
            ref = x;
            return *this;
        }

        operator std::uint64_t() const { return ref; }
    };

    range_proxy range(int /*hi*/, int /*lo*/) {
        return range_proxy(v);
    }

    std::uint64_t range(int /*hi*/, int /*lo*/) const {
        return v;
    }
};

// ------------------------------------------------------------
// ap_int
// ------------------------------------------------------------
template<int W>
struct ap_int {
    std::int64_t v;

    ap_int(std::int64_t x = 0) : v(x) {}

    template<int W2>
    ap_int(const ap_int<W2>& x) : v(static_cast<std::int64_t>(x)) {}

    template<int W2>
    ap_int(const ap_uint<W2>& x)
        : v(static_cast<std::int64_t>(static_cast<std::uint64_t>(x))) {}

    ap_int& operator=(std::int64_t x) {
        v = x;
        return *this;
    }

    template<int W2>
    ap_int& operator=(const ap_int<W2>& x) {
        v = static_cast<std::int64_t>(x);
        return *this;
    }

    template<int W2>
    ap_int& operator=(const ap_uint<W2>& x) {
        v = static_cast<std::int64_t>(static_cast<std::uint64_t>(x));
        return *this;
    }

    operator std::int64_t() const { return v; }

    int to_int() const { return static_cast<int>(v); }

    struct range_proxy {
        std::int64_t& ref;

        range_proxy(std::int64_t& r) : ref(r) {}

        range_proxy& operator=(std::int64_t x) {
            ref = x;
            return *this;
        }

        operator std::int64_t() const { return ref; }
    };

    range_proxy range(int /*hi*/, int /*lo*/) {
        return range_proxy(v);
    }

    std::int64_t range(int /*hi*/, int /*lo*/) const {
        return v;
    }
};

// ap_uint methods that depend on ap_int
template<int W>
template<int W2>
ap_uint<W>::ap_uint(const ap_int<W2>& x)
    : v(static_cast<std::uint64_t>(static_cast<std::int64_t>(x))) {}

template<int W>
template<int W2>
ap_uint<W>& ap_uint<W>::operator=(const ap_int<W2>& x) {
    v = static_cast<std::uint64_t>(static_cast<std::int64_t>(x));
    return *this;
}

// ------------------------------------------------------------
// ap_fixed
// ------------------------------------------------------------
template<int W, int I, int Q = 0>
struct ap_fixed {
    float v;

    ap_fixed(float x = 0.0f) : v(x) {}
    ap_fixed(double x) : v(static_cast<float>(x)) {}
    ap_fixed(int x) : v(static_cast<float>(x)) {}
    ap_fixed(long x) : v(static_cast<float>(x)) {}
    ap_fixed(long long x) : v(static_cast<float>(x)) {}

    template<int W2, int I2, int Q2>
    ap_fixed(const ap_fixed<W2, I2, Q2>& x) : v(x.v) {}

    template<int W2>
    ap_fixed(const ap_int<W2>& x)
        : v(static_cast<float>(static_cast<std::int64_t>(x))) {}

    template<int W2>
    ap_fixed(const ap_uint<W2>& x)
        : v(static_cast<float>(static_cast<std::uint64_t>(x))) {}

    ap_fixed& operator=(float x) {
        v = x;
        return *this;
    }

    ap_fixed& operator=(double x) {
        v = static_cast<float>(x);
        return *this;
    }

    template<int W2, int I2, int Q2>
    ap_fixed& operator=(const ap_fixed<W2, I2, Q2>& x) {
        v = x.v;
        return *this;
    }

    operator float() const { return v; }

    float to_float() const { return v; }
    double to_double() const { return static_cast<double>(v); }
    int to_int() const { return static_cast<int>(v); }

    template<int W2, int I2, int Q2>
    ap_fixed& operator+=(const ap_fixed<W2, I2, Q2>& other) {
        v += other.v;
        return *this;
    }

    template<int W2, int I2, int Q2>
    ap_fixed& operator-=(const ap_fixed<W2, I2, Q2>& other) {
        v -= other.v;
        return *this;
    }

    template<int W2, int I2, int Q2>
    ap_fixed& operator*=(const ap_fixed<W2, I2, Q2>& other) {
        v *= other.v;
        return *this;
    }

    template<int W2, int I2, int Q2>
    ap_fixed& operator/=(const ap_fixed<W2, I2, Q2>& other) {
        v /= other.v;
        return *this;
    }

    ap_int<W> range() const {
        return ap_int<W>(static_cast<std::int64_t>(v));
    }

    ap_int<W> range(int /*hi*/, int /*lo*/) const {
        return ap_int<W>(static_cast<std::int64_t>(v));
    }
};

// ------------------------------------------------------------
// Mixed ap_fixed/ap_fixed arithmetic
// ------------------------------------------------------------
template<int W1, int I1, int Q1, int W2, int I2, int Q2>
inline ap_fixed<ap_max<W1,W2>::value, ap_max<I1,I2>::value, 0>
operator+(const ap_fixed<W1,I1,Q1>& a, const ap_fixed<W2,I2,Q2>& b) {
    typedef ap_fixed<ap_max<W1,W2>::value, ap_max<I1,I2>::value, 0> R;
    return R(a.v + b.v);
}

template<int W1, int I1, int Q1, int W2, int I2, int Q2>
inline ap_fixed<ap_max<W1,W2>::value, ap_max<I1,I2>::value, 0>
operator-(const ap_fixed<W1,I1,Q1>& a, const ap_fixed<W2,I2,Q2>& b) {
    typedef ap_fixed<ap_max<W1,W2>::value, ap_max<I1,I2>::value, 0> R;
    return R(a.v - b.v);
}

template<int W1, int I1, int Q1, int W2, int I2, int Q2>
inline ap_fixed<ap_max<W1,W2>::value, ap_max<I1,I2>::value, 0>
operator*(const ap_fixed<W1,I1,Q1>& a, const ap_fixed<W2,I2,Q2>& b) {
    typedef ap_fixed<ap_max<W1,W2>::value, ap_max<I1,I2>::value, 0> R;
    return R(a.v * b.v);
}

template<int W1, int I1, int Q1, int W2, int I2, int Q2>
inline ap_fixed<ap_max<W1,W2>::value, ap_max<I1,I2>::value, 0>
operator/(const ap_fixed<W1,I1,Q1>& a, const ap_fixed<W2,I2,Q2>& b) {
    typedef ap_fixed<ap_max<W1,W2>::value, ap_max<I1,I2>::value, 0> R;
    return R(a.v / b.v);
}

// ------------------------------------------------------------
// ap_fixed with scalar
// ------------------------------------------------------------
template<int W, int I, int Q>
inline ap_fixed<W,I,Q> operator+(const ap_fixed<W,I,Q>& a, float b) {
    return ap_fixed<W,I,Q>(a.v + b);
}

template<int W, int I, int Q>
inline ap_fixed<W,I,Q> operator-(const ap_fixed<W,I,Q>& a, float b) {
    return ap_fixed<W,I,Q>(a.v - b);
}

template<int W, int I, int Q>
inline ap_fixed<W,I,Q> operator*(const ap_fixed<W,I,Q>& a, float b) {
    return ap_fixed<W,I,Q>(a.v * b);
}

template<int W, int I, int Q>
inline ap_fixed<W,I,Q> operator/(const ap_fixed<W,I,Q>& a, float b) {
    return ap_fixed<W,I,Q>(a.v / b);
}

template<int W, int I, int Q>
inline ap_fixed<W,I,Q> operator+(float a, const ap_fixed<W,I,Q>& b) {
    return ap_fixed<W,I,Q>(a + b.v);
}

template<int W, int I, int Q>
inline ap_fixed<W,I,Q> operator-(float a, const ap_fixed<W,I,Q>& b) {
    return ap_fixed<W,I,Q>(a - b.v);
}

template<int W, int I, int Q>
inline ap_fixed<W,I,Q> operator*(float a, const ap_fixed<W,I,Q>& b) {
    return ap_fixed<W,I,Q>(a * b.v);
}

template<int W, int I, int Q>
inline ap_fixed<W,I,Q> operator/(float a, const ap_fixed<W,I,Q>& b) {
    return ap_fixed<W,I,Q>(a / b.v);
}

// ------------------------------------------------------------
// Comparisons
// ------------------------------------------------------------
template<int W1, int I1, int Q1, int W2, int I2, int Q2>
inline bool operator<(const ap_fixed<W1,I1,Q1>& a, const ap_fixed<W2,I2,Q2>& b) {
    return a.v < b.v;
}

template<int W1, int I1, int Q1, int W2, int I2, int Q2>
inline bool operator>(const ap_fixed<W1,I1,Q1>& a, const ap_fixed<W2,I2,Q2>& b) {
    return a.v > b.v;
}

template<int W1, int I1, int Q1, int W2, int I2, int Q2>
inline bool operator<=(const ap_fixed<W1,I1,Q1>& a, const ap_fixed<W2,I2,Q2>& b) {
    return a.v <= b.v;
}

template<int W1, int I1, int Q1, int W2, int I2, int Q2>
inline bool operator>=(const ap_fixed<W1,I1,Q1>& a, const ap_fixed<W2,I2,Q2>& b) {
    return a.v >= b.v;
}

template<int W1, int I1, int Q1, int W2, int I2, int Q2>
inline bool operator==(const ap_fixed<W1,I1,Q1>& a, const ap_fixed<W2,I2,Q2>& b) {
    return a.v == b.v;
}

template<int W1, int I1, int Q1, int W2, int I2, int Q2>
inline bool operator!=(const ap_fixed<W1,I1,Q1>& a, const ap_fixed<W2,I2,Q2>& b) {
    return a.v != b.v;
}

// ------------------------------------------------------------
// Math overload
// ------------------------------------------------------------
template<int W, int I, int Q>
inline ap_fixed<W,I,Q> tanh(const ap_fixed<W,I,Q>& x) {
    return ap_fixed<W,I,Q>(std::tanh(x.v));
}

#endif // AP_FIXED_STUB_H
