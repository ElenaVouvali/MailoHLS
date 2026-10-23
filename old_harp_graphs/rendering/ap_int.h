#ifndef AP_INT_H
#define AP_INT_H

#include <stdint.h>

template<int W, bool SIGNED>
class ap_int_base {
public:
  long long v;

  ap_int_base(long long x = 0) : v(x) {}

  template<int W2, bool S2>
  ap_int_base(const ap_int_base<W2, S2>& other) : v((long long)other) {}

  operator long long() const { return v; }

  ap_int_base& operator=(long long x) {
    v = x;
    return *this;
  }

  template<int W2, bool S2>
  ap_int_base& operator=(const ap_int_base<W2, S2>& other) {
    v = (long long)other;
    return *this;
  }

  ap_int_base& operator++() {
    ++v;
    return *this;
  }

  ap_int_base operator++(int) {
    ap_int_base tmp(*this);
    ++v;
    return tmp;
  }

  ap_int_base& operator--() {
    --v;
    return *this;
  }

  ap_int_base operator--(int) {
    ap_int_base tmp(*this);
    --v;
    return tmp;
  }

  class range_ref {
  public:
    ap_int_base& parent;
    int hi;
    int lo;

    range_ref(ap_int_base& p, int h, int l) : parent(p), hi(h), lo(l) {}

    range_ref& operator=(long long x) {
      int width = hi - lo + 1;
      unsigned long long mask;

      if (width >= 64) {
        mask = ~0ULL;
      } else {
        mask = ((1ULL << width) - 1ULL) << lo;
      }

      unsigned long long pv = (unsigned long long)parent.v;
      pv &= ~mask;
      pv |= (((unsigned long long)x << lo) & mask);
      parent.v = (long long)pv;
      return *this;
    }

    template<int W2, bool S2>
    range_ref& operator=(const ap_int_base<W2, S2>& x) {
      return (*this = (long long)x);
    }

    operator long long() const {
      int width = hi - lo + 1;
      unsigned long long uv = (unsigned long long)parent.v >> lo;

      if (width >= 64) {
        return (long long)uv;
      }

      unsigned long long mask = (1ULL << width) - 1ULL;
      return (long long)(uv & mask);
    }
  };

  range_ref operator()(int hi, int lo) {
    return range_ref(*this, hi, lo);
  }

  long long operator()(int hi, int lo) const {
    int width = hi - lo + 1;
    unsigned long long uv = (unsigned long long)v >> lo;

    if (width >= 64) {
      return (long long)uv;
    }

    unsigned long long mask = (1ULL << width) - 1ULL;
    return (long long)(uv & mask);
  }
};

template<int W>
using ap_uint = ap_int_base<W, false>;

template<int W>
using ap_int = ap_int_base<W, true>;

#endif
