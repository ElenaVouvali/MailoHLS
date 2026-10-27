#ifndef MAILOHLS_AP_INT_STUB
#define MAILOHLS_AP_INT_STUB
// AP_INT_STUB: structural-analysis surrogate, never use for Vitis synthesis.
// This subset supports unsigned widths 1..64, increment and bit slices.
// Intermediate arithmetic and lowered widths are not vendor AP semantics.
#include <cstdint>
template<int W> struct ap_uint {
  static_assert(W > 0 && W <= 64, "Analysis surrogate supports widths 1..64 only");
  static constexpr std::uint64_t mask = W == 64 ? ~std::uint64_t(0) : (std::uint64_t(1) << W) - 1;
  static std::uint64_t range_mask(int high, int low) {
    const int bits = high - low + 1;
    return bits == 64 ? ~std::uint64_t(0) : (std::uint64_t(1) << bits) - 1;
  }
  std::uint64_t value;
  ap_uint() = default;
  ap_uint(std::uint64_t x):value(x & mask) {}
  ap_uint& operator=(std::uint64_t x){value=x & mask;return *this;}
  operator std::uint64_t() const {return value;}
  ap_uint& operator++(){value=(value+1)&mask;return *this;}
  ap_uint operator++(int){ap_uint previous=*this;++*this;return previous;}
  struct range_proxy {
    std::uint64_t *storage;
    int high, low;
    operator std::uint64_t() const {return (*storage >> low) & range_mask(high, low);}
    range_proxy& operator=(std::uint64_t x){
      std::uint64_t field_mask=range_mask(high, low) << low;
      *storage=((*storage & ~field_mask) | ((x << low) & field_mask)) & mask;
      return *this;
    }
  };
  range_proxy operator()(int high,int low){return range_proxy{&value,high,low};}
  std::uint64_t operator()(int high,int low) const {return (value >> low) & range_mask(high, low);}
};
#endif
