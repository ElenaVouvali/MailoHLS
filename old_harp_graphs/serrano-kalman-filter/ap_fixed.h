#ifndef AP_FIXED_STUB_H
#define AP_FIXED_STUB_H

enum { AP_RND = 0 };

// We don't care about true fixed-point semantics for control-flow analysis.
// Treat ap_fixed<> as just a float.
template<int W, int I, int Q = 0>
using ap_fixed = float;

#endif // AP_FIXED_STUB_H

