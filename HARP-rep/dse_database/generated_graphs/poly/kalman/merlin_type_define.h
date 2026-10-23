
#ifndef __MERLIN_TYPE_DEFINE_H__

#define __MERLIN_TYPE_DEFINE_H__
typedef class ap_uint< 256 > INTERFACE_WIDTH;
typedef class ap_fixed< 19 , 13 , AP_RND , AP_WRAP , 0 > DTYPE;
typedef union {int raw_val;float float_val;}raw_float;

#endif
