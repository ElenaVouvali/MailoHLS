#include "ap_int.h"
#include <cstdio>
extern "C" void krnl_KALMAN(ap_uint<256>*,ap_uint<256>*);
extern "C" void reference_krnl_KALMAN(ap_uint<256>*,ap_uint<256>*);
int main(){static ap_uint<256> in[32768],out[32768],ref[32768];union {float f;unsigned u;} v;
for(int i=0;i<32768;i++){in[i]=0;out[i]=ref[i]=0;for(int j=0;j<8;j++){v.f=((i*8+j)%127)/128.0f;in[i].range(j*32+31,j*32)=v.u;}}
reference_krnl_KALMAN(in,ref);krnl_KALMAN(in,out);int errors=0;for(int i=0;i<32768;i++)if(out[i]!=ref[i])errors++;printf("Kalman mismatches=%d\n",errors);return errors!=0;}
