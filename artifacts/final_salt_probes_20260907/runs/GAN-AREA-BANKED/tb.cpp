#include <cmath>
#include <cstdio>
#include "network.h"
void reference_forward_propagation(float*,float*);
int main(){float x[392],y[392],ref[392];for(int i=0;i<392;i++){x[i]=((i*7)%31-15)/64.0f;y[i]=ref[i]=0;}
reference_forward_propagation(x,ref);forward_propagation(x,y);int errors=0;for(int i=0;i<392;i++)if(!std::isfinite(y[i])||fabs(y[i]-ref[i])>1e-6)errors++;printf("GAN mismatches=%d\n",errors);return errors!=0;}
