#include <cmath>
#include <cstdio>
extern "C" void kernel_syr2k(double,double,double[80][80],double[80][60],double[80][60]);
int main(){static double a[80][60],b[80][60],c[80][80],ref[80][80];for(int i=0;i<80;i++){for(int k=0;k<60;k++){a[i][k]=((i+k)%11+1)/100.0;b[i][k]=((i*3+k)%7+1)/100.0;}for(int j=0;j<80;j++)c[i][j]=ref[i][j]=(i+j+1)/200.0;}
for(int i=0;i<80;i++){for(int j=0;j<=i;j++)ref[i][j]*=0.75;for(int k=0;k<60;k++)for(int j=0;j<=i;j++)ref[i][j]+=a[j][k]*1.25*b[i][k]+b[j][k]*1.25*a[i][k];}
kernel_syr2k(1.25,0.75,c,a,b);int errors=0;for(int i=0;i<80;i++)for(int j=0;j<80;j++)if(!std::isfinite(c[i][j])||fabs(c[i][j]-ref[i][j])>1e-8)errors++;printf("SYR2K mismatches=%d\n",errors);return errors!=0;}
