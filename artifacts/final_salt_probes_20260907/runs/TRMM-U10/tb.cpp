#include <cmath>
#include <cstdio>
extern "C" void kernel_trmm(double,double[60][60],double[60][80]);
int main(){static double a[60][60],b[60][80],ref[60][80];for(int i=0;i<60;i++){for(int k=0;k<60;k++)a[i][k]=((i+k)%7+1)/100.0;for(int j=0;j<80;j++)b[i][j]=ref[i][j]=((i*3+j)%13+1)/20.0;}
for(int i=0;i<60;i++)for(int j=0;j<80;j++){double s=ref[i][j];for(int k=i+1;k<60;k++)s+=a[k][i]*ref[k][j];ref[i][j]=1.25*s;}
kernel_trmm(1.25,a,b);int errors=0;for(int i=0;i<60;i++)for(int j=0;j<80;j++)if(!std::isfinite(b[i][j])||fabs(b[i][j]-ref[i][j])>1e-8)errors++;printf("TRMM mismatches=%d\n",errors);return errors!=0;}
