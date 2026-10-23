#include <cmath>
#include <cstdio>
extern "C" void kernel_jacobi_2d(int,int,double[90][90],double[90][90]);
int main(){static double a[90][90],b[90][90],ra[90][90],rb[90][90];
for(int i=0;i<90;i++)for(int j=0;j<90;j++){a[i][j]=ra[i][j]=(i*7+j*3+1)/100.0;b[i][j]=rb[i][j]=(i+j+2)/200.0;}
for(int t=0;t<40;t++){for(int i=1;i<89;i++)for(int j=1;j<89;j++)rb[i][j]=0.2*(ra[i][j]+ra[i][j-1]+ra[i][j+1]+ra[i+1][j]+ra[i-1][j]);for(int i=1;i<89;i++)for(int j=1;j<89;j++)ra[i][j]=0.2*(rb[i][j]+rb[i][j-1]+rb[i][j+1]+rb[i+1][j]+rb[i-1][j]);}
kernel_jacobi_2d(40,90,a,b);int errors=0;for(int i=0;i<90;i++)for(int j=0;j<90;j++)if(!std::isfinite(a[i][j])||!std::isfinite(b[i][j])||fabs(a[i][j]-ra[i][j])>1e-8||fabs(b[i][j]-rb[i][j])>1e-8)errors++;
printf("Jacobi mismatches=%d\n",errors);return errors!=0;}
