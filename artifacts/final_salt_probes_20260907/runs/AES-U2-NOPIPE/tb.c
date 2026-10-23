#include <stdio.h>
int chstone_aes(void);
int main(void) { int r=chstone_aes(); printf("AES errors=%d\n",r); return r!=0; }
