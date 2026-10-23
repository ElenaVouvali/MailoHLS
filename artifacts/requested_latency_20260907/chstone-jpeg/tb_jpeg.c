#include <stdio.h>

int chstone_jpeg(void);

int main(void)
{
    int result = chstone_jpeg();

    if (result != 0) {
        fprintf(stderr, "CHStone JPEG FAILED: main_result=%d\n", result);
        return 1;
    }

    printf("CHStone JPEG PASSED\n");
    return 0;
}

