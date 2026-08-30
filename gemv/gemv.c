#define M 32
#define N 32

void gemv() {
    float A[N][M];
    float x[M];
    float y[N];

    for (int i = 0; i < N; i++) {
        float acc = 0;
        for (int j = 0; j < M; j++) {
            acc += A[i][j] * x[j];
        }
        y[i] = acc;
    }
}
