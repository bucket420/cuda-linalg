#include <stdio.h>
#include <random>
#include <time.h>
#include "LinearAlgebra.h"

float dot(Vector *a, Vector *b) {
    float sum = 0.0f;
    for (int i = 0; i < a->size; i++) {
        sum += a->data[i] * b->data[i];
    }
    return sum;
}

__global__ void dotKernel(const float *a, const float *b, float *c, int size) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
        atomicAdd(c, a[idx] * b[idx]);
    }
}

float dotCUDA(Vector *a, Vector *b, int blockSize, int numBlocks) {
    float c = 0.0f;
    float *da, *db, *dc;
    cudaMalloc(&da, a->size * sizeof(float));
    cudaMalloc(&db, b->size * sizeof(float));
    cudaMalloc(&dc, sizeof(float));
    cudaMemcpy(da, a->data, a->size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b->data, b->size * sizeof(float), cudaMemcpyHostToDevice);
    dotKernel<<<numBlocks, blockSize>>>(da, db, dc, a->size);
    cudaMemcpy(&c, dc, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return c;
}