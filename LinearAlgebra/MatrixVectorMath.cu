#include <stdio.h>
#include <random>
#include <time.h>
#include "LinearAlgebra.h"

void matrixVectorMul(Matrix* mxn, Vector* nx1, Vector* mx1) {
    for (int i = 0; i < mxn->height; i++) {
        float sum = 0.0f;
        for (int j = 0; j < mxn->width; j++) {
            sum += mxn->data[i * mxn->width + j] * nx1->data[j];
        }
        mx1->data[i] = sum;
    }
}

__global__ void matrixVectorMulGlobalMemoryKernel(const float* mxn, const float* nx1, float* mx1, int m, int n) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < m; idx += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += mxn[idx * n + j] * nx1[j];
        }
        mx1[idx] = sum;
    }
}

__global__ void matrixVectorMulSharedMemoryKernelNoPadding(const float* mxn, const float* nx1, float* mx1, int m, int n, int paddedN, int sharedMemorySize) {
    extern __shared__ float nx1s[];
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < m; idx += blockDim.x * gridDim.x) {
        float temp = 0.0f;
        for (int j = 0; j < paddedN / sharedMemorySize; j++) {
            // for (int k = threadIdx.x; k < sharedMemorySize; k += blockDim.x) {
            //     if (k + j * sharedMemorySize < n) {
            //         nx1s[k] = nx1[k + j * sharedMemorySize];
            //     } else {
            //         nx1s[k] = 0.0f;
            //     }
            // }
            if (threadIdx.x == 0) {
                for (int k = 0; k < sharedMemorySize; k++) {
                    if (k + j * sharedMemorySize < n) {
                        nx1s[k] = nx1[k + j * sharedMemorySize];
                    } else {
                        nx1s[k] = 0.0f;
                    }
                }
            }
            __syncthreads();
            for (int k = 0; k < sharedMemorySize; k++) {
                temp += mxn[idx * n + k + j * sharedMemorySize] * nx1s[k];
            }
            __syncthreads();
        }
        mx1[idx] = temp;
    }
}

void matrixVectorMulCUDAGlobalMemory(Matrix* mxn, Vector* nx1, Vector* mx1, int blockSize, int numBlocks) {
    float *dmxn, *dnx1, *dmx1;
    cudaMalloc(&dmxn, mxn->width * mxn->height * sizeof(float));
    cudaMalloc(&dnx1, nx1->size * sizeof(float));
    cudaMalloc(&dmx1, mx1->size * sizeof(float));
    cudaMemcpy(dmxn, mxn->data, mxn->width * mxn->height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dnx1, nx1->data, nx1->size * sizeof(float), cudaMemcpyHostToDevice);
    matrixVectorMulGlobalMemoryKernel<<<numBlocks, blockSize>>>(dmxn, dnx1, dmx1, mxn->height, mxn->width);
    cudaMemcpy(mx1->data, dmx1, mx1->size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dmxn);
    cudaFree(dnx1);
    cudaFree(dmx1);
}

void matrixVectorMulCUDASharedMemoryNoPadding(Matrix* mxn, Vector* nx1, Vector* mx1, int blockSize, int numBlocks, int sharedMemorySize) {
    float *dmxn, *dnx1, *dmx1;
    int paddedN = mxn->width;
    if (mxn->width % sharedMemorySize != 0) {
        paddedN = mxn->width + sharedMemorySize - mxn->width % sharedMemorySize;
    }
    cudaMalloc(&dmxn, mxn->width * mxn->height * sizeof(float));
    cudaMalloc(&dnx1, nx1->size * sizeof(float));
    cudaMalloc(&dmx1, mx1->size * sizeof(float));
    cudaMemcpy(dmxn, mxn->data, mxn->width * mxn->height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dnx1, nx1->data, nx1->size * sizeof(float), cudaMemcpyHostToDevice);
    matrixVectorMulSharedMemoryKernelNoPadding<<<numBlocks, blockSize, sharedMemorySize * sizeof(float)>>>(dmxn, dnx1, dmx1, mxn->height, mxn->width, paddedN, sharedMemorySize);
    cudaMemcpy(mx1->data, dmx1, mx1->size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dmxn);
    cudaFree(dnx1);
    cudaFree(dmx1);
}

void matrixVectorMulCUDA(Matrix* mxn, Vector* nx1, Vector* mx1, int blockSize, int numBlocks, int mode, int sharedMemorySize) {
    switch (mode) {
        case 0:
            matrixVectorMulCUDAGlobalMemory(mxn, nx1, mx1, blockSize, numBlocks);
            break;
        case 1:
            matrixVectorMulCUDASharedMemoryNoPadding(mxn, nx1, mx1, blockSize, numBlocks, sharedMemorySize);
            break;  
    }
}
