#include <stdio.h>
#include <random>
#include <time.h>
#include "LinearAlgebra.h"

void matrixMul(Matrix* mxn, Matrix *nxp, Matrix* mxp) {
    for (int i = 0; i < mxp->height; i++) {
        for (int j = 0; j < mxp->width; j++) {
            float sum = 0.0;
            for (int k = 0; k < mxn->width; k++) {
                sum += mxn->data[i * mxn->width + k] * nxp->data[k * nxp->width + j];
            }
            mxp->data[i * mxp->width + j] = sum;
        }
    }
}

__global__ void matrixMulSharedMemoryKernelNoPadding(const float *mxn, const float *nxp, float *mxp, 
                                            int n, int p, int m, int paddedN, int paddedP, int paddedM) {
    extern __shared__ float cache[];
    float *mxns = cache;
    float *nxps = &cache[blockDim.x * blockDim.x];
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < paddedP; idx += blockDim.x * gridDim.x) {
        for (int idy = threadIdx.y + blockDim.y * blockIdx.y; idy < paddedM; idy += blockDim.y * gridDim.y) {
            float temp = 0;
            for (int i = 0; i < paddedN / blockDim.x; i++) {
                if (i * blockDim.x + threadIdx.x < n && idy < m) {
                    mxns[threadIdx.y * blockDim.x + threadIdx.x] = mxn[idy * n + (i * blockDim.x + threadIdx.x)];
                } else {
                    mxns[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
                }
                if (i * blockDim.x + threadIdx.y < n && idx < p) {
                    nxps[threadIdx.y * blockDim.x + threadIdx.x] = nxp[(i * blockDim.x + threadIdx.y) * p + idx];
                } else {
                    nxps[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
                }
                __syncthreads();
                for (int k = 0; k < blockDim.x; k++) {
                    temp += mxns[threadIdx.y * blockDim.x + k] * nxps[k * blockDim.x + threadIdx.x];
                }
                __syncthreads();
            }
            if (idx < p && idy < m)
                mxp[idy * p + idx] = temp;
        }
    }
}

__global__ void matrixMulSharedMemoryKernelWithPadding(const float *mxn, const float *nxp, float *mxp, int n, int p, int m) {
    extern __shared__ float cache[];
    float *mxns = cache;
    float *nxps = &cache[blockDim.x * blockDim.x];
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < p; idx += blockDim.x * gridDim.x) {
        for (int idy = threadIdx.y + blockDim.y * blockIdx.y; idy < m; idy += blockDim.y * gridDim.y) {
            float temp = 0;
            for (int i = 0; i < n / blockDim.x; i++) {
                mxns[threadIdx.y * blockDim.x + threadIdx.x] = mxn[idy * n + (i * blockDim.x + threadIdx.x)];
                nxps[threadIdx.y * blockDim.x + threadIdx.x] = nxp[(i * blockDim.x + threadIdx.y) * p + idx];

                __syncthreads();
                for (int k = 0; k < blockDim.x; k++) {
                    temp += mxns[threadIdx.y * blockDim.x + k] * nxps[k * blockDim.x + threadIdx.x];
                }
                __syncthreads();
            }
            mxp[idy * p + idx] = temp;
        }
    }
}

__global__ void matrixMulGlobalMemoryKernel(const float *mxn, const float *nxp, float *mxp, int n, int p, int m) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < p; idx += blockDim.x * gridDim.x) {
        for (int idy = threadIdx.y + blockDim.y * blockIdx.y; idy < m; idy += blockDim.y * gridDim.y) {
            float temp = 0;
            for (int i = 0; i < n; i++) 
                temp += mxn[idy * n + i] * nxp[i * p + idx];
            mxp[idy * p + idx] = temp;
        }
    }
}

void matrixMulCUDAGlobalMemory(Matrix* mxn, Matrix *nxp, Matrix* mxp, int blockSize, dim3 grid) {
    float *dmxn, *dnxp, *dmxp;
    cudaMalloc(&dmxn, mxn->width * mxn->height * sizeof(float));
    cudaMalloc(&dnxp, nxp->width * nxp->height * sizeof(float));
    cudaMalloc(&dmxp, mxp->width * mxp->height * sizeof(float));
    cudaMemcpy(dmxn, mxn->data, mxn->width * mxn->height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dnxp, nxp->data, nxp->width * nxp->height * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(blockSize, blockSize); 
    matrixMulGlobalMemoryKernel<<<grid, block>>>(dmxn, dnxp, dmxp, mxn->width, mxp->width, mxp->height);
    cudaMemcpy(mxp->data, dmxp, mxp->width * mxp->height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dmxn);
    cudaFree(dnxp);
    cudaFree(dmxp);
}

void matrixMulCUDASharedMemoryNoPadding(Matrix* mxn, Matrix *nxp, Matrix* mxp, int blockSize, dim3 grid) {
    int paddedN = mxn->width, paddedP = nxp->width, paddedM = mxp->height;
    if (mxn->width % blockSize != 0) {
        paddedN = (mxn->width / blockSize + 1) * blockSize;
    }
    if (nxp->width % blockSize != 0) {
        paddedP = (nxp->width / blockSize + 1) * blockSize;
    }
    if (mxp->height % blockSize != 0) {
        paddedM = (mxp->height / blockSize + 1) * blockSize;
    }
    float *dmxn, *dnxp, *dmxp;
    cudaMalloc(&dmxn, mxn->width * mxn->height * sizeof(float));
    cudaMalloc(&dnxp, nxp->width * nxp->height * sizeof(float));
    cudaMalloc(&dmxp, mxp->width * mxp->height * sizeof(float));
    cudaMemcpy(dmxn, mxn->data, mxn->width * mxn->height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dnxp, nxp->data, nxp->width * nxp->height * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(blockSize, blockSize); 
    matrixMulSharedMemoryKernelNoPadding<<<grid, block, 2 * blockSize * blockSize * sizeof(float)>>>(dmxn, dnxp, dmxp, mxn->width, mxp->width, mxp->height, paddedN, paddedP, paddedM);
    cudaMemcpy(mxp->data, dmxp, mxp->width * mxp->height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dmxn);
    cudaFree(dnxp);
    cudaFree(dmxp);
}

void matrixMulCUDASharedMemoryWithPadding(Matrix* mxn, Matrix *nxp, Matrix* mxp, int blockSize, dim3 grid) {
    mxn->pad(blockSize);
    nxp->pad(blockSize);
    mxp->pad(blockSize);
    float *dmxn, *dnxp, *dmxp;
    cudaMalloc(&dmxn, mxn->width * mxn->height * sizeof(float));
    cudaMalloc(&dnxp, nxp->width * nxp->height * sizeof(float));
    cudaMalloc(&dmxp, mxp->width * mxp->height * sizeof(float));
    cudaMemcpy(dmxn, mxn->data, mxn->width * mxn->height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dnxp, nxp->data, nxp->width * nxp->height * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(blockSize, blockSize); 
    matrixMulSharedMemoryKernelWithPadding<<<grid, block, 2 * blockSize * blockSize * sizeof(float)>>>(dmxn, dnxp, dmxp, mxn->width, mxp->width, mxp->height);
    cudaMemcpy(mxp->data, dmxp, mxp->width * mxp->height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dmxn);
    cudaFree(dnxp);
    cudaFree(dmxp);
}

void matrixMulCUDA(Matrix* mxn, Matrix *nxp, Matrix* mxp, int blockSize, dim3 grid, int mode) {
    switch (mode) {
        case 0:
            matrixMulCUDAGlobalMemory(mxn, nxp, mxp, blockSize, grid);
            break;
        case 1:
            matrixMulCUDASharedMemoryNoPadding(mxn, nxp, mxp, blockSize, grid);
            break;
        case 2:
            matrixMulCUDASharedMemoryWithPadding(mxn, nxp, mxp, blockSize, grid);
            break;
    }
}