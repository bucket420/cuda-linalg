#include <stdio.h>
#include <random>
#include <time.h>
#include "LinearAlgebra.h"

Matrix::Matrix(int width, int height) : width(width), height(height) {
    this->data = (float*) calloc(this->width * this->height, sizeof(float));
}

Matrix::~Matrix() {
    free(this->data);
}

void Matrix::print() {
    if (this->width > 16 || this->height > 16) {
        printf("Matrix too large to print\n");
        return;
    }
    for (int i = 0; i < this->height; i++) {
        printf("[");
        for (int j = 0; j < this->width; j++) {
            printf("%7.2f", this->data[i * this->width + j]);
            if (j < this->width - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    }
}

void Matrix::pad(int blockSize) {
    int new_width = this->width, new_height = this->height;
    if (this->width % blockSize == 0 && this->height % blockSize == 0) {
        return;
    }
    if (this->width % blockSize != 0) {
        new_width = (this->width / blockSize + 1) * blockSize;
    }
    if (this->height % blockSize != 0) {
        new_height = (this->height / blockSize + 1) * blockSize;
    }
    float *new_data = (float*) calloc(new_width * new_height, sizeof(float));
    for (int i = 0; i < this->height; i++) {
        memcpy(new_data + i * new_width, this->data + i * this->width, this->width * sizeof(float));
    }
    free(this->data);
    this->padded_width = new_width - this->width;
    this->padded_height = new_height - this->height;
    this->data = new_data;
    this->width = new_width;
    this->height = new_height;
}

void Matrix::unpad() {
    if (this->padded_width == 0 && this->padded_height == 0) {
        return;
    }
    float *new_data = (float*) calloc((this->width - this->padded_width) * (this->height - this->padded_height), sizeof(float));
    for (int i = 0; i < this->height - this->padded_height; i++) {
        memcpy(new_data + i * (this->width - this->padded_width), this->data + i * this->width, (this->width - this->padded_width) * sizeof(float));
    }
    free(this->data);
    this->data = new_data;
    this->width -= this->padded_width;
    this->height -= this->padded_height;
    this->padded_width = 0;
    this->padded_height = 0;
}

void Matrix::zero() {
    memset(this->data, 0, this->width * this->height * sizeof(float));
}

void Matrix::transpose() {
    float *new_data = (float*) calloc(this->width * this->height, sizeof(float));
    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            new_data[j * this->height + i] = this->data[i * this->width + j];
        }
    }
    free(this->data);
    this->data = new_data;
    int temp = this->width;
    this->width = this->height;
    this->height = temp;
}

IdentityMatrix::IdentityMatrix(int size) : Matrix(size, size) {
    for (int i = 0; i < size; i++) {
        this->data[i * size + i] = 1.0f;
    }
}

Vector::Vector(int size) {
    this->size = size;
    this->data = (float*) calloc(this->size, sizeof(float));
}

Vector::~Vector() {
    free(this->data);
}

void Vector::pad(int blockSize) {
    int new_size;
    if (this->size % blockSize == 0) {
        return;
    }
    new_size = (this->size / blockSize + 1) * blockSize;
    float *new_data = (float*) calloc(new_size, sizeof(float));
    memcpy(new_data, this->data, this->size * sizeof(float));
    free(this->data);
    this->padded_size = new_size - this->size;
    this->data = new_data;
    this->size = new_size;
}

void Vector::unpad() {
    if (this->padded_size == 0) {
        return;
    }
    float *new_data = (float*) calloc(this->size - this->padded_size, sizeof(float));
    memcpy(new_data, this->data, (this->size - this->padded_size) * sizeof(float));
    free(this->data);
    this->data = new_data;
    this->size -= this->padded_size;
    this->padded_size = 0;
}

void Vector::zero() {
    memset(this->data, 0, this->size * sizeof(float));
}

void matrixMul(Matrix* mxn, Matrix *nxp, Matrix* mxp) {
    for (int i = 0; i < mxp->height; i++) {
        for (int j = 0; j < mxp->width; j++) {
            float sum = 0.0f;
            for (int k = 0; k < mxn->width; k++) {
                sum += mxn->data[i * mxn->width + k] * nxp->data[k * nxp->width + j];
            }
            mxp->data[i * mxp->width + j] = sum;
        }
    }
}

__global__ void matrixMulSharedMemoryKernel(const float *mxn, const float *nxp, float *mxp, 
                                            int n, int p, int m, int paddedN, int paddedP, int paddedM) {
    extern __shared__ float cache[];
    float *As = cache;
    float *Bs = &cache[blockDim.x * blockDim.x];
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < paddedP; idx += blockDim.x * gridDim.x) {
        for (int idy = threadIdx.y + blockDim.y * blockIdx.y; idy < paddedM; idy += blockDim.y * gridDim.y) {
            float temp = 0;
            for (int i = 0; i < paddedN; i++) {
                if (i * blockDim.x + threadIdx.x < n && idy < m) {
                    As[threadIdx.y * blockDim.x + threadIdx.x] = mxn[idy * n + (i * blockDim.x + threadIdx.x)];
                } else {
                    As[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
                }
                if (i * blockDim.x + threadIdx.y < n && idx < p) {
                    Bs[threadIdx.y * blockDim.x + threadIdx.x] = nxp[(i * blockDim.x + threadIdx.y) * p + idx];
                } else {
                    Bs[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
                }
                __syncthreads();
                for (int k = 0; k < blockDim.x; k++) {
                    temp += As[threadIdx.y * blockDim.x + k] * Bs[k * blockDim.x + threadIdx.x];
                }
                __syncthreads();
            }
            if (idx < p && idy < m)
                mxp[idy * p + idx] = temp;
        }
    }
}

__global__ void matrixMulKernel(const float *mxn, const float *nxp, float *mxp, int n, int p, int m) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < p; idx += blockDim.x * gridDim.x) {
        for (int idy = threadIdx.y + blockDim.y * blockIdx.y; idy < m; idy += blockDim.y * gridDim.y) {
            float temp = 0;
            for (int i = 0; i < n; i++) 
                temp += mxn[idy * n + i] * nxp[i * p + idx];   // dot product of row and column
            mxp[idy * p + idx] += temp;
        }
    }
}

void matrixMulCUDA(Matrix* mxn, Matrix *nxp, Matrix* mxp, int blockSize, dim3 grid, bool useSharedMemory) {
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
    if (useSharedMemory) {
        matrixMulSharedMemoryKernel<<<grid, block, 2 * blockSize * blockSize * sizeof(float)>>>(dmxn, dnxp, dmxp, mxn->width, mxp->width, mxp->height, paddedN, paddedP, paddedM);
    } else {
        matrixMulKernel<<<grid, block>>>(dmxn, dnxp, dmxp, mxn->width, mxp->width, mxp->height);
    }
    cudaMemcpy(mxp->data, dmxp, mxp->width * mxp->height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dmxn);
    cudaFree(dnxp);
    cudaFree(dmxp);
}

float dot(Vector *a, Vector *b) {
    float sum = 0.0f;
    for (int i = 0; i < a->size; i++) {
        sum += a->data[i] * b->data[i];
    }
    return sum;
}

__global__ void dotKernel(const float *a, const float *b, float *c, int size) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
        c[idx] = a[idx] * b[idx];
    }
}

float dotCUDA(Vector *a, Vector *b, int blockSize, int numBlocks, bool useSharedMemory) {
    return 0.0;
}