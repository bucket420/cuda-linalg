#include <stdio.h>
#include <random>
#include <time.h>
#include "LinearAlgebra.h"


// const int DSIZE = 8;
// CUDA maximum is 1024 *total* threads in block
// const float A_val = 1.0f;
// const float B_val = 2.0f;


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
    int new_width, new_height;
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




__global__ void matrixMulSharedMemoryKernel(const float *A, const float *B, float *C, int widthA, int widthC, int heightC) {
    extern __shared__ float cache[];
    float *As = cache;
    float *Bs = &cache[blockDim.x * blockDim.x];
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < widthC; idx += blockDim.x * gridDim.x) {
        for (int idy = threadIdx.y + blockDim.y * blockIdx.y; idy < heightC; idy += blockDim.y * gridDim.y) {
            float temp = 0;
            for (int i = 0; i < widthA / blockDim.x; i++) {
                As[threadIdx.y * blockDim.x + threadIdx.x] = A[idy * widthA + (i * blockDim.x + threadIdx.x)];
                Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[(i * blockDim.x + threadIdx.y) * widthC + idx];
                __syncthreads();
                for (int k = 0; k < blockDim.x; k++) {
                    temp += As[threadIdx.y * blockDim.x + k] * Bs[k * blockDim.x + threadIdx.x];
                }
                __syncthreads();
            }
            C[idy * widthC + idx] += temp;
        }
    }
}

__global__ void matrixMulKernel(const float *A, const float *B, float *C, int widthA, int widthC, int heightC) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < widthC; idx += blockDim.x * gridDim.x) {
        for (int idy = threadIdx.y + blockDim.y * blockIdx.y; idy < heightC; idy += blockDim.y * gridDim.y) {
            float temp = 0;
            for (int i = 0; i < widthA; i++) 
                temp += A[idy * widthA + i] * B[i * widthC + idx];   // dot product of row and column
            C[idy * widthC + idx] = temp;
        }
    }
}

void matrixMulSequential(Matrix* A, Matrix *B, Matrix* C) {
    for (int i = 0; i < C->height; i++) {
        for (int j = 0; j < C->width; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A->width; k++) {
                sum += A->data[i * A->width + k] * B->data[k * B->width + j];
            }
            C->data[i * C->width + j] = sum;
        }
    }
}

void matrixMulCUDA(Matrix* A, Matrix *B, Matrix* C, int blockSize, dim3 grid, bool useSharedMemory) {
    A->pad(blockSize);
    B->pad(blockSize);
    C->pad(blockSize);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, A->width * A->height * sizeof(float));
    cudaMalloc(&dB, B->width * B->height * sizeof(float));
    cudaMalloc(&dC, C->width * C->height * sizeof(float));
    cudaMemcpy(dA, A->data, A->width * A->height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B->data, B->width * B->height * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(blockSize, blockSize); 
    if (useSharedMemory) {
        matrixMulSharedMemoryKernel<<<grid, block, 2 * blockSize * blockSize * sizeof(float)>>>(dA, dB, dC, A->width, C->width, C->height);
    } else {
        matrixMulKernel<<<grid, block>>>(dA, dB, dC, A->width, C->width, C->height);
    }
    cudaMemcpy(C->data, dC, C->width * C->height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}