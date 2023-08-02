#include <stdio.h>
#include <random>
#include <time.h>


// const int DSIZE = 8;
// CUDA maximum is 1024 *total* threads in block
// const float A_val = 1.0f;
// const float B_val = 2.0f;

class Matrix {
public:
    float *data;
    int width;
    int height;

    Matrix(int width, int height) : width(width), height(height) {
        this->data = (float*) calloc(this->width * this->height, sizeof(float));
    }

    ~Matrix() {
        free(this->data);
    }

    void print() {
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

    void zero_pad(int blockSize) {
        int padded_width, padded_height;
        if (this->width % blockSize == 0 && this->height % blockSize == 0) {
            return;
        }
        if (this->width % blockSize != 0) {
            padded_width = (this->width / blockSize + 1) * blockSize;
        }
        if (this->height % blockSize != 0) {
            padded_height = (this->height / blockSize + 1) * blockSize;
        }
        float *padded_data = (float*) calloc(padded_width * padded_height, sizeof(float));
        for (int i = 0; i < this->height; i++) {
            memcpy(padded_data + i * padded_width, this->data + i * this->width, this->width * sizeof(float));
        }
        free(this->data);
        this->data = padded_data;
        this->width = padded_width;
        this->height = padded_height;
    }

    void zero() {
        memset(this->data, 0, this->width * this->height * sizeof(float));
    }
};

__global__ void matrixMulSharedMemoryKernel(const float *A, const float *B, float *C, int widthA, int widthC, int heightC) {
    extern __shared__ float cache[];
    float *As = cache;
    float *Bs = &cache[blockDim.x * blockDim.x];
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < widthC; idx += blockDim.x * gridDim.x) {
        for (int idy = threadIdx.y + blockDim.y * blockIdx.y; idy < heightC; idy += blockDim.y * gridDim.y) {
            float temp = 0;
            for (int i = 0; i < gridDim.x; i++) {
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
            for (int i = 0; i < gridDim.x * blockDim.x; i++) 
                temp += A[idy * widthA + i] * B[i * widthC + idx];   // dot product of row and column
            C[idy * widthC + idx] += temp;
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

void matrixMul(Matrix* A, Matrix *B, Matrix* C, int blockSize, dim3 grid, int useSharedMemory) {
    A->zero_pad(blockSize);
    B->zero_pad(blockSize);
    C->zero_pad(blockSize);
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

int main()
{
    const int blockSize = 2;
    dim3 grid(4, 4);
    int widthA = 11;
    int widthC = 11;
    int heightC = 11;

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 10.0);

    // these are just for timing
    clock_t t0, t1, t2, t3, t4, t5;

    // create matrices
    Matrix *A = new Matrix(widthA, heightC);
    Matrix *B = new Matrix(widthC, widthA);
    Matrix *C = new Matrix(widthC, heightC);

    // initialize matrices
    for (int i = 0; i < A->height; i++) {
        for (int j = 0; j < A->width; j++) {
            A->data[i * A->width + j] = dis(gen);
        }
    }

    for (int i = 0; i < B->height; i++) {
        for (int j = 0; j < B->width; j++) {
            B->data[i * B->width + j] = dis(gen);
        }
    }

    // print input
    printf("Matrix A:\n");
    A->print();
    printf("Matrix B:\n");
    B->print();

    // multiply
    if (widthC <= 1000 && heightC <= 1000) {
        t4 = clock();
        matrixMulSequential(A, B, C);
        t5 = clock();
        printf("Matrix C (sequential):\n");
        C->print();
        printf("Compute took %f seconds (sequential)\n", ((double)(t5 - t4)) / CLOCKS_PER_SEC);
    }
    C->zero();

    t0 = clock();
    matrixMul(A, B, C, blockSize, grid, 1);
    t1 = clock();
    printf("Matrix C (CUDA shared):\n");
    C->print();
    printf("Compute took %f seconds (CUDA shared)\n", ((double)(t1 - t0)) / CLOCKS_PER_SEC);

    C->zero();

    t2 = clock();
    matrixMul(A, B, C, blockSize, grid, 0);
    t3 = clock();
    printf("Matrix C (CUDA not shared):\n");
    C->print();
    printf("Compute took %f seconds (CUDA not shared)\n", ((double)(t3 - t2)) / CLOCKS_PER_SEC);

    delete A;
    delete B;
    delete C;
}
