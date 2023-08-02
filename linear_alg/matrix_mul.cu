#include <stdio.h>
#include <random>
#include <time.h>


// const int DSIZE = 8;
// CUDA maximum is 1024 *total* threads in block
// const float A_val = 1.0f;
// const float B_val = 2.0f;
const int block_size = 2;

class Matrix {
public:
    float *data;
    int width;
    int height;

    Matrix(int width, int height) : width(width), height(height) {
        this->data = (float*) calloc((size_t) this->width * this->height, sizeof(float));
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

    void zero_pad(int block_size) {
        int padded_width, padded_height;
        if (this->width % block_size == 0 && this->height % block_size == 0) {
            return;
        }
        if (this->width % block_size != 0) {
            padded_width = (this->width / block_size + 1) * block_size;
        }
        if (this->height % block_size != 0) {
            padded_height = (this->height / block_size + 1) * block_size;
        }
        float *padded_data = (float*) calloc((size_t) padded_width * padded_height, sizeof(float));
        for (int i = 0; i < this->height; i++) {
            memcpy(padded_data + i * padded_width, this->data + i * this->width, this->width * sizeof(float));
        }
        free(this->data);
        this->data = padded_data;
        this->width = padded_width;
        this->height = padded_height;
    }
};

__global__ void mmul_kernel(const float *A, const float *B, float *C, int A_width, int C_width, int C_height) {
    __shared__ float As[block_size][block_size];
    __shared__ float Bs[block_size][block_size];
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < C_width; idx += blockDim.x * gridDim.x) {
        for (int idy = threadIdx.y + blockDim.y * blockIdx.y; idy < C_height; idy += blockDim.y * gridDim.y) {
            float temp = 0;
            for (int i = 0; i < gridDim.x; i++) {
                As[threadIdx.y][threadIdx.x] = A[idy * A_width + (i * block_size + threadIdx.x)];
                Bs[threadIdx.y][threadIdx.x] = B[(i * block_size + threadIdx.y) * C_width + idx];
                __syncthreads();
                for (int k = 0; k < block_size; k++) {
                    temp += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
                __syncthreads();
            }
            C[idy * C_width + idx] += temp;
        }
    }
}

__global__ void mmul_kernel_not_shared(const float *A, const float *B, float *C, int A_width, int C_width, int C_height) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < C_width; idx += blockDim.x * gridDim.x) {
        for (int idy = threadIdx.y + blockDim.y * blockIdx.y; idy < C_height; idy += blockDim.y * gridDim.y) {
            float temp = 0;
            for (int i = 0; i < gridDim.x; i++) 
                temp += A[idy * A_width + i] * B[i * C_width + idx];   // dot product of row and column
            C[idy * C_width + idx] += temp;
        }
    }
}

void mmul_sequential(Matrix* A, Matrix *B, Matrix* C) {
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

void mmul(Matrix* A, Matrix *B, Matrix* C) {
    A->zero_pad(block_size);
    B->zero_pad(block_size);
    C->zero_pad(block_size);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A->width * A->height * sizeof(float));
    cudaMalloc(&d_B, B->width * B->height * sizeof(float));
    cudaMalloc(&d_C, C->width * C->height * sizeof(float));
    cudaMemcpy(d_A, A->data, A->width * A->height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B->data, B->width * B->height * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(block_size, block_size); 
    dim3 grid(4, 4);
    mmul_kernel<<<grid, block>>>(d_A, d_B, d_C, A->width, C->width, C->height);
    cudaMemcpy(C->data, d_C, C->width * C->height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void mmul_not_shared(Matrix* A, Matrix *B, Matrix* C) {
    A->zero_pad(block_size);
    B->zero_pad(block_size);
    C->zero_pad(block_size);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A->width * A->height * sizeof(float));
    cudaMalloc(&d_B, B->width * B->height * sizeof(float));
    cudaMalloc(&d_C, C->width * C->height * sizeof(float));
    cudaMemcpy(d_A, A->data, A->width * A->height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B->data, B->width * B->height * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(block_size, block_size); 
    dim3 grid(4, 4);
    mmul_kernel_not_shared<<<grid, block>>>(d_A, d_B, d_C, A->width, C->width, C->height);
    cudaMemcpy(C->data, d_C, C->width * C->height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 10.0);

    // these are just for timing
    clock_t t0, t1, t2;
    double t1sum = 0.0;
    double t2sum = 0.0;

    // start timing
    t0 = clock();

    // create matrices
    Matrix *A = new Matrix(5, 7);
    Matrix *B = new Matrix(7, 5);
    Matrix *C_sequential = new Matrix(7, 7);
    Matrix *C_CUDA_shared = new Matrix(7, 7);
    Matrix *C_CUDA_not_shared = new Matrix(7, 7);

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
    mmul(A, B, C_CUDA_shared);
    mmul_not_shared(A, B, C_CUDA_not_shared);
    mmul_sequential(A, B, C_sequential);

    // print output
    printf("Matrix C (sequential):\n");
    C_sequential->print();
    printf("Matrix C (CUDA shared):\n");
    C_CUDA_shared->print();
    printf("Matrix C (CUDA not shared):\n");
    C_CUDA_not_shared->print();


    // GPU timing
    // t2 = clock();
    // t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    // printf("Done. Compute took %f seconds\n", t2sum);
}
