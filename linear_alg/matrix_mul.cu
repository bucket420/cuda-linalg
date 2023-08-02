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
    extern __shared__ float cache[];
    float *As = cache;
    float *Bs = &cache[blockDim.x * blockDim.x];
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < C_width; idx += blockDim.x * gridDim.x) {
        for (int idy = threadIdx.y + blockDim.y * blockIdx.y; idy < C_height; idy += blockDim.y * gridDim.y) {
            float temp = 0;
            for (int i = 0; i < gridDim.x; i++) {
                As[threadIdx.y * blockDim.x + threadIdx.x] = A[idy * A_width + (i * blockDim.x + threadIdx.x)];
                Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[(i * blockDim.x + threadIdx.y) * C_width + idx];
                __syncthreads();
                for (int k = 0; k < blockDim.x; k++) {
                    temp += As[threadIdx.y * blockDim.x + k] * Bs[k * blockDim.x + threadIdx.x];
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
            for (int i = 0; i < gridDim.x * blockDim.x; i++) 
                temp += A[idy * A_width + i] * B[i * C_width + idx];   // dot product of row and column
            C[idy * C_width + idx] += temp;
        }
    }
}

void mmul_sequential(Matrix* A, Matrix *B, Matrix* C) {
    for (int i = 0; i < C->height; i++) {
        for (int j = 0; j < C->width; j++) {
            for (int k = 0; k < A->width; k++) {
                C->data[i * C->width + j] += A->data[i * A->width + k] * B->data[k * B->width + j];
            }
        }
    }
}

void mmul(Matrix* A, Matrix *B, Matrix* C, int block_size, dim3 grid) {
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
    mmul_kernel<<<grid, block, 2 * block_size * block_size * sizeof(float)>>>(d_A, d_B, d_C, A->width, C->width, C->height);
    cudaMemcpy(C->data, d_C, C->width * C->height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void mmul_not_shared(Matrix* A, Matrix *B, Matrix* C, int block_size, dim3 grid) {
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
    mmul_kernel_not_shared<<<grid, block>>>(d_A, d_B, d_C, A->width, C->width, C->height);
    cudaMemcpy(C->data, d_C, C->width * C->height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    const int block_size = 16;
    dim3 grid(128, 128);
    int A_width = 10000;
    int C_width = 10000;
    int C_height = 10000;

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 10.0);

    // these are just for timing
    clock_t t0, t1, t2, t3;
    double t1sum = 0.0;
    double t2sum = 0.0;

    // create matrices
    Matrix *A = new Matrix(A_width, C_height);
    Matrix *B = new Matrix(C_width, A_width);
    // Matrix *C_sequential = new Matrix(C_width, C_height);
    Matrix *C = new Matrix(C_width, C_height);
    // Matrix *C_CUDA_not_shared = new Matrix(C_width, C_height);

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
    t0 = clock();
    mmul(A, B, C, block_size, grid);
    t1 = clock();
    printf("Matrix C (CUDA shared):\n");
    C->print();
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Compute took %f seconds (CUDA shared)\n", t1sum);



    t2 = clock();
    mmul_not_shared(A, B, C, block_size, grid);
    t3 = clock();
    printf("Matrix C (CUDA not shared):\n");
    C->print();
    t2sum = ((double)(t3 - t2)) / CLOCKS_PER_SEC;
    printf("Compute took %f seconds (CUDA not shared)\n", t2sum);
}
