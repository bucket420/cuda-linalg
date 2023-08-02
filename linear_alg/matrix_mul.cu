#include <stdio.h>
#include <time.h>


const int DSIZE = 8;
const int block_size = 4; // CUDA maximum is 1024 *total* threads in block
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


// matrix multiply (naive) kernel: C = A * B
__global__ void mmul_kernel(const float *A, const float *B, float *C, int ds)
{

    // declare cache in shared memory
    __shared__ float As[block_size][block_size];
    __shared__ float Bs[block_size][block_size];

    int idx = threadIdx.x + blockDim.x * blockIdx.x; // create thread x index
    int idy = threadIdx.y + blockDim.y * blockIdx.y; // create thread y index

    if ((idx < ds) && (idy < ds))
    {
        float temp = 0;
        for (int i = 0; i < ds / block_size; i++)
        {

            // Load data into shared memory
            As[threadIdx.y][threadIdx.x] = A[idy * ds + (i * block_size + threadIdx.x)];
            Bs[threadIdx.y][threadIdx.x] = B[(i * block_size + threadIdx.y) * ds + idx];

            // Synchronize
            __syncthreads();

            // Keep track of the running sum
            for (int k = 0; k < block_size; k++)
                temp += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // dot product of row and column
            __syncthreads();
        }

        // Write to global memory
        C[idy * ds + idx] = temp;
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
    dim3 grid((C->width + block.x - 1) / block.x, (C->height + block.y - 1) / block.y);
    mmul_kernel<<<grid, block>>>(d_A, d_B, d_C, C->width);
    cudaMemcpy(C->data, d_C, C->width * C->height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    // these are just for timing
    clock_t t0, t1, t2;
    double t1sum = 0.0;
    double t2sum = 0.0;

    // start timing
    t0 = clock();

    // create matrices
    Matrix *A = new Matrix(DSIZE, DSIZE);
    Matrix *B = new Matrix(DSIZE, DSIZE);
    Matrix *C = new Matrix(DSIZE, DSIZE);
    Matrix *C2 = new Matrix(DSIZE, DSIZE);

    // initialize matrices
    for (int i = 0; i < A->height; i++) {
        for (int j = 0; j < A->width; j++) {
            A->data[i * A->width + j] = i + j * 0.1;
        }
    }

    for (int i = 0; i < B->height; i++) {
        for (int j = 0; j < B->width; j++) {
            B->data[i * B->width + j] = i + j * 0.2;
        }
    }

    // print input
    printf("Matrix A:\n");
    A->print();
    printf("Matrix B:\n");
    B->print();

    // multiply
    mmul(A, B, C);
    mmul_sequential(A, B, C2);

    // print output
    printf("Matrix C:\n");
    C->print();
    printf("Matrix C2:\n");
    C2->print();

    // GPU timing
    // t2 = clock();
    // t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    // printf("Done. Compute took %f seconds\n", t2sum);
}
