#include <stdio.h>
#include <random>
#include <time.h>
#include "LinearAlgebra.h"


// const int DSIZE = 8;
// CUDA maximum is 1024 *total* threads in block
// const float A_val = 1.0f;
// const float B_val = 2.0f;

int main()
{
    const int blockSize = 64;
    dim3 grid(512, 512);
    int widthA = 10000;
    int widthC = 10000;
    int heightC = 10000;

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
        C->unpad();
        C->print();
        printf("Compute took %f seconds (sequential)\n", ((double)(t5 - t4)) / CLOCKS_PER_SEC);
    }
    C->zero();
    C->pad(blockSize);

    t0 = clock();
    matrixMulCUDA(A, B, C, blockSize, grid, true);
    t1 = clock();
    printf("Matrix C (CUDA with shared memory):\n");
    C->unpad();
    C->print();
    printf("Compute took %f seconds (CUDA shared)\n", ((double)(t1 - t0)) / CLOCKS_PER_SEC);

    C->zero();
    C->pad(blockSize);

    t2 = clock();
    matrixMulCUDA(A, B, C, blockSize, grid, false);
    t3 = clock();
    printf("Matrix C (CUDA without shared memory):\n");
    C->unpad();
    C->print();
    printf("Compute took %f seconds (CUDA not shared)\n", ((double)(t3 - t2)) / CLOCKS_PER_SEC);

    // free memory
    delete A;
    delete B;
    delete C;
}
