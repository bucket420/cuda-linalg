#include <stdio.h>
#include <random>
#include <time.h>
#include "LinearAlgebra.h"


// const int DSIZE = 8;
// CUDA maximum is 1024 *total* threads in block
// const float A_val = 1.0f;
// const float B_val = 2.0f;

bool floatCompare(float x, float y, float epsilon = 0.01f){
   if(fabs(x - y) < epsilon)
      return true;
      return false;
}

void verify(Matrix* result, Matrix* expected) {
    if (result->height != expected->height || result->width != expected->width) {
        printf("ERROR: result has incorrect dimensions\n");
        return;
    }
    for (int i = 0; i < result->height; i++) {
        for (int j = 0; j < result->width; j++) {
            if (!floatCompare(result->data[i * result->width + j], expected->data[i * expected->width + j])) {
                printf("ERROR: result has incorrect value at (%d, %d)\n", i, j);
                return;
            }
        }
    }
    printf("SUCCESS: result matches expected\n");
}


void testMatrixMul()
{
    const int blockSize = 16;
    dim3 grid(128, 128);
    int m = 690;
    int n = 960;
    int p = 420;

    // const int blockSize = 2;
    // dim3 grid(2, 2);
    // int m = 13;
    // int n = 15;
    // int p = 11;
    

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 10.0);

    // these are just for timing
    clock_t t0, t1, t2, t3, t4, t5;

    // create matrices
    Matrix *mxn = new Matrix(n, m);
    Matrix *nxp = new Matrix(p, n);
    Matrix *mxp = new Matrix(p, m);
    Matrix *mxpSequential = new Matrix(p, m);

    // initialize matrices
    for (int i = 0; i < mxn->height; i++) {
        for (int j = 0; j < mxn->width; j++) {
            mxn->data[i * mxn->width + j] = dis(gen);
        }
    }

    for (int i = 0; i < nxp->height; i++) {
        for (int j = 0; j < nxp->width; j++) {
            nxp->data[i * nxp->width + j] = dis(gen);
        }
    }

    // print input
    printf("Matrix mxn:\n");
    mxn->print();
    printf("Matrix nxp:\n");
    nxp->print();

    // multiply
    t4 = clock();
    matrixMul(mxn, nxp, mxpSequential);
    t5 = clock();
    printf("Matrix mxp (sequential):\n");
    mxpSequential->print();
    printf("Compute took %f seconds (sequential)\n", ((double)(t5 - t4)) / CLOCKS_PER_SEC);

    t0 = clock();
    matrixMulCUDA(mxn, nxp, mxp, blockSize, grid, true);
    t1 = clock();
    printf("Matrix mxp (CUDA with shared memory):\n");
    mxp->print();
    printf("Compute took %f seconds (CUDA shared)\n", ((double)(t1 - t0)) / CLOCKS_PER_SEC);
    verify(mxp, mxpSequential);

    mxp->zero();
    // mxp->pad(blockSize);

    t2 = clock();
    matrixMulCUDA(mxn, nxp, mxp, blockSize, grid, false);
    t3 = clock();
    printf("Matrix mxp (CUDA without shared memory):\n");
    mxp->print();
    printf("Compute took %f seconds (CUDA not shared)\n", ((double)(t3 - t2)) / CLOCKS_PER_SEC);
    verify(mxp, mxpSequential);

    // free memory
    delete mxn;
    delete nxp;
    delete mxp;
    delete mxpSequential;
}

void testTranspose() {
    Matrix *A = new Matrix(4, 4);
    Matrix *B = new Matrix(3, 4);
    Matrix *C = new Matrix(4, 3);

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 10.0);

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

    for (int i = 0; i < C->height; i++) {
        for (int j = 0; j < C->width; j++) {
            C->data[i * C->width + j] = dis(gen);
        }
    }

    printf("Matrix A:\n");
    A->print();
    A->transpose();
    printf("Matrix A (transposed):\n");
    A->print();

    printf("Matrix B:\n");
    B->print();
    B->transpose();
    printf("Matrix B (transposed):\n");
    B->print();

    printf("Matrix C:\n");
    C->print();
    C->transpose();
    printf("Matrix C (transposed):\n");
    C->print();

    delete A;
    delete B;
    delete C;
}

int main(int argc, char **argv)
{
    testMatrixMul();
    testTranspose();
    return 0;
}
