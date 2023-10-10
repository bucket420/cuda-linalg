#include <stdio.h>
#include <random>
#include <time.h>
#include "LinearAlgebra.h"


// const int DSIZE = 8;
// CUDA maximum is 1024 *total* threads in block
// const float A_val = 1.0f;
// const float B_val = 2.0f;

bool floatCompare(float nx1, float mx1, float epsilon = 0.02f) {
   return fabs(nx1 - mx1) < epsilon;
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
                printf("Expected: %f, Actual: %f\n", expected->data[i * expected->width + j], result->data[i * result->width + j]);
                return;
            }
        }
    }
    printf("SUCCESS: result matches expected\n");
}

void verify(Vector* result, Vector* expected) {
    if (result->size != expected->size) {
        printf("ERROR: result has incorrect dimensions\n");
        return;
    }
    for (int i = 0; i < result->size; i++) {
        if (!floatCompare(result->data[i], expected->data[i], 1.0f)) {
            printf("ERROR: result has incorrect value at (%d)\n", i);
            printf("Expected: %f, Actual: %f\n", expected->data[i], result->data[i]);
            return;
        }
    }
    printf("SUCCESS: result matches expected\n");
}


void testMatrixMul()
{
    const int blockSize = 16;
    dim3 grid(512, 512);
    int m = 1000;
    int n = 1000;
    int p = 1000;
    
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 10.0);

    // these are just for timing
    clock_t start, end;

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
    start = clock();
    matrixMul(mxn, nxp, mxpSequential);
    end = clock();
    printf("Matrix mxp (sequential):\n");
    mxpSequential->print();
    printf("Compute took %f seconds (sequential)\n", ((double)(end - start)) / CLOCKS_PER_SEC);


    // mxp->pad(blockSize);
    start = clock();
    matrixMulCUDA(mxn, nxp, mxp, blockSize, grid, 0);
    end = clock();
    end = clock();
    printf("Matrix mxp (CUDA without shared memory):\n");
    mxp->print();
    printf("Compute took %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    verify(mxp, mxpSequential);
    mxp->zero();

    start = clock();
    matrixMulCUDA(mxn, nxp, mxp, blockSize, grid, 1);
    end = clock();
    printf("Matrix mxp (CUDA with shared memory, no padding):\n");
    mxp->print();
    printf("Compute took %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    verify(mxp, mxpSequential);
    mxp->zero();

    start = clock();
    matrixMulCUDA(mxn, nxp, mxp, blockSize, grid, 2);
    end = clock();
    printf("Matrix mxp (CUDA with shared memory, with padding):\n");
    mxp->unpad();
    mxp->print();
    printf("Compute took %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    verify(mxp, mxpSequential);

    // free memory
    delete mxn;
    delete nxp;
    delete mxp;
    delete mxpSequential;
}

void timeMatrixMulCUDA()
{
    const int blockSize = 16;
    dim3 grid(512, 512);
    int m = 6969;
    int n = 4242;
    int p = 9696;
    
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 10.0);

    // these are just for timing
    clock_t start, end;

    // create matrices
    Matrix *mxn = new Matrix(n, m);
    Matrix *nxp = new Matrix(p, n);
    Matrix *mxp = new Matrix(p, m);

    // initialize matrices
    for (int i = 0; i < mxn->height; i++) {
        for (int j = 0; j < mxn->width; j++) {
            // mxn->data[i * mxn->width + j] = dis(gen);
            mxn->data[i * mxn->width + j] = i / (i + 100.0f);
        }
    }

    for (int i = 0; i < nxp->height; i++) {
        for (int j = 0; j < nxp->width; j++) {
            // nxp->data[i * nxp->width + j] = dis(gen);
            nxp->data[i * nxp->width + j] = (i + 2.0f) / (i + 1.0f);
        }
    }

    // mxp->pad(blockSize);
    start = clock();
    matrixMulCUDA(mxn, nxp, mxp, blockSize, grid, 0);
    end = clock();
    printf("Matrix mxp (CUDA without shared memory):\n");
    printf("Compute took %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    mxp->zero();

    start = clock();
    matrixMulCUDA(mxn, nxp, mxp, blockSize, grid, 1);
    end = clock();
    printf("Matrix mxp (CUDA with shared memory, no padding):\n");
    printf("Compute took %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    mxp->zero();

    start = clock();
    matrixMulCUDA(mxn, nxp, mxp, blockSize, grid, 2);
    end = clock();
    printf("Matrix mxp (CUDA with shared memory, with padding):\n");
    printf("Compute took %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    // free memory
    delete mxn;
    delete nxp;
    delete mxp;
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

void testDot() {
    const int blockSize = 64;
    int numBlocks = 512;
    int size = 100000000;

    clock_t start, end;

    Vector *A = new Vector(size);
    Vector *B = new Vector(size);

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 10.0);

    for (int i = 0; i < A->size; i++) {
        // A->data[i] = dis(gen);
        A->data[i] = i / (i + 100.0f); 
    }

    for (int i = 0; i < B->size; i++) {
        // B->data[i] = dis(gen);
        B->data[i] = (i + 2.0f) / (i + 1.0f);
    }

    printf("Vector A:\n");
    A->print();
    printf("Vector B:\n");
    B->print();

    start = clock();
    float sequential = dot(A, B);
    end = clock();
    printf("Dot product (sequential): %f\n", sequential);
    printf("Compute took %f seconds (sequential)\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    start = clock();
    float cuda = dotCUDA(A, B, blockSize, numBlocks);
    end = clock();
    printf("Dot product (CUDA): %f\n", cuda);
    printf("Compute took %f seconds (CUDA)\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    if (floatCompare(sequential, cuda, 1.0f)) {
        printf("SUCCESS: result matches expected\n");
    } else {
        printf("ERROR: result does not match expected\n");
    }

    delete A;
    delete B;
}

void testMatrixVectorMul() {
    const int blockSize = 128;
    int numBlocks = 512;
    int m = 6969;
    int n = 4242;
    int sharedMemorySize = 512;
    
    clock_t start, end;

    Matrix *mxn = new Matrix(n, m);
    Vector *nx1 = new Vector(n);
    Vector *mx1 = new Vector(m);
    Vector *mx1Sequential = new Vector(m);

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 10.0);

    // initialize matrices
    for (int i = 0; i < mxn->height; i++) {
        for (int j = 0; j < mxn->width; j++) {
            // mxn->data[i * mxn->width + j] = dis(gen);
            mxn->data[i * mxn->width + j] = i / (i + 100.0f);
        }
    }

    for (int i = 0; i < nx1->size; i++) {
        // nx1->data[i] = dis(gen);
        nx1->data[i] = (i + 2.0f) / (i + 1.0f);
    }

    start = clock();
    matrixVectorMul(mxn, nx1, mx1Sequential);
    end = clock();
    printf("Matrix mxn:\n");
    mxn->print();
    printf("Vector nx1:\n");
    nx1->print();
    printf("Vector mx1 (sequential):\n");
    mx1Sequential->print();
    printf("Compute took %f seconds (sequential)\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    start = clock();
    matrixVectorMulCUDA(mxn, nx1, mx1, blockSize, numBlocks, 0);
    end = clock();
    printf("Vector mx1 (CUDA without shared memory):\n");
    mx1->print();
    printf("Compute took %f seconds (CUDA without shared memory)\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    verify(mx1, mx1Sequential);
    mx1->zero();

    start = clock();
    matrixVectorMulCUDA(mxn, nx1, mx1, blockSize, numBlocks, 1, sharedMemorySize);
    end = clock();
    printf("Vector mx1 (CUDA with shared memory):\n");
    mx1->print();
    printf("Compute took %f seconds (CUDA with shared memory)\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    verify(mx1, mx1Sequential);



    delete mxn;
    delete nx1;
    delete mx1;
    delete mx1Sequential;
}

int main(int argc, char **argv)
{
    // testMatrixMul();
    timeMatrixMulCUDA();
    // testDot();
    // testTranspose();
    testMatrixVectorMul();
    return 0;
}
