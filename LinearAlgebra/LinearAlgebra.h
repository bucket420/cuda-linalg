
class Matrix {
public:
    float *data;
    int width;
    int height;
    int padded_width = 0;
    int padded_height = 0;

    Matrix(int width, int height);
    ~Matrix();

    void print();
    void pad(int blockSize);
    void unpad();
    void zero();
    void transpose();
};

class Vector {
public:
    float *data;
    int size;
    int padded_size = 0;

    Vector(int size);
    ~Vector();

    void print();
    void pad(int blockSize);
    void unpad();
    void zero();
};

__global__ void matrixMulSharedMemoryKernel(const float *A, const float *B, float *C, int widthA, int widthC, int heightC);
__global__ void matrixMulKernel(const float *A, const float *B, float *C, int widthA, int widthC, int heightC);
void matrixMulSequential(Matrix* A, Matrix *B, Matrix* C);
void matrixMulCUDA(Matrix* A, Matrix *B, Matrix* C, int blockSize, dim3 grid, bool useSharedMemory);