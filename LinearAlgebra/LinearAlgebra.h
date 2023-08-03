
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

class IdentityMatrix : public Matrix {
public:
    IdentityMatrix(int size);
};

void matrixMul(Matrix* mxn, Matrix *nxp, Matrix* mxp);
void matrixMulCUDA(Matrix* mxn, Matrix *nxp, Matrix* mxp, int blockSize, dim3 grid, bool useSharedMemory);

float dot(Vector* a, Vector* b);
float dotCUDA(Vector* a, Vector* b, int blockSize, int numBlocks, bool useSharedMemory=false);

