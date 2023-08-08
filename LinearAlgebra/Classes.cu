#include <stdio.h>
#include <random>
#include <time.h>
#include "LinearAlgebra.h"

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
    int new_width = this->width, new_height = this->height;
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

IdentityMatrix::IdentityMatrix(int size) : Matrix(size, size) {
    for (int i = 0; i < size; i++) {
        this->data[i * size + i] = 1.0f;
    }
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

void Vector::print() {
    if (this->size > 16) {
        printf("Vector too large to print\n");
        return;
    }
    printf("[");
    for (int i = 0; i < this->size; i++) {
        printf("%7.2f", this->data[i]);
        if (i < this->size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

void Vector::zero() {
    memset(this->data, 0, this->size * sizeof(float));
}