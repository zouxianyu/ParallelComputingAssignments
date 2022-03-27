#ifndef DYNAMIC_MATRIX_DYNAMIC_MATRIX_H
#define DYNAMIC_MATRIX_DYNAMIC_MATRIX_H

#include <cstddef>
#include <cassert>
#include <cstring>

template<typename T>
class DynamicMatrix {
private:
    class Proxy {
    private:
        T *mem;
        int rows;
        int cols;
        int row;
    public:
        Proxy(T *mem, int rows, int cols, int row);

        T &operator[](int column);
    };

    T *mem;
    int rows;
    int cols;
    int align;
public:
    DynamicMatrix(int rows, int cols, int align = 0);

    ~DynamicMatrix();

    DynamicMatrix(const DynamicMatrix &other);

    DynamicMatrix(DynamicMatrix &&other) noexcept;

    Proxy operator[](int row);

    int get_rows() const;

    int get_cols() const;
};

template<typename T>
T &DynamicMatrix<T>::Proxy::operator[](int col) {
    assert(col >= -1 && col < cols);
    assert(row >= -1 && row < rows);
    return *(mem + row * cols + col);
}

template<typename T>
DynamicMatrix<T>::Proxy::Proxy(T *mem, int rows, int cols, int row) : mem(mem), rows(rows), cols(cols), row(row) {
}

template<typename T>
DynamicMatrix<T>::DynamicMatrix(int rows, int cols, int align) : rows(rows), cols(cols), align(align) {
    if (align == 0) {
        mem = new T[rows * cols];
    } else {
        mem = new(std::align_val_t(align)) T[rows * cols];
    }
}

template<typename T>
DynamicMatrix<T>::~DynamicMatrix() {
    delete[] mem;
}

template<typename T>
DynamicMatrix<T>::DynamicMatrix(const DynamicMatrix &other) {
    rows = other.rows;
    cols = other.cols;
    align = other.align;
    size_t length = other.rows * other.cols;
    if (align == 0) {
        mem = new T[length];
    } else {
        mem = new(std::align_val_t(align)) T[length];
    }
    memcpy(mem, other.mem, length * sizeof(T));
}

template<typename T>
DynamicMatrix<T>::DynamicMatrix(DynamicMatrix &&other) noexcept {
    rows = other.rows;
    cols = other.cols;
    mem = other.mem;
    other.mem = nullptr;
}

template<typename T>
typename DynamicMatrix<T>::Proxy DynamicMatrix<T>::operator[](int row) {
    return DynamicMatrix::Proxy(mem, rows, cols, row);
}

template<typename T>
int DynamicMatrix<T>::get_rows() const {
    return rows;
}

template<typename T>
int DynamicMatrix<T>::get_cols() const {
    return cols;
}

#endif //DYNAMIC_MATRIX_DYNAMIC_MATRIX_H
