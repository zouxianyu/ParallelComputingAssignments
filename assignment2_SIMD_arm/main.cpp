#include <iostream>
#include <chrono>
#include <ratio>
#include <iomanip>
#include <random>
#include <fstream>

#include <arm_neon.h>

#include "dynamic_matrix.h"

#define ROUND_DOWN(a, b) ((uintptr_t)(a) & ~((uintptr_t)(b) - 1))
#define ROUND_UP(a, b) (((uintptr_t)(a) + ((uintptr_t)(b) - 1)) & ~((uintptr_t)(b) - 1))

void matrix_print(DynamicMatrix<float> &m) {
    for (int i = 0; i < m.get_rows(); i++) {
        for (int j = 0; j < m.get_cols(); j++) {
            std::cout << std::setw(16) << m[i][j];
        }
        std::cout << std::endl;
    }
}

void matrix_init(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());

    static std::default_random_engine generator(1337);
    static std::uniform_real_distribution<float> distribution(-1.0, 1.0);

    int n = m.get_rows();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++)
            m[i][j] = 0;
        m[i][i] = 1.0;
        for (int j = i + 1; j < n; j++)
            m[i][j] = distribution(generator);
    }
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i][j] += m[k][j];
            }
        }
    }
}

void gauss_normal(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1;
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

void gauss_NEON_aligned(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    for (int k = 0; k < n; k++) {

        // solve the beginning of unaligned columns
        for (int j = k + 1; j < ROUND_UP(k + 1, 4); j++) {
            m[k][j] = m[k][j] / m[k][k];
        }

        // load main diagonal element
        float32x4_t main_reg = vmovq_n_f32(m[k][k]);

        // solve the 16 byte aligned columns
        for (int j = ROUND_UP(k + 1, 4); j < ROUND_DOWN(n, 4); j += 4) {
            float* ptr = &m[k][j];
            __builtin_assume_aligned(ptr, 16);
            float32x4_t dividend_reg = vld1q_f32(ptr);
            float32x4_t result_reg = vaddq_f32(dividend_reg, main_reg);
            vst1q_f32(ptr, result_reg);
        }

        // solve the end of unaligned columns
        for (int j = ROUND_DOWN(n, 4); j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }

        // set the main diagonal of current row to 1
        m[k][k] = 1;

        // solve the rest rows
        for (int i = k + 1; i < n; i++) {

            // solve the beginning of unaligned columns
            for (int j = k + 1; j < ROUND_UP(k + 1, 4); j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }

            // load main diagonal element
            float32x4_t first_reg = vmovq_n_f32(m[i][k]);

            // solve the aligned columns
            for (int j = ROUND_UP(k + 1, 4); j < ROUND_DOWN(n, 4); j += 4) {
                float* t1_ptr = &m[k][j];
                float* t2_ptr = &m[i][j];
                __builtin_assume_aligned(t1_ptr, 16);
                __builtin_assume_aligned(t2_ptr, 16);
                float32x4_t t1_reg = vld1q_f32(t1_ptr);
                t1_reg = vmulq_f32(first_reg, t1_reg);
                float32x4_t t2_reg = vld1q_f32(t2_ptr);
                t2_reg = vsubq_f32(t2_reg, t1_reg);
                vst1q_f32(t2_ptr, t2_reg);
            }

            // solve the end of unaligned columns
            for (int j = ROUND_DOWN(n, 4); j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }

            // set the fisrt element of current row to 0
            m[i][k] = 0;
        }
    }
}

void gauss_NEON_unaligned(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    for (int k = 0; k < n; k++) {

        // load main diagonal element
        float32x4_t main_reg = vmovq_n_f32(m[k][k]);

        int j;
        // using unaligned load and store
        for (j = k + 1; j + 4 <= n; j += 4) {
            float32x4_t dividend_reg = vld1q_f32(&m[k][j]);
            float32x4_t result_reg = vdivq_f32(dividend_reg, main_reg);
            vst1q_f32(&m[k][j], result_reg);
        }

        // solve the end of unaligned columns
        for (; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }

        // set the main diagonal of current row to 1
        m[k][k] = 1;

        // solve the rest rows
        for (int i = k + 1; i < n; i++) {

            // load main diagonal element
            float32x4_t first_reg = vmovq_n_f32(m[i][k]);

            // using unaligned load and store
            for (j = k + 1; j + 4 <= n; j += 4) {
                float32x4_t t1_reg = vld1q_f32(&m[k][j]);
                t1_reg = vmulq_f32(first_reg, t1_reg);
                float32x4_t t2_reg = vld1q_f32(&m[i][j]);
                t2_reg = vsubq_f32(t2_reg, t1_reg);
                vst1q_f32(&m[i][j], t2_reg);
            }

            // solve the end of unaligned columns
            for (; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }

            // set the fisrt element of current row to 0
            m[i][k] = 0;
        }
    }
}

typedef void (*gauss_func)(DynamicMatrix<float> &);

double test(int n, gauss_func gauss, int times) {
    DynamicMatrix<float> m(n, n,64);

    std::chrono::duration<double, std::milli> elapsed{};
    for (int i = 0; i < times; i++) {
        matrix_init(m);
        auto start = std::chrono::high_resolution_clock::now();
        gauss(m);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed += end - start;
    }
    return elapsed.count() / times;
}

int main() {
    gauss_func gauss_funcs[] = {
            gauss_normal,
            gauss_NEON_aligned, gauss_NEON_unaligned,
    };

    std::string names[] = {
            "normal",
            "NEON_aligned", "NEON_unaligned",
    };

    const int times = 100;
    const int begin = 16;
    const int end = 1024;


    std::ofstream outfile("gauss_time.csv");

    //generate the table bar
    outfile << ",";
    for (auto & name : names) {
        outfile << name << ",";
    }
    outfile << std::endl;

    //generate the table content
    for (int n = begin; n <= end; n *= 2) {
        outfile << n << ",";
        for (auto & func : gauss_funcs) {
            outfile << test(n, func, times) << ",";
        }
        outfile << std::endl;
        std::cout<< n << "\tfinished" << std::endl;
    }

    outfile.close();
}
