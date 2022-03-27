#include <iostream>
#include <chrono>
#include <ratio>
#include <iomanip>
#include <random>

#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
#include <fstream>

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

void gauss_SSE_aligned(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    for (int k = 0; k < n; k++) {

        // solve the beginning of unaligned columns
        for (int j = k + 1; j < ROUND_UP(k + 1, 4); j++) {
            m[k][j] = m[k][j] / m[k][k];
        }

        // load main diagonal element
        __m128 main_reg = _mm_set1_ps(m[k][k]);

        // solve the 16 byte aligned columns
        for (int j = ROUND_UP(k + 1, 4); j < ROUND_DOWN(n, 4); j += 4) {
            __m128 dividend_reg = _mm_load_ps(&m[k][j]);
            __m128 result_reg = _mm_div_ps(dividend_reg, main_reg);
            _mm_store_ps(&m[k][j], result_reg);
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
            __m128 first_reg = _mm_set1_ps(m[i][k]);

            // solve the aligned columns
            for (int j = ROUND_UP(k + 1, 4); j < ROUND_DOWN(n, 4); j += 4) {
                __m128 t1_reg = _mm_load_ps(&m[k][j]);
                t1_reg = _mm_mul_ps(first_reg, t1_reg);
                __m128 t2_reg = _mm_load_ps(&m[i][j]);
                t2_reg = _mm_sub_ps(t2_reg, t1_reg);
                _mm_store_ps(&m[i][j], t2_reg);
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

void gauss_SSE_unaligned(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    for (int k = 0; k < n; k++) {

        // load main diagonal element
        __m128 main_reg = _mm_set1_ps(m[k][k]);

        int j;
        // using unaligned load and store
        for (j = k + 1; j + 4 <= n; j += 4) {
            __m128 dividend_reg = _mm_loadu_ps(&m[k][j]);
            __m128 result_reg = _mm_div_ps(dividend_reg, main_reg);
            _mm_storeu_ps(&m[k][j], result_reg);
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
            __m128 first_reg = _mm_set1_ps(m[i][k]);

            // using unaligned load and store
            for (j = k + 1; j + 4 <= n; j += 4) {
                __m128 t1_reg = _mm_loadu_ps(&m[k][j]);
                t1_reg = _mm_mul_ps(first_reg, t1_reg);
                __m128 t2_reg = _mm_loadu_ps(&m[i][j]);
                t2_reg = _mm_sub_ps(t2_reg, t1_reg);
                _mm_storeu_ps(&m[i][j], t2_reg);
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

void gauss_AVX_aligned(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    for (int k = 0; k < n; k++) {

        // solve the beginning of unaligned columns
        for (int j = k + 1; j < ROUND_UP(k + 1, 8); j++) {
            m[k][j] = m[k][j] / m[k][k];
        }

        // load main diagonal element
        __m256 main_reg = _mm256_set1_ps(m[k][k]);

        // solve the aligned columns
        for (int j = ROUND_UP(k + 1, 8); j < ROUND_DOWN(n, 8); j += 8) {
            __m256 dividend_reg = _mm256_load_ps(&m[k][j]);
            __m256 result_reg = _mm256_div_ps(dividend_reg, main_reg);
            _mm256_store_ps(&m[k][j], result_reg);
        }

        // solve the end of unaligned columns
        for (int j = ROUND_DOWN(n, 8); j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }

        // set the main diagonal of current row to 1
        m[k][k] = 1;

        // solve the rest rows
        for (int i = k + 1; i < n; i++) {

            // solve the beginning of unaligned columns
            for (int j = k + 1; j < ROUND_UP(k + 1, 8); j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }

            // load main diagonal element
            __m256 first_reg = _mm256_set1_ps(m[i][k]);

            // solve the aligned columns
            for (int j = ROUND_UP(k + 1, 8); j < ROUND_DOWN(n, 8); j += 8) {
                __m256 t1_reg = _mm256_load_ps(&m[k][j]);
                t1_reg = _mm256_mul_ps(first_reg, t1_reg);
                __m256 t2_reg = _mm256_load_ps(&m[i][j]);
                t2_reg = _mm256_sub_ps(t2_reg, t1_reg);
                _mm256_store_ps(&m[i][j], t2_reg);
            }

            // solve the end of unaligned columns
            for (int j = ROUND_DOWN(n, 8); j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }

            // set the fisrt element of current row to 0
            m[i][k] = 0;
        }
    }
}

void gauss_AVX_unaligned(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    for (int k = 0; k < n; k++) {

        // load main diagonal element
        __m256 main_reg = _mm256_set1_ps(m[k][k]);

        int j;
        // using unaligned load and store
        for (j = k + 1; j + 8 <= n; j += 8) {
            __m256 dividend_reg = _mm256_loadu_ps(&m[k][j]);
            __m256 result_reg = _mm256_div_ps(dividend_reg, main_reg);
            _mm256_storeu_ps(&m[k][j], result_reg);
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
            __m256 first_reg = _mm256_set1_ps(m[i][k]);

            // using unaligned load and store
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 t1_reg = _mm256_loadu_ps(&m[k][j]);
                t1_reg = _mm256_mul_ps(first_reg, t1_reg);
                __m256 t2_reg = _mm256_loadu_ps(&m[i][j]);
                t2_reg = _mm256_sub_ps(t2_reg, t1_reg);
                _mm256_storeu_ps(&m[i][j], t2_reg);
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

void gauss_AVX512_aligned(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    for (int k = 0; k < n; k++) {

        // solve the beginning of unaligned columns
        for (int j = k + 1; j < ROUND_UP(k + 1, 16); j++) {
            m[k][j] = m[k][j] / m[k][k];
        }

        // load main diagonal element
        __m512 main_reg = _mm512_set1_ps(m[k][k]);

        // solve the aligned columns
        for (int j = ROUND_UP(k + 1, 16); j < ROUND_DOWN(n, 16); j += 16) {
            __m512 dividend_reg = _mm512_load_ps(&m[k][j]);
            __m512 result_reg = _mm512_div_ps(dividend_reg, main_reg);
            _mm512_store_ps(&m[k][j], result_reg);
        }

        // solve the end of unaligned columns
        for (int j = ROUND_DOWN(n, 16); j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }

        // set the main diagonal of current row to 1
        m[k][k] = 1;

        // solve the rest rows
        for (int i = k + 1; i < n; i++) {

            // solve the beginning of unaligned columns
            for (int j = k + 1; j < ROUND_UP(k + 1, 16); j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }

            // load main diagonal element
            __m512 first_reg = _mm512_set1_ps(m[i][k]);

            // solve the aligned columns
            for (int j = ROUND_UP(k + 1, 16); j < ROUND_DOWN(n, 16); j += 16) {
                __m512 t1_reg = _mm512_load_ps(&m[k][j]);
                t1_reg = _mm512_mul_ps(first_reg, t1_reg);
                __m512 t2_reg = _mm512_load_ps(&m[i][j]);
                t2_reg = _mm512_sub_ps(t2_reg, t1_reg);
                _mm512_store_ps(&m[i][j], t2_reg);
            }

            // solve the end of unaligned columns
            for (int j = ROUND_DOWN(n, 16); j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }

            // set the fisrt element of current row to 0
            m[i][k] = 0;
        }
    }
}

void gauss_AVX512_unaligned(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    for (int k = 0; k < n; k++) {

        // load main diagonal element
        __m512 main_reg = _mm512_set1_ps(m[k][k]);

        int j;
        // using unaligned load and store
        for (j = k + 1; j + 16 <= n; j += 16) {
            __m512 dividend_reg = _mm512_loadu_ps(&m[k][j]);
            __m512 result_reg = _mm512_div_ps(dividend_reg, main_reg);
            _mm512_storeu_ps(&m[k][j], result_reg);
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
            __m512 first_reg = _mm512_set1_ps(m[i][k]);

            // using unaligned load and store
            for (j = k + 1; j + 16 <= n; j += 16) {
                __m512 t1_reg = _mm512_loadu_ps(&m[k][j]);
                t1_reg = _mm512_mul_ps(first_reg, t1_reg);
                __m512 t2_reg = _mm512_loadu_ps(&m[i][j]);
                t2_reg = _mm512_sub_ps(t2_reg, t1_reg);
                _mm512_storeu_ps(&m[i][j], t2_reg);
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
            gauss_SSE_aligned, gauss_SSE_unaligned,
            gauss_AVX_aligned, gauss_AVX_unaligned,
            gauss_AVX512_aligned, gauss_AVX512_unaligned
    };

    std::string names[] = {
            "normal",
            "SSE_aligned", "SSE_unaligned",
            "AVX_aligned", "AVX_unaligned",
            "AVX512_aligned", "AVX512_unaligned"
    };

    const int times = 10;
    const int begin = 16;
    const int end = 512;


    std::ofstream outfile("gauss_time.csv");

    //generate the table bar
    outfile << "问题规模,";
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
