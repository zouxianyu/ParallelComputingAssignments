
#include <iostream>
#include <fstream>
#include <chrono>
#include <ratio>
#include <iomanip>
#include <random>
#include <thread>
#include <functional>
#include <semaphore.h>
#include <omp.h>
#include <arm_neon.h>
#include "dynamic_matrix.h"

#define ROUND_DOWN(a, b) ((uintptr_t)(a) & ~((uintptr_t)(b) - 1))
#define ROUND_UP(a, b) (((uintptr_t)(a) + ((uintptr_t)(b) - 1)) & ~((uintptr_t)(b) - 1))

int max_threads = 8;
const int cache_line_size = 64;

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

void gauss_omp_naive(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    int i, j, k;
#pragma omp parallel num_threads(max_threads), default(none), private(i, j, k), shared(m, n)
    for (k = 0; k < n; k++) {
#pragma omp single
        {
            for (j = k + 1; j < n; j++) {
                m[k][j] = m[k][j] / m[k][k];
            }
            m[k][k] = 1;
        }
#pragma omp for
        for (i = k + 1; i < n; i++) {
            for (j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

void gauss_omp_add_first(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    int i, j, k;
#pragma omp parallel num_threads(max_threads), default(none), private(i, j, k), shared(m, n)
    for (k = 0; k < n; k++) {
#pragma omp for
        for (j = k + 1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1;
#pragma omp for
        for (i = k + 1; i < n; i++) {
            for (j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

void gauss_omp_roll(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    int i, j, k;
#pragma omp parallel num_threads(max_threads), default(none), private(i, j, k), shared(m, n)
    for (k = 0; k < n; k++) {
#pragma omp single
        {
            for (j = k + 1; j < n; j++) {
                m[k][j] = m[k][j] / m[k][k];
            }
            m[k][k] = 1;
        }
#pragma omp for schedule(static, 1)
        for (i = k + 1; i < n; i++) {
            for (j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

void gauss_omp_column(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    int i, j, k;
#pragma omp parallel num_threads(max_threads), default(none), private(i, j, k), shared(m, n)
    for (k = 0; k < n; k++) {
#pragma omp for
        for (j = k + 1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
            for (i = k + 1; i < n; i++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
        }
#pragma omp single
        {
            m[k][k] = 1;
            for (i = k + 1; i < n; i++) {
                m[i][k] = 0;
            }
        }
    }
}

void gauss_thread_worker_block(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    // initialize the sempahores for master thread and slave threads
    std::vector<sem_t> sems(max_threads);
    for (auto &sem: sems) {
        sem_init(&sem, 0, 0);
    }

    bool end = false;
    int k;
    int step;

    std::vector<std::thread> threads;
    for (int t = 0; t < max_threads - 1; t++) {
        threads.emplace_back([&, n, t]() {
            while (true) {
                sem_wait(&sems[t]);
                if (end) {
                    break;
                }

                int i_start = k + 1 + t * step;
                int i_end = i_start + step;
                for (int i = i_start; i < i_end; i++) {
                    for (int j = k + 1; j < n; j++) {
                        m[i][j] = m[i][j] - m[i][k] * m[k][j];
                    }
                    m[i][k] = 0;
                }
                sem_post(&sems[max_threads - 1]);
            }
        });
    }

    for (k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1;

        int remainder = n - (k + 1);
        if (remainder == 0) {
            break;
        }

        // if the remaining rows are less than max_threads,
        // then the number of threads is equal to the remaining rows
        int threads_count = std::min(max_threads, remainder);
        step = remainder / threads_count;

        // resume salve threads
        // the last work is assigned to master thread
        for (int t = 0; t < threads_count - 1; t++) {
            sem_post(&sems[t]);
        }

        // solve the last part of the work by master thread
        for (int i = k + 1 + (threads_count - 1) * step; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }

        // wait for all threads to finish
        for (int t = 0; t < threads_count - 1; t++) {
            sem_wait(&sems[max_threads - 1]);
        }
    }
    // stop all slave threads and wait them to exit before master thread exit
    end = true;
    for (int t = 0; t < max_threads - 1; t++) {
        sem_post(&sems[t]);
    }
    for (auto &t: threads) {
        t.join();
    }
}

void gauss_thread_worker_roll(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    // initialize the sempahores for master thread and slave threads
    std::vector<sem_t> sems(max_threads);
    for (auto &sem: sems) {
        sem_init(&sem, 0, 0);
    }

    bool end = false;
    int k;
    int threads_count;

    std::vector<std::thread> threads;
    for (int t = 1; t < max_threads; t++) {
        threads.emplace_back([&, n, t]() {
            while (true) {
                sem_wait(&sems[t]);
                if (end) {
                    break;
                }
                for (int i = k + 1 + t; i < n; i += threads_count) {
                    for (int j = k + 1; j < n; j++) {
                        m[i][j] = m[i][j] - m[i][k] * m[k][j];
                    }
                    m[i][k] = 0;
                }
                sem_post(&sems[0]);
            }
        });
    }

    for (k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1;

        int remainder = n - (k + 1);
        if (remainder == 0) {
            break;
        }

        // if the remaining rows are less than max_threads,
        // then the number of threads is equal to the remaining rows
        threads_count = std::min(max_threads, remainder);

        // the first work is assigned to master thread
        for (int t = 1; t < threads_count; t++) {
            sem_post(&sems[t]);
        }

        // solve the first part of the work by master thread
        for (int i = k + 1; i < n; i += threads_count) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }

        // wait for all threads to finish
        for (int t = 1; t < threads_count; t++) {
            sem_wait(&sems[0]);
        }
    }
    // stop all slave threads and wait them to exit before master thread exit
    end = true;
    for (int t = 1; t < max_threads; t++) {
        sem_post(&sems[t]);
    }
    for (auto &t: threads) {
        t.join();
    }
}

void gauss_omp_NEON_man(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    int i, j, k;
#pragma omp parallel num_threads(max_threads), default(none), private(i, j, k), shared(m, n)
    for (k = 0; k < n; k++) {
#pragma omp single
        {
            // solve the beginning of unaligned columns
            for (j = k + 1; j < ROUND_UP(k + 1, 4); j++) {
                m[k][j] = m[k][j] / m[k][k];
            }

            // load main diagonal element
            float32x4_t main_reg = vmovq_n_f32(m[k][k]);

            // using aligned load and store
            for (; j < ROUND_DOWN(n, 4); j += 4) {
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
        }

#pragma omp for
        for (i = k + 1; i < n; i++) {

            // solve the beginning of unaligned columns
            for (j = k + 1; j < ROUND_UP(k + 1, 4); j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }

            // load main diagonal element
            float32x4_t first_reg = vmovq_n_f32(m[i][k]);

            // using aligned load and store
            for (; j < ROUND_DOWN(n, 4); j += 4) {
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

void gauss_omp_NEON_auto(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    int i, j, k;
#pragma omp parallel num_threads(max_threads), default(none), private(i, j, k), shared(m, n)
    for (k = 0; k < n; k++) {
#pragma omp single
        {
#pragma omp simd
            for (j = k + 1; j < n; j++) {
                m[k][j] = m[k][j] / m[k][k];
            }
            m[k][k] = 1;
        }
#pragma omp for
        for (i = k + 1; i < n; i++) {
#pragma omp simd
            for (j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

using gauss_func = std::function<void(DynamicMatrix<float> &)>;

double test(int n, const gauss_func &gauss, int times) {
    DynamicMatrix<float> m(n, n, 64);

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

void
normal_test_output(const std::vector<gauss_func> &gauss_funcs,
                   const std::vector<std::string> &names,
                   int times,
                   int begin,
                   int end) {

    std::ofstream outfile("gauss_time.csv");

    //generate the table bar
    outfile << "问题规模,";
    for (auto &name: names) {
        outfile << name << ",";
    }
    outfile << std::endl;

    //generate the table content
    for (int n = begin; n <= end; n *= 2) {
        outfile << n << ",";
        for (auto &func: gauss_funcs) {
            outfile << test(n, func, times) << ",";
        }
        outfile << std::endl;
        std::cout << n << "\tfinished" << std::endl;
    }

    outfile.close();
}

void
thread_test_output(const gauss_func &gauss,
                   const std::vector<int> &threads,
                   int times,
                   int begin,
                   int end) {

    std::ofstream outfile("gauss_time.csv");

    //generate the table bar
    outfile << "问题规模,";
    for (int n = begin; n <= end; n *= 2) {
        outfile << n << ",";
    }
    outfile << std::endl;

    //generate the table content
    for (int t: threads) {
        outfile << t << ",";
        ::max_threads = t;
        for (int n = begin; n <= end; n *= 2) {
            outfile << test(n, gauss, times) << ",";
        }
        outfile << std::endl;
        std::cout << "threads_count = " << t << " finished" << std::endl;
    }

    outfile.close();
}

int main() {
//    const int n = 1024;
//    DynamicMatrix<float> m(n, n, 64);
//    matrix_init(m);
//    std::cout << "gauss start" << std::endl;
//    auto start = std::chrono::high_resolution_clock::now();
//    gauss_thread_worker_roll_param_reduce_AVX_aligned(m);
//    auto end = std::chrono::high_resolution_clock::now();
//    std::cout << "elapsed time: " << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;
//    return 0;

//    const int n = 16;
//    DynamicMatrix<float> m1(n, n, 64);
//    matrix_init(m1);
//    DynamicMatrix<float> m2(m1);
//
//    gauss_normal(m1);
//    matrix_print(m1);
//    std::cout << "gauss_omp_NEON" << std::endl;
//    gauss_omp_NEON(m2);
//    matrix_print(m2);
//    return 0;


    std::vector<gauss_func> gauss_funcs = {
            gauss_normal,
            gauss_omp_naive,
            gauss_omp_add_first,
            gauss_omp_roll,
            gauss_omp_column,
            gauss_thread_worker_block,
            gauss_thread_worker_roll,
            gauss_omp_NEON_man,
            gauss_omp_NEON_auto,
    };

    std::vector<std::string> names = {
            "normal",
            "omp_naive",
            "omp_add_first",
            "omp_roll",
            "omp_column",
            "thread_worker_block",
            "thread_worker_roll",
            "omp_NEON_man",
            "omp_NEON_auto",
    };

    const int times = 10;
    const int begin = 16;
    const int end = 1024;


    normal_test_output(gauss_funcs, names, times, begin, end);
//    thread_test_output(
//            gauss_thread_worker_roll_param_reduce,
//            {2, 4, 8, 16, 32},
//            times,
//            begin,
//            end
//    );
}
