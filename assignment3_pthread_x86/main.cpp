#include <iostream>
#include <fstream>
#include <chrono>
#include <ratio>
#include <iomanip>
#include <random>
#include <thread>
#include <functional>
#include <semaphore.h>

#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2

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


void gauss_thread_naive(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1;

        int remainder = n - (k + 1);
        if (remainder == 0) {
            // no need to do anything
            break;
        }
        // if the remaining rows are less than max_threads,
        // then the number of threads is equal to the remaining rows
        int threads_count = std::min(max_threads, remainder);
        std::vector<std::thread> threads;
        int step = remainder / threads_count;

        int i_start = k + 1;
        int i_end = i_start + step;
        // using slave threads
        // the master thread will handle the remaining rows
        for (; i_end + step < n;
               i_start += step, i_end += step) {
            // &m, k, n, i_start, i_end
            threads.emplace_back([=, &m]() {
                for (int i = i_start; i < i_end; i++) {
                    for (int j = k + 1; j < n; j++) {
                        m[i][j] = m[i][j] - m[i][k] * m[k][j];
                    }
                    m[i][k] = 0;
                }
            });
        }
        // solve the end part of the matrix by master thread
        for (int i = i_start; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
        // wait for all threads to finish
        for (auto &t: threads) {
            t.join();
        }
    }
}

void gauss_thread_roll(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1;

        int remainder = n - (k + 1);
        if (remainder == 0) {
            // no need to do anything
            break;
        }
        // if the remaining rows are less than max_threads,
        // then the number of threads is equal to the remaining rows
        int threads_count = std::min(max_threads, remainder);
        std::vector<std::thread> threads;

        // assign works to each thread
        std::vector<std::vector<int>> works(threads_count);
        for (int i = k + 1; i < n; i++) {
            works[i % threads_count].push_back(i);
        }
        // create slave threads
        // the first work is assigned to master thread
        for (int t = 1; t < threads_count; t++) {
            threads.emplace_back([=, &m, &works]() {
                for (int i: works[t]) {
                    for (int j = k + 1; j < n; j++) {
                        m[i][j] = m[i][j] - m[i][k] * m[k][j];
                    }
                    m[i][k] = 0;
                }
            });
        }
        // solve the first part of the work by master thread
        for (int i: works[0]) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
        // wait for all threads to finish
        for (auto &t: threads) {
            t.join();
        }
    }
}

void gauss_thread_roll_param_reduce(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1;

        int remainder = n - (k + 1);
        if (remainder == 0) {
            // no need to do anything
            break;
        }
        // if the remaining rows are less than max_threads,
        // then the number of threads is equal to the remaining rows
        int threads_count = std::min(max_threads, remainder);
        std::vector<std::thread> threads;

        // create slave threads
        // the first work is assigned to master thread
        for (int t = 1; t < threads_count; t++) {
            threads.emplace_back([=, &m]() {
                for (int i = k + 1 + t; i < n; i += threads_count) {
                    for (int j = k + 1; j < n; j++) {
                        m[i][j] = m[i][j] - m[i][k] * m[k][j];
                    }
                    m[i][k] = 0;
                }
            });
        }
        // solve the first part of the work by master thread
        for (int i = k + 1; i < n; i += threads_count) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
        // wait for all threads to finish
        for (auto &t: threads) {
            t.join();
        }
    }
}

struct work_t {
    // volatile data
    DynamicMatrix<float> &m;
    std::vector<sem_t> &sems;
    bool &end;
    int &k;
    int &threads_count;

    // static data
    int n;
    int t;
};

void *worker_thread(void *arg) {
    work_t &work = *(work_t *) arg;
    DynamicMatrix<float> &m = work.m;
    std::vector<sem_t> &sems = work.sems;
    bool &end = work.end;
    int &k = work.k;
    int &threads_count = work.threads_count;
    int n = work.n;
    int t = work.t;
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
    return nullptr;
}

void gauss_thread_worker_roll_param_reduce_pthread(DynamicMatrix<float> &m) {
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

    std::vector<pthread_t> threads(max_threads - 1);
    for (int t = 1; t < max_threads; t++) {
        pthread_create(&threads[t - 1], nullptr, worker_thread,
                       (void *) new work_t{m, sems, end, k, threads_count, n, t});
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
        pthread_join(t, nullptr);
    }
}

void gauss_thread_worker_roll_param_reduce(DynamicMatrix<float> &m) {
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

void gauss_thread_worker_roll_param_reduce_avoid_false_sharing(DynamicMatrix<float> &m) {
    assert(m.get_rows() == m.get_cols());
    int n = m.get_rows();

    // cache line size aligned sem_t
    struct alignas(cache_line_size) aligned_sem_t {
        sem_t v;
    };
    // initialize the sempahores for master thread and slave threads
    std::vector<aligned_sem_t> sems(max_threads);
    for (auto &sem: sems) {
        sem_init(&sem.v, 0, 0);
    }

    bool end = false;
    int k;
    int threads_count;

    std::vector<std::thread> threads;
    for (int t = 1; t < max_threads; t++) {
        threads.emplace_back([&, n, t]() {
            while (true) {
                sem_wait(&sems[t].v);
                if (end) {
                    break;
                }
                for (int i = k + 1 + t; i < n; i += threads_count) {
                    for (int j = k + 1; j < n; j++) {
                        m[i][j] = m[i][j] - m[i][k] * m[k][j];
                    }
                    m[i][k] = 0;
                }
                sem_post(&sems[0].v);
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
            sem_post(&sems[t].v);
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
            sem_wait(&sems[0].v);
        }
    }
    // stop all slave threads and wait them to exit before master thread exit
    end = true;
    for (int t = 1; t < max_threads; t++) {
        sem_post(&sems[t].v);
    }
    for (auto &t: threads) {
        t.join();
    }
}

void gauss_thread_worker_roll_param_reduce_prefetch(DynamicMatrix<float> &m) {
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
                    // prefetch
                    if (i + threads_count < n) {
                        __builtin_prefetch(&m[i + threads_count][k]);
                    }
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
            // prefetch
            if (i + threads_count < n) {
                __builtin_prefetch(&m[i + threads_count][k]);
            }
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

void gauss_thread_worker_roll_param_reduce_SSE_aligned(DynamicMatrix<float> &m) {
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
                sem_post(&sems[0]);
            }
        });
    }

    for (k = 0; k < n; k++) {

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

void gauss_thread_worker_roll_param_reduce_AVX_aligned(DynamicMatrix<float> &m) {
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
                sem_post(&sems[0]);
            }
        });
    }

    for (k = 0; k < n; k++) {

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

void gauss_thread_worker_roll_param_reduce_AVX512_aligned(DynamicMatrix<float> &m) {
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
                sem_post(&sems[0]);
            }
        });
    }

    for (k = 0; k < n; k++) {

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
//    std::cout << "gauss_thread_worker_roll_param_reduce_AVX512_aligned" << std::endl;
//    gauss_thread_worker_roll_param_reduce_AVX512_aligned(m2);
//    matrix_print(m2);
//    return 0;


    std::vector<gauss_func> gauss_funcs = {
//            gauss_normal,
//            gauss_thread_naive,
//            gauss_thread_roll,
//            gauss_thread_roll_param_reduce,
            gauss_thread_worker_roll_param_reduce,
//            gauss_thread_worker_roll_param_reduce_pthread,
//            gauss_thread_worker_roll_param_reduce_prefetch,
//            gauss_thread_worker_roll_param_reduce_avoid_false_sharing,
            gauss_thread_worker_roll_param_reduce_SSE_aligned,
            gauss_thread_worker_roll_param_reduce_AVX_aligned,
            gauss_thread_worker_roll_param_reduce_AVX512_aligned,
    };

    std::vector<std::string> names = {
//            "normal",
//            "thread_naive",
//            "thread_roll",
//            "thread_roll_param_reduce",
            "thread_worker_roll_param_reduce",
//            "thread_worker_roll_param_reduce_pthread",
//            "thread_worker_roll_param_reduce_prefetch",
//            "thread_worker_roll_param_reduce_avoid_false_sharing",
            "thread_worker_roll_param_reduce_SSE_aligned",
            "gauss_thread_worker_roll_param_reduce_AVX_aligned",
            "gauss_thread_worker_roll_param_reduce_AVX512_aligned",
    };

    const int times = 10;
    const int begin = 16;
    const int end = 512;


//    normal_test_output(gauss_funcs, names, times, begin, end);
    thread_test_output(
            gauss_thread_worker_roll_param_reduce,
            {2, 4, 8, 16, 32},
            times,
            begin,
            end
    );
}
