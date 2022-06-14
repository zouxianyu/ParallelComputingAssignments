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
#include <mpi.h>

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

int max_threads = 2;
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

void gauss_normal(DynamicMatrix<float> &m, int id, int procs) {
    if (id == 0) {
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
    MPI_Barrier(MPI_COMM_WORLD);
}

std::pair<int, int> get_work_range_unbalance(int n, int k, int id, int procs) {
    int global_work_size, local_work_size, start, end;
    global_work_size = n - (k + 1);
    local_work_size = global_work_size / procs;
    if (id == 0) {
        local_work_size += global_work_size % procs;
        start = 0;
        end = start + local_work_size;
    } else {
        start = id * local_work_size + global_work_size % procs;
        end = start + local_work_size;
    }
    return std::make_pair(k + 1 + start, k + 1 + end);
}

std::pair<int, int> get_work_range(int n, int k, int id, int procs) {
    int global_work_size, local_work_size, start, end;
    global_work_size = n - (k + 1);
    local_work_size = global_work_size / procs;
    if (id < global_work_size % procs) {
        local_work_size++;
        start = id * local_work_size;
        end = start + local_work_size;
    } else {
        start = id * local_work_size + global_work_size % procs;
        end = start + local_work_size;
    }
    return std::make_pair(k + 1 + start, k + 1 + end);
}

void gauss_mpi_unbalance(DynamicMatrix<float> &m, int id, int procs) {
    int n = m.get_rows();

    // only for root process
    std::vector <std::pair<int, int>> work_ranges(procs);

    for (int k = 0; k < n; k++) {
        // only the root process do the first step
        if (id == 0) {
            for (int j = k + 1; j < n; j++) {
                m[k][j] = m[k][j] / m[k][k];
            }
            m[k][k] = 1;

        }

        // broadcast the k-th row
        MPI_Bcast(&m[k][0], n, MPI_FLOAT, 0, MPI_COMM_WORLD);

        std::pair<int, int> work_range = get_work_range_unbalance(n, k, id, procs);
        if (work_range.first == work_range.second) {
            continue;
        }

        // scatter
        if (id == 0) {
            // assign works
            if (n - (k + 1) >= procs) {
                for (int i = 1; i < procs; i++) {
                    work_ranges[i] = get_work_range_unbalance(n, k, i, procs);
                    MPI_Send(&m[work_ranges[i].first][0], (work_ranges[i].second - work_ranges[i].first) * n, MPI_FLOAT,
                             i,
                             0, MPI_COMM_WORLD);
                }
            }
        } else {
            // receive works
            MPI_Recv(&m[work_range.first][0], (work_range.second - work_range.first) * n, MPI_FLOAT, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // do works
        for (int i = work_range.first; i < work_range.second; i++) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }

        // gather
        if (id == 0) {
            if (n - (k + 1) >= procs) {
                for (int i = 1; i < procs; i++) {
                    MPI_Recv(&m[work_ranges[i].first][0], (work_ranges[i].second - work_ranges[i].first) * n, MPI_FLOAT,
                             i,
                             0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        } else {
            MPI_Send(&m[work_range.first][0], (work_range.second - work_range.first) * n, MPI_FLOAT, 0, 0,
                     MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}


void gauss_mpi_normal(DynamicMatrix<float> &m, int id, int procs) {
    int n = m.get_rows();

    // only for root process
    std::vector <std::pair<int, int>> work_ranges(procs);

    for (int k = 0; k < n; k++) {
        // only the root process do the first step
        if (id == 0) {
            for (int j = k + 1; j < n; j++) {
                m[k][j] = m[k][j] / m[k][k];
            }
            m[k][k] = 1;

        }

        // broadcast the k-th row
        MPI_Bcast(&m[k][0], n, MPI_FLOAT, 0, MPI_COMM_WORLD);

        std::pair<int, int> work_range = get_work_range(n, k, id, procs);
        if (work_range.first == work_range.second) {
            continue;
        }

        // scatter
        if (id == 0) {
            // assign works
            for (int i = 1; i < std::min(procs, n - (k + 1)); i++) {
                work_ranges[i] = get_work_range(n, k, i, procs);
                MPI_Send(&m[work_ranges[i].first][0], (work_ranges[i].second - work_ranges[i].first) * n, MPI_FLOAT, i,
                         0, MPI_COMM_WORLD);
            }
        } else {
            // receive works
            MPI_Recv(&m[work_range.first][0], (work_range.second - work_range.first) * n, MPI_FLOAT, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // do works
        for (int i = work_range.first; i < work_range.second; i++) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }

        // gather
        if (id == 0) {
            for (int i = 1; i < std::min(procs, n - (k + 1)); i++) {
                MPI_Recv(&m[work_ranges[i].first][0], (work_ranges[i].second - work_ranges[i].first) * n, MPI_FLOAT, i,
                         0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            MPI_Send(&m[work_range.first][0], (work_range.second - work_range.first) * n, MPI_FLOAT, 0, 0,
                     MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void gauss_mpi_comm_reduce(DynamicMatrix<float> &m, int id, int procs) {
    int n = m.get_rows();

    // only for root process
    std::vector <std::pair<int, int>> work_ranges(procs);

    for (int k = 0; k < n; k++) {
        // only the root process do the first step
        if (id == 0) {
            for (int j = k + 1; j < n; j++) {
                m[k][j] = m[k][j] / m[k][k];
            }
            m[k][k] = 1;

        }

        // broadcast the k-th row
        MPI_Bcast(&m[k][k], n - k, MPI_FLOAT, 0, MPI_COMM_WORLD);


        std::pair<int, int> work_range = get_work_range(n, k, id, procs);
        if (work_range.first == work_range.second) {
            continue;
        }

        // scatter
        if (id == 0) {
            // assign works
            for (int i = 1; i < std::min(procs, n - (k + 1)); i++) {
                work_ranges[i] = get_work_range(n, k, i, procs);
                int starts[] = {work_ranges[i].first, k};
                int subsizes[] = {work_ranges[i].second - work_ranges[i].first, n - k};
                int bigsize[] = {n, n};
                MPI_Datatype subarray;
                MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                MPI_Type_commit(&subarray);
                MPI_Send(&m[0][0], 1, subarray, i, 0, MPI_COMM_WORLD);
                MPI_Type_free(&subarray);
            }
        } else {
            // receive works
            int starts[] = {work_range.first, k};
            int subsizes[] = {work_range.second - work_range.first, n - k};
            int bigsize[] = {n, n};
            MPI_Datatype subarray;
            MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
            MPI_Type_commit(&subarray);
            MPI_Recv(&m[0][0], 1, subarray, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Type_free(&subarray);
        }

        // do works
        for (int i = work_range.first; i < work_range.second; i++) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }

        // gather
        if (id == 0) {
            for (int i = 1; i < std::min(procs, n - (k + 1)); i++) {
                int starts[] = {work_ranges[i].first, k};
                int subsizes[] = {work_ranges[i].second - work_ranges[i].first, n - k};
                int bigsize[] = {n, n};
                MPI_Datatype subarray;
                MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                MPI_Type_commit(&subarray);
                MPI_Recv(&m[0][0], 1, subarray, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Type_free(&subarray);
            }
        } else {
            int starts[] = {work_range.first, k};
            int subsizes[] = {work_range.second - work_range.first, n - k};
            int bigsize[] = {n, n};
            MPI_Datatype subarray;
            MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
            MPI_Type_commit(&subarray);
            MPI_Send(&m[0][0], 1, subarray, 0, 0, MPI_COMM_WORLD);
            MPI_Type_free(&subarray);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void gauss_mpi_comm_reduce_openmp(DynamicMatrix<float> &m, int id, int procs) {
    int n = m.get_rows();

    // only for root process
    std::vector <std::pair<int, int>> work_ranges(procs);

    int i, j, k;
    bool end = false;
    std::pair<int, int> work_range;
#pragma omp parallel num_threads(max_threads), default(none), private(i, j, k), shared(m, n, id, procs, work_ranges, work_range, end)
    for (k = 0; k < n; k++) {
#pragma omp single
        if (!end) {
            // only the root process do the first step
            if (id == 0) {
                for (j = k + 1; j < n; j++) {
                    m[k][j] = m[k][j] / m[k][k];
                }
                m[k][k] = 1;
            }

            // broadcast the k-th row
            MPI_Bcast(&m[k][k], n - k, MPI_FLOAT, 0, MPI_COMM_WORLD);

            work_range = get_work_range(n, k, id, procs);
            if (work_range.first == work_range.second) {
                end = true;
            }

            // scatter
            if (!end) {
                if (id == 0) {
                    // assign works
                    for (i = 1; i < std::min(procs, n - (k + 1)); i++) {
                        work_ranges[i] = get_work_range(n, k, i, procs);
                        int starts[] = {work_ranges[i].first, k};
                        int subsizes[] = {work_ranges[i].second - work_ranges[i].first, n - k};
                        int bigsize[] = {n, n};
                        MPI_Datatype subarray;
                        MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                        MPI_Type_commit(&subarray);
                        MPI_Send(&m[0][0], 1, subarray, i, 0, MPI_COMM_WORLD);
                        MPI_Type_free(&subarray);
                    }
                } else {
                    // receive works
                    int starts[] = {work_range.first, k};
                    int subsizes[] = {work_range.second - work_range.first, n - k};
                    int bigsize[] = {n, n};
                    MPI_Datatype subarray;
                    MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                    MPI_Type_commit(&subarray);
                    MPI_Recv(&m[0][0], 1, subarray, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Type_free(&subarray);
                }
            }

        } else {
            MPI_Bcast(&m[k][k], n - k, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }


        // do works
        if (!end) {
#pragma omp for
            for (i = work_range.first; i < work_range.second; i++) {
                for (j = k + 1; j < n; j++) {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }
                m[i][k] = 0;
            }
        }

#pragma omp single
        if (!end) {
            // gather
            if (id == 0) {
                for (i = 1; i < std::min(procs, n - (k + 1)); i++) {
                    int starts[] = {work_ranges[i].first, k};
                    int subsizes[] = {work_ranges[i].second - work_ranges[i].first, n - k};
                    int bigsize[] = {n, n};
                    MPI_Datatype subarray;
                    MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                    MPI_Type_commit(&subarray);
                    MPI_Recv(&m[0][0], 1, subarray, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Type_free(&subarray);
                }
            } else {
                int starts[] = {work_range.first, k};
                int subsizes[] = {work_range.second - work_range.first, n - k};
                int bigsize[] = {n, n};
                MPI_Datatype subarray;
                MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                MPI_Type_commit(&subarray);
                MPI_Send(&m[0][0], 1, subarray, 0, 0, MPI_COMM_WORLD);
                MPI_Type_free(&subarray);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void gauss_mpi_comm_reduce_openmp_SSE(DynamicMatrix<float> &m, int id, int procs) {
    int n = m.get_rows();

    // only for root process
    std::vector <std::pair<int, int>> work_ranges(procs);

    int i, j, k;
    bool end = false;
    std::pair<int, int> work_range;
#pragma omp parallel num_threads(max_threads), default(none), private(i, j, k), shared(m, n, id, procs, work_ranges, work_range, end)
    for (k = 0; k < n; k++) {
#pragma omp single
        if (!end) {
            // only the root process do the first step
            if (id == 0) {
                // solve the beginning of unaligned columns
                for (j = k + 1; j < ROUND_UP(k + 1, 4); j++) {
                    m[k][j] = m[k][j] / m[k][k];
                }

                // load main diagonal element
                __m128 main_reg = _mm_set1_ps(m[k][k]);

                // using aligned load and store
                for (; j < ROUND_DOWN(n, 4); j += 4) {
                    __m128 dividend_reg = _mm_load_ps(&m[k][j]);
                    __m128 result_reg = _mm_div_ps(dividend_reg, main_reg);
                    _mm_store_ps(&m[k][j], result_reg);
                }

                // solve the end of unaligned columns
                for (; j < n; j++) {
                    m[k][j] = m[k][j] / m[k][k];
                }

                // set the main diagonal of current row to 1
                m[k][k] = 1;
            }

            // broadcast the k-th row
            MPI_Bcast(&m[k][k], n - k, MPI_FLOAT, 0, MPI_COMM_WORLD);

            work_range = get_work_range(n, k, id, procs);
            if (work_range.first == work_range.second) {
                end = true;
            }

            // scatter
            if (!end) {
                if (id == 0) {
                    // assign works
                    for (i = 1; i < std::min(procs, n - (k + 1)); i++) {
                        work_ranges[i] = get_work_range(n, k, i, procs);
                        int starts[] = {work_ranges[i].first, k};
                        int subsizes[] = {work_ranges[i].second - work_ranges[i].first, n - k};
                        int bigsize[] = {n, n};
                        MPI_Datatype subarray;
                        MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                        MPI_Type_commit(&subarray);
                        MPI_Send(&m[0][0], 1, subarray, i, 0, MPI_COMM_WORLD);
                        MPI_Type_free(&subarray);
                    }
                } else {
                    // receive works
                    int starts[] = {work_range.first, k};
                    int subsizes[] = {work_range.second - work_range.first, n - k};
                    int bigsize[] = {n, n};
                    MPI_Datatype subarray;
                    MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                    MPI_Type_commit(&subarray);
                    MPI_Recv(&m[0][0], 1, subarray, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Type_free(&subarray);
                }
            }

        } else {
            MPI_Bcast(&m[k][k], n - k, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }


        // do works
        if (!end) {
#pragma omp for
            for (i = work_range.first; i < work_range.second; i++) {
                // solve the beginning of unaligned columns
                for (j = k + 1; j < ROUND_UP(k + 1, 4); j++) {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }

                // load main diagonal element
                __m128 first_reg = _mm_set1_ps(m[i][k]);

                // using aligned load and store
                for (; j < ROUND_DOWN(n, 4); j += 4) {
                    __m128 t1_reg = _mm_load_ps(&m[k][j]);
                    t1_reg = _mm_mul_ps(first_reg, t1_reg);
                    __m128 t2_reg = _mm_load_ps(&m[i][j]);
                    t2_reg = _mm_sub_ps(t2_reg, t1_reg);
                    _mm_store_ps(&m[i][j], t2_reg);
                }

                // solve the end of unaligned columns
                for (; j < n; j++) {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }

                // set the fisrt element of current row to 0
                m[i][k] = 0;
            }
        }

#pragma omp single
        if (!end) {
            // gather
            if (id == 0) {
                for (i = 1; i < std::min(procs, n - (k + 1)); i++) {
                    int starts[] = {work_ranges[i].first, k};
                    int subsizes[] = {work_ranges[i].second - work_ranges[i].first, n - k};
                    int bigsize[] = {n, n};
                    MPI_Datatype subarray;
                    MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                    MPI_Type_commit(&subarray);
                    MPI_Recv(&m[0][0], 1, subarray, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Type_free(&subarray);
                }
            } else {
                int starts[] = {work_range.first, k};
                int subsizes[] = {work_range.second - work_range.first, n - k};
                int bigsize[] = {n, n};
                MPI_Datatype subarray;
                MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                MPI_Type_commit(&subarray);
                MPI_Send(&m[0][0], 1, subarray, 0, 0, MPI_COMM_WORLD);
                MPI_Type_free(&subarray);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void gauss_mpi_comm_reduce_openmp_AVX(DynamicMatrix<float> &m, int id, int procs) {
    int n = m.get_rows();

    // only for root process
    std::vector <std::pair<int, int>> work_ranges(procs);

    int i, j, k;
    bool end = false;
    std::pair<int, int> work_range;
#pragma omp parallel num_threads(max_threads), default(none), private(i, j, k), shared(m, n, id, procs, work_ranges, work_range, end)
    for (k = 0; k < n; k++) {
#pragma omp single
        if (!end) {
            // only the root process do the first step
            if (id == 0) {
                // solve the beginning of unaligned columns
                for (j = k + 1; j < ROUND_UP(k + 1, 8); j++) {
                    m[k][j] = m[k][j] / m[k][k];
                }

                // load main diagonal element
                __m256 main_reg = _mm256_set1_ps(m[k][k]);

                // using aligned load and store
                for (; j < ROUND_DOWN(n, 8); j += 8) {
                    __m256 dividend_reg = _mm256_load_ps(&m[k][j]);
                    __m256 result_reg = _mm256_div_ps(dividend_reg, main_reg);
                    _mm256_store_ps(&m[k][j], result_reg);
                }

                // solve the end of unaligned columns
                for (; j < n; j++) {
                    m[k][j] = m[k][j] / m[k][k];
                }

                // set the main diagonal of current row to 1
                m[k][k] = 1;
            }

            // broadcast the k-th row
            MPI_Bcast(&m[k][k], n - k, MPI_FLOAT, 0, MPI_COMM_WORLD);

            work_range = get_work_range(n, k, id, procs);
            if (work_range.first == work_range.second) {
                end = true;
            }

            // scatter
            if (!end) {
                if (id == 0) {
                    // assign works
                    for (i = 1; i < std::min(procs, n - (k + 1)); i++) {
                        work_ranges[i] = get_work_range(n, k, i, procs);
                        int starts[] = {work_ranges[i].first, k};
                        int subsizes[] = {work_ranges[i].second - work_ranges[i].first, n - k};
                        int bigsize[] = {n, n};
                        MPI_Datatype subarray;
                        MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                        MPI_Type_commit(&subarray);
                        MPI_Send(&m[0][0], 1, subarray, i, 0, MPI_COMM_WORLD);
                        MPI_Type_free(&subarray);
                    }
                } else {
                    // receive works
                    int starts[] = {work_range.first, k};
                    int subsizes[] = {work_range.second - work_range.first, n - k};
                    int bigsize[] = {n, n};
                    MPI_Datatype subarray;
                    MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                    MPI_Type_commit(&subarray);
                    MPI_Recv(&m[0][0], 1, subarray, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Type_free(&subarray);
                }
            }

        } else {
            MPI_Bcast(&m[k][k], n - k, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }


        // do works
        if (!end) {
#pragma omp for
            for (i = work_range.first; i < work_range.second; i++) {
                // solve the beginning of unaligned columns
                for (j = k + 1; j < ROUND_UP(k + 1, 8); j++) {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }

                // load main diagonal element
                __m256 first_reg = _mm256_set1_ps(m[i][k]);

                // using aligned load and store
                for (; j < ROUND_DOWN(n, 8); j += 8) {
                    __m256 t1_reg = _mm256_load_ps(&m[k][j]);
                    t1_reg = _mm256_mul_ps(first_reg, t1_reg);
                    __m256 t2_reg = _mm256_load_ps(&m[i][j]);
                    t2_reg = _mm256_sub_ps(t2_reg, t1_reg);
                    _mm256_store_ps(&m[i][j], t2_reg);
                }

                // solve the end of unaligned columns
                for (; j < n; j++) {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }

                // set the fisrt element of current row to 0
                m[i][k] = 0;
            }
        }

#pragma omp single
        if (!end) {
            // gather
            if (id == 0) {
                for (i = 1; i < std::min(procs, n - (k + 1)); i++) {
                    int starts[] = {work_ranges[i].first, k};
                    int subsizes[] = {work_ranges[i].second - work_ranges[i].first, n - k};
                    int bigsize[] = {n, n};
                    MPI_Datatype subarray;
                    MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                    MPI_Type_commit(&subarray);
                    MPI_Recv(&m[0][0], 1, subarray, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Type_free(&subarray);
                }
            } else {
                int starts[] = {work_range.first, k};
                int subsizes[] = {work_range.second - work_range.first, n - k};
                int bigsize[] = {n, n};
                MPI_Datatype subarray;
                MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                MPI_Type_commit(&subarray);
                MPI_Send(&m[0][0], 1, subarray, 0, 0, MPI_COMM_WORLD);
                MPI_Type_free(&subarray);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void gauss_mpi_comm_reduce_openmp_AVX512(DynamicMatrix<float> &m, int id, int procs) {
    int n = m.get_rows();

    // only for root process
    std::vector <std::pair<int, int>> work_ranges(procs);

    int i, j, k;
    bool end = false;
    std::pair<int, int> work_range;
#pragma omp parallel num_threads(max_threads), default(none), private(i, j, k), shared(m, n, id, procs, work_ranges, work_range, end)
    for (k = 0; k < n; k++) {
#pragma omp single
        if (!end) {
            // only the root process do the first step
            if (id == 0) {
                // solve the beginning of unaligned columns
                for (j = k + 1; j < ROUND_UP(k + 1, 16); j++) {
                    m[k][j] = m[k][j] / m[k][k];
                }

                // load main diagonal element
                __m512 main_reg = _mm512_set1_ps(m[k][k]);

                // using aligned load and store
                for (; j < ROUND_DOWN(n, 16); j += 16) {
                    __m512 dividend_reg = _mm512_load_ps(&m[k][j]);
                    __m512 result_reg = _mm512_div_ps(dividend_reg, main_reg);
                    _mm512_store_ps(&m[k][j], result_reg);
                }

                // solve the end of unaligned columns
                for (; j < n; j++) {
                    m[k][j] = m[k][j] / m[k][k];
                }

                // set the main diagonal of current row to 1
                m[k][k] = 1;
            }

            // broadcast the k-th row
            MPI_Bcast(&m[k][k], n - k, MPI_FLOAT, 0, MPI_COMM_WORLD);

            work_range = get_work_range(n, k, id, procs);
            if (work_range.first == work_range.second) {
                end = true;
            }

            // scatter
            if (!end) {
                if (id == 0) {
                    // assign works
                    for (i = 1; i < std::min(procs, n - (k + 1)); i++) {
                        work_ranges[i] = get_work_range(n, k, i, procs);
                        int starts[] = {work_ranges[i].first, k};
                        int subsizes[] = {work_ranges[i].second - work_ranges[i].first, n - k};
                        int bigsize[] = {n, n};
                        MPI_Datatype subarray;
                        MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                        MPI_Type_commit(&subarray);
                        MPI_Send(&m[0][0], 1, subarray, i, 0, MPI_COMM_WORLD);
                        MPI_Type_free(&subarray);
                    }
                } else {
                    // receive works
                    int starts[] = {work_range.first, k};
                    int subsizes[] = {work_range.second - work_range.first, n - k};
                    int bigsize[] = {n, n};
                    MPI_Datatype subarray;
                    MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                    MPI_Type_commit(&subarray);
                    MPI_Recv(&m[0][0], 1, subarray, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Type_free(&subarray);
                }
            }

        } else {
            MPI_Bcast(&m[k][k], n - k, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }


        // do works
        if (!end) {
#pragma omp for
            for (i = work_range.first; i < work_range.second; i++) {
                // solve the beginning of unaligned columns
                for (j = k + 1; j < ROUND_UP(k + 1, 16); j++) {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }

                // load main diagonal element
                __m512 first_reg = _mm512_set1_ps(m[i][k]);

                // using aligned load and store
                for (; j < ROUND_DOWN(n, 16); j += 16) {
                    __m512 t1_reg = _mm512_load_ps(&m[k][j]);
                    t1_reg = _mm512_mul_ps(first_reg, t1_reg);
                    __m512 t2_reg = _mm512_load_ps(&m[i][j]);
                    t2_reg = _mm512_sub_ps(t2_reg, t1_reg);
                    _mm512_store_ps(&m[i][j], t2_reg);
                }

                // solve the end of unaligned columns
                for (; j < n; j++) {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }

                // set the fisrt element of current row to 0
                m[i][k] = 0;
            }
        }

#pragma omp single
        if (!end) {
            // gather
            if (id == 0) {
                for (i = 1; i < std::min(procs, n - (k + 1)); i++) {
                    int starts[] = {work_ranges[i].first, k};
                    int subsizes[] = {work_ranges[i].second - work_ranges[i].first, n - k};
                    int bigsize[] = {n, n};
                    MPI_Datatype subarray;
                    MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                    MPI_Type_commit(&subarray);
                    MPI_Recv(&m[0][0], 1, subarray, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Type_free(&subarray);
                }
            } else {
                int starts[] = {work_range.first, k};
                int subsizes[] = {work_range.second - work_range.first, n - k};
                int bigsize[] = {n, n};
                MPI_Datatype subarray;
                MPI_Type_create_subarray(2, bigsize, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &subarray);
                MPI_Type_commit(&subarray);
                MPI_Send(&m[0][0], 1, subarray, 0, 0, MPI_COMM_WORLD);
                MPI_Type_free(&subarray);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}


using gauss_func = std::function<void(DynamicMatrix<float> &, int, int)>;

double test(int n, const gauss_func &gauss, int times, int id, int procs) {
    DynamicMatrix<float> m(n, n, 64);

    std::chrono::duration<double, std::milli> elapsed{};
    for (int i = 0; i < times; i++) {
        if (id == 0) {
            matrix_init(m);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
        gauss(m, id, procs);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed += end - start;
    }
    return elapsed.count() / times;
}

void
normal_test_output(const std::vector <gauss_func> &gauss_funcs,
                   const std::vector <std::string> &names,
                   int times,
                   int begin,
                   int end,
                   int id,
                   int procs) {

    //generate the table bar
    std::cout << "问题规模,";
    for (auto &name: names) {
        std::cout << name << ",";
    }
    std::cout << std::endl;

    //generate the table content
    for (int n = begin; n <= end; n *= 2) {
        std::cout << n << ",";
        for (auto &func: gauss_funcs) {
            std::cout << test(n, func, times, id, procs) << ",";
        }
        std::cout << std::endl;
        std::cout << n << "\tfinished" << std::endl;
    }

}

void
normal_test(const std::vector <gauss_func> &gauss_funcs,
            const std::vector <std::string> &names,
            int times,
            int begin,
            int end,
            int id,
            int procs) {

    for (int n = begin; n <= end; n *= 2) {
        for (auto &func: gauss_funcs) {
            test(n, func, times, id, procs);
        }
    }
}

int main(int argc, char *argv[]) {
//    int provided;
//    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
//    if (provided < MPI_THREAD_MULTIPLE) {
//        MPI_Abort(MPI_COMM_WORLD, 1);
//    }
//
//    const int n = 16;
//    DynamicMatrix<float> m(n, n, 64);
//
//    int id, procs;
//    MPI_Comm_rank(MPI_COMM_WORLD, &id);
//    MPI_Comm_size(MPI_COMM_WORLD, &procs);
//    if (id == 0) {
//        matrix_init(m);
//    }
//    MPI_Barrier(MPI_COMM_WORLD);
//    gauss_mpi_unbalance(m, id, procs);
//    if (id == 0) {
//        matrix_print(m);
//    }
//    MPI_Finalize();
//    return 0;

//    const int n = 4096;
//    DynamicMatrix<float> m(n, n, 64);
//    matrix_init(m);
//    std::cout << "gauss start" << std::endl;
//    auto start = std::chrono::high_resolution_clock::now();
//    gauss_thread_worker_block(m);
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
//    std::cout << "gauss_omp_AVX512" << std::endl;
//    gauss_omp_AVX512(m2);
//    matrix_print(m2);
//    return 0;


    std::vector <gauss_func> gauss_funcs = {
            gauss_normal,
            gauss_mpi_unbalance,
            gauss_mpi_normal,
            gauss_mpi_comm_reduce,
            gauss_mpi_comm_reduce_openmp,
            gauss_mpi_comm_reduce_openmp_SSE,
            gauss_mpi_comm_reduce_openmp_AVX,
            gauss_mpi_comm_reduce_openmp_AVX512,
    };

    std::vector <std::string> names = {
            "normal",
            "MPI_unbalance",
            "MPI_normal",
            "MPI_comm_reduce",
            "MPI_comm_reduce_openmp",
            "MPI_comm_reduce_openmp_SSE",
            "MPI_comm_reduce_openmp_AVX",
            "MPI_comm_reduce_openmp_AVX512",
    };

    const int times = 10;
    const int begin = 16;
    const int end = 1024;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int id, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    if (id == 0) {
        normal_test_output(gauss_funcs, names, times, begin, end, id, procs);
    } else {
        normal_test(gauss_funcs, names, times, begin, end, id, procs);
    }

    MPI_Finalize();
    return 0;
}
