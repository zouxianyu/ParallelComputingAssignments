#include <iostream>
#include <chrono>
#include <ratio>
#include <cassert>
#include <numeric>

const int N = (8192<<10)/sizeof(int64_t);//cache size
int64_t V[N];
int64_t S;//avoid overflow

/*
 * vector initialization
 */
void init() {
    for (int i = 0; i < N; ++i) {
        V[i] = i;
    }
}

/*
 * naive algorithm
 */
void naive_sum() {
    S = 0;
    for (int i = 0; i < N; i++) {
        S += V[i];
    }
}

/*
 * multiplexing algorithm
 */
void multiplexing2_sum() {
    int64_t s1 = 0;
    int64_t s2 = 0;
    S = 0;
    for (int i = 0; i < N; i += 2) {
        s1 += V[i];
        s2 += V[i + 1];
    }
    S = s1 + s2;
}

void multiplexing4_sum() {
    int64_t s1 = 0;
    int64_t s2 = 0;
    int64_t s3 = 0;
    int64_t s4 = 0;
    S = 0;
    for (int i = 0; i < N; i += 4) {
        s1 += V[i];
        s2 += V[i + 1];
        s3 += V[i + 2];
        s4 += V[i + 3];
    }
    S = s1 + s2 + s3 + s4;
}

/*
 * recursive algorithm
 */
void recursive_sum() {
    for (int m = N; m > 1; m /= 2) {
        for (int i = 0; i < m / 2; i++) {
            V[i] = V[2 * i] + V[2 * i + 1];
        }
    }
    S = V[0];
}

/*
 * test wrapper function
 */
void test(void (*func)(), int n) {
    init();
    int64_t result = std::accumulate(V, V + N, 0l);

    std::chrono::duration<double, std::milli> elapsed{};

    for (int i = 0; i < n; i++) {
        init();
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        elapsed += end - start;
    }
    std::cout << "avg time: " << elapsed.count() / n << " ms" << std::endl;
    assert(S==result);
}

/*
 * main function
 */
int main(int argc, char **argv) {

    if (argc != 3) {
        std::cout << "invalid arguments" << std::endl;
        return 1;
    }


    //check the method argument
    if (std::string(argv[1]) == "naive") {
        test(naive_sum, atol(argv[2]));
    } else if (std::string(argv[1]) == "mul2") {
        test(multiplexing2_sum, atol(argv[2]));
    } else if (std::string(argv[1]) == "mul4") {
        test(multiplexing4_sum, atol(argv[2]));
    } else if (std::string(argv[1]) == "rec") {
        test(recursive_sum, atol(argv[2]));
    } else {
        std::cout << "invalid arguments" << std::endl;
        return 1;
    }

    return 0;
}