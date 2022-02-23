#include <iostream>
#include <chrono>
#include <ratio>

const int N = 10000;
double M[N][N];
double V[N];
double S[N];

/*
 * matrix and vector initialization
 */
void init() {
    //matrix initialization
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            M[i][j] = i + j;
        }
    }

    //vector initialization
    for (int i = 0; i < N; i++) {
        V[i] = i;
    }
}

/*
 * optimized algorithm
 */
void row_major_calc() {
    for (int i = 0; i < N; i++) {
        S[i] = 0;
    }
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            S[i] += M[j][i] * V[j];
        }
    }
}

/*
 * naive algorithm
 */
void col_major_calc() {
    for (int i = 0; i < N; i++) {
        S[i] = 0;
        for (int j = 0; j < N; j++) {
            S[i] += M[j][i] * V[j];
        }
    }
}

/*
 * test wrapper function
 */
void test(void (*func)(), int n) {
    init();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "avg time: " << elapsed.count() / n << " ms" << std::endl;
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
    if (std::string(argv[1]) == "row") {
        test(row_major_calc, atol(argv[2]));
    } else if (std::string(argv[1]) == "col") {
        test(col_major_calc, atol(argv[2]));
    } else {
        std::cout << "invalid arguments" << std::endl;
        return 1;
    }

    return 0;
}
