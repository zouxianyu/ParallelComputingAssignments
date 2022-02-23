#include <iostream>
#include <chrono>
#include <ratio>

int main() {

    auto start = std::chrono::steady_clock::now();

    int temp;
    for (int i = 0; i < 1000; ++i) {
        temp = i;
    }

    auto end = std::chrono::steady_clock::now();

    std::cout << std::chrono::duration <double, std::milli> (end - start).count() << " ms" << std::endl;
}
