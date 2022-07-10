#include <iostream>
#include <fstream>
#include <chrono>
#include <ratio>
#include <iomanip>
#include <random>
#include <thread>
#include <functional>

#include <CL/sycl.hpp>

using namespace cl::sycl;

void matrix_print(float *m, int n) {
	
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			std::cout << std::setw(16) << m[i*n + j];
		}
		std::cout << std::endl;
	}
}

void matrix_copy(float* md, float* ms, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			md[i*n+j] = ms[i*n+j];
		}
	}
}

void matrix_init(float* m, int n) {

	static std::default_random_engine generator(1337);
	static std::uniform_real_distribution<float> distribution(-1.0, 1.0);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++)
			m[i*n+j] = 0;
		m[i*n+i] = 1.0;
		for (int j = i + 1; j < n; j++)
			m[i * n + j] = distribution(generator);
	}
	for (int k = 0; k < n; k++) {
		for (int i = k + 1; i < n; i++) {
			for (int j = 0; j < n; j++) {
				m[i * n + j] += m[k * n + j];
			}
		}
	}
}

void gauss_normal(float* m, int n, queue& q) {
	for (int k = 0; k < n; k++) {
		for (int j = k + 1; j < n; j++) {
			m[k * n + j] = m[k * n + j] / m[k * n + k];
		}
		m[k * n + k] = 1;
		for (int i = k + 1; i < n; i++) {
			for (int j = k + 1; j < n; j++) {
				m[i * n + j] = m[i * n + j] - m[i * n + k] * m[k * n + j];
			}
			m[i * n + k] = 0;
		}
	}
}

void gauss_oneapi(float* m, int n, queue& q) {

	//device my_device = q.get_device();
	//std::cout << "Device: " << my_device.get_info<info::device::name>() << std::endl;

	for (int k = 0; k < n; k++) {

		q.submit([&](handler& h) {
			h.parallel_for(range(n - k), [=](auto idx) {
				int j = k + idx;
				m[k*n+j] = m[k * n + j] / m[k * n + k];
				});
			});

		q.submit([&](handler& h) {
			h.parallel_for(range(n - (k + 1), n - (k + 1)), [=](auto idx) {
				int i = k + 1 + idx.get_id(0);
				int j = k + 1 + idx.get_id(1);
				m[i * n + j] = m[i * n + j] - m[i * n + k] * m[k * n + j];
				});
			});

		q.submit([&](handler& h) {
			h.parallel_for(range(n - (k + 1)), [=](auto idx) {
				int i = k + 1 + idx;
				m[i * n + k] = 0;
				});
			});
	}
	q.wait();
}

using gauss_func = std::function<void(float* m, int n, queue& q)>;

double test(int n, const gauss_func& gauss, int times, queue& q) {
	float* buf = (float*)malloc_shared(n * n * sizeof(float), q);

	//warm up
	matrix_init(buf, n);
	gauss(buf, n, q);

	std::chrono::duration<double, std::milli> elapsed{};
	for (int i = 0; i < times; i++) {
		matrix_init(buf, n);
		auto start = std::chrono::high_resolution_clock::now();
		gauss(buf,n,q);
		auto end = std::chrono::high_resolution_clock::now();
		elapsed += end - start;
	}

	double t = elapsed.count() / times;
	free(buf, q);
	return t;
}

void
normal_test_output(const std::vector<gauss_func>& gauss_funcs,
	const std::vector<std::string>& names,
	int times,
	int begin,
	int end,
	queue& q) {

	//generate the table bar
	std::cout << "n,";
	for (auto& name : names) {
		std::cout << name << ",";
	}
	std::cout << std::endl;

	//generate the table content
	for (int n = begin; n <= end; n *= 2) {
		std::cout << n << ",";
		for (auto& func : gauss_funcs) {
			std::cout << test(n, func, times, q) << ",";
		}
		std::cout << std::endl;
	}
}

int main() {

	queue q{ property::queue::in_order() };
	device my_device = q.get_device();
	std::cout << "Device: " << my_device.get_info<info::device::name>() << std::endl;

	std::vector<gauss_func> gauss_funcs = {
			//gauss_normal,
			gauss_oneapi,
	};

	std::vector<std::string> names = {
			//"normal",
			"oneapi",
	};

	const int times = 10;
	const int begin = 16;
	const int end = 1024;


	normal_test_output(gauss_funcs, names, times, begin, end, q);
}

