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

void matrix_print(buffer<float, 2>& buf) {
	host_accessor m{ buf ,read_only };
	auto range = m.get_range();
	for (int i = 0; i < range[0]; i++) {
		for (int j = 0; j < range[1]; j++) {
			std::cout << std::setw(16) << m[i][j];
		}
		std::cout << std::endl;
	}
}

void matrix_copy(buffer<float, 2>& to, buffer<float, 2>& from) {
	host_accessor ms{ from ,read_only };
	host_accessor md{ to ,write_only };
	assert(ms.get_range() == md.get_range());
	auto range = ms.get_range();
	for (int i = 0; i < range[0]; i++) {
		for (int j = 0; j < range[1]; j++) {
			md[i][j] = ms[i][j];
		}
	}
}

void matrix_init(buffer<float, 2>& buf) {
	host_accessor m{ buf ,read_write };
	int n = m.get_range()[0];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			m[i][j] = 0;
		}
	}
	int lower_bound = std::max(1, n / 3);
	int upper_bound = std::min(n - 1, n * 2 / 3);
	for (int i = lower_bound; i < upper_bound; i++) {
		for (int j = lower_bound; j < upper_bound; j++) {
			m[i][j] = 30.0;
		}
	}
}

void alg_naive(buffer<float, 2>& buf, float dx2, float dy2, float a, float dt, int steps, queue& q) {

	int n = buf.get_range()[0];
	buffer<float, 2> buf2(range(n, n));

	// copy all data contain edges
	matrix_copy(buf2, buf);

	for (int k = 0; k < steps; k++) {

		// scoped accessor
		{
			// create two accessor for swap usage
			host_accessor acc_curr{ buf2, write_only };
			host_accessor acc_prev{ buf, read_only };

			for (int i = 1; i < n - 1; i++) {
				for (int j = 1; j < n - 1; j++) {
					// 2D heat equation
					acc_curr[i][j] = acc_prev[i][j] + a * dt * (
						(acc_prev[i][j - 1] - 2.0 * acc_prev[i][j] + acc_prev[i][j + 1]) / dx2 +
						(acc_prev[i - 1][j] - 2.0 * acc_prev[i][j] + acc_prev[i + 1][j]) / dy2
						);
				}
			}
		}

		// copy back the buffer
		matrix_copy(buf, buf2);
	}
}

void alg_swap(buffer<float, 2>& buf, float dx2, float dy2, float a, float dt, int steps, queue& q) {

	int n = buf.get_range()[0];
	buffer<float, 2> buf_swap(range(n, n));

	// copy all data contain edges
	matrix_copy(buf_swap, buf);

	for (int k = 0; k < steps; k++) {
		// create two accessor for swap usage
		host_accessor acc_curr{ (k % 2 == 1) ? buf : buf_swap, write_only };
		host_accessor acc_prev{ (k % 2 == 0) ? buf : buf_swap, read_only };

		for (int i = 1; i < n - 1; i++) {
			for (int j = 1; j < n - 1; j++) {
				// 2D heat equation
				acc_curr[i][j] = acc_prev[i][j] + a * dt * (
					(acc_prev[i][j - 1] - 2.0 * acc_prev[i][j] + acc_prev[i][j + 1]) / dx2 +
					(acc_prev[i - 1][j] - 2.0 * acc_prev[i][j] + acc_prev[i + 1][j]) / dy2
					);
			}
		}
	}

	// copy back from the swap buffer
	if (steps % 2 == 1) {
		matrix_copy(buf, buf_swap);
	}
}

void alg_oneapi(buffer<float, 2>& buf, const float dx2, const float dy2, const float a, const float dt, const int steps, queue& q) {

	int n = buf.get_range()[0];
	buffer<float, 2> buf_swap(range(n, n));

	// copy all data contain edges
	matrix_copy(buf_swap, buf);

	for (int k = 0; k < steps; k++) {
		q.submit([&](handler& h) {
			// create two accessor for swap usage
			accessor acc_curr{ (k % 2 == 1) ? buf : buf_swap, h, write_only };
			accessor acc_prev{ (k % 2 == 0) ? buf : buf_swap, h, read_only };

			// use parallel_for
			h.parallel_for(range(n - 2, n - 2), [=](auto idx) {
				int i = 1 + idx.get_id(0);
				int j = 1 + idx.get_id(1);
				
				acc_curr[i][j] = acc_prev[i][j] + a * dt * (
					(acc_prev[i][j - 1] - 2.0 * acc_prev[i][j] + acc_prev[i][j + 1]) / dx2 +
					(acc_prev[i - 1][j] - 2.0 * acc_prev[i][j] + acc_prev[i + 1][j]) / dy2
					);
				});
			});
	}
	q.wait();

	// copy back from the swap buffer
	if (steps % 2 == 1) {
		matrix_copy(buf, buf_swap);
	}
}

using test_func = std::function<void(buffer<float, 2>&, queue&)>;

//double test(int n, const gauss_func& gauss, int times, queue& q) {
//	buffer<float, 2> buf(range(n, n));
//
//	// warm up
//	matrix_init(buf);
//	gauss(buf, q);
//
//	std::chrono::duration<double, std::milli> elapsed{};
//	for (int i = 0; i < times; i++) {
//		matrix_init(buf);
//		auto start = std::chrono::high_resolution_clock::now();
//		gauss(buf, q);
//		auto end = std::chrono::high_resolution_clock::now();
//		elapsed += end - start;
//	}
//	return elapsed.count() / times;
//}
//
//void
//normal_test_output(const std::vector<gauss_func>& gauss_funcs,
//	const std::vector<std::string>& names,
//	int times,
//	int begin,
//	int end,
//	queue& q) {
//
//	//generate the table bar
//	std::cout << "问题规模,";
//	for (auto& name : names) {
//		std::cout << name << ",";
//	}
//	std::cout << std::endl;
//
//	//generate the table content
//	for (int n = begin; n <= end; n *= 2) {
//		std::cout << n << ",";
//		for (auto& func : gauss_funcs) {
//			std::cout << test(n, func, times, q) << ",";
//		}
//		std::cout << std::endl;
//	}
//}

int main() {

	queue q;
	device my_device = q.get_device();
	std::cout << "Device: " << my_device.get_info<info::device::name>() << std::endl;

	const int n = 50;
	buffer<float, 2> buf(range(n, n));

	// warm up
	matrix_init(buf);
	//matrix_print(buf);
	//std::cout << std::endl;
	alg_oneapi(buf, 1.0, 1.0, 1.0, 0.00001, 500, q);
	//matrix_print(buf);

	//std::vector<gauss_func> gauss_funcs = {
	//		gauss_normal,
	//		gauss_oneapi,
	//};

	//std::vector<std::string> names = {
	//		"normal",
	//		"oneapi",
	//};

	//const int times = 10;
	//const int begin = 16;
	//const int end = 1024;

	//normal_test_output(gauss_funcs, names, times, begin, end, q);
}
