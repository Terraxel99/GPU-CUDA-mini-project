#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>

void initialize_arrays(int* a, int* b, int arr_size);
long long cpu_computation(int* a, int* b, int* cpu_res, int arr_size);

int main() {
	const int array_size = 5;

	int* host_a = new int[array_size];
	int* host_b = new int[array_size];

	int* host_c = new int[array_size];
	int* host_cpu_res = new int[array_size];

	initialize_arrays(host_a, host_b, array_size);

	long long cpu_runtime_microsec = cpu_computation(host_a, host_b, host_cpu_res, array_size);

	delete[] host_a;
	delete[] host_b;
	delete[] host_c;
	delete[] host_cpu_res;
}

long long cpu_computation(int* a, int* b, int* cpu_res, int arr_size) {
	std::chrono::steady_clock::time_point start_cpu = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < arr_size; i++) {
		cpu_res[i] = a[i] + b[i];
	}
	std::chrono::steady_clock::time_point stop_cpu = std::chrono::high_resolution_clock::now();
	auto cpu_runtime_us = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu).count();

	return cpu_runtime_us;
}

void initialize_arrays(int* a, int* b, int arr_size) {
	for (int i = 0; i < arr_size; i++) {
		a[i] = i;
		b[i] = arr_size - i;
	}
}