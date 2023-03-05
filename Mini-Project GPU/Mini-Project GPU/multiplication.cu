#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>

#define NB_THREADS 128

void initialize_arrays(int* a, int* b, int arr_size);
long long cpu_computation(int* a, int* b, int* cpu_res, int arr_size);
float gpu_computation(int* host_a, int* host_b, int* host_c, int* cpu_results, int arr_size);
bool check_results(int* cpu, int* gpu, int arr_size);
void display_metrics(long long cpu_runtime_microsec, float gpu_runtime_millisec, int arr_size);
void printCUDADeviceDetails();

__global__ void gpu_multiply_kernel(const int* a, const int* b, int* c, int arr_size) {
	int globalId = blockDim.x * blockIdx.x + threadIdx.x;

	if (globalId >= arr_size) {
		return;
	}

	c[globalId] = a[globalId] * b[globalId];
}

int main() {
	const int array_size = 100000;

	int* host_a = new int[array_size];
	int* host_b = new int[array_size];

	int* host_c = new int[array_size];
	int* host_cpu_res = new int[array_size];

	initialize_arrays(host_a, host_b, array_size);

	long long cpu_runtime_microsec = cpu_computation(host_a, host_b, host_cpu_res, array_size);
	float gpu_runtime_millisec = gpu_computation(host_a, host_b, host_c, host_cpu_res, array_size);

	display_metrics(cpu_runtime_microsec, gpu_runtime_millisec, array_size);

	delete[] host_a;
	delete[] host_b;
	delete[] host_c;
	delete[] host_cpu_res;

	return 0;
}

long long cpu_computation(int* a, int* b, int* cpu_res, int arr_size) {
	std::chrono::steady_clock::time_point start_cpu = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < arr_size; i++) {
		cpu_res[i] = a[i] * b[i];
	}
	std::chrono::steady_clock::time_point stop_cpu = std::chrono::high_resolution_clock::now();
	auto cpu_runtime_us = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu).count();

	return cpu_runtime_us;
}

float gpu_computation(int* host_a, int* host_b, int* host_c, int* cpu_results, int arr_size) {
	int* device_a = 0;
	int* device_b = 0;
	int* device_c = 0;

	cudaEvent_t start_gpu, stop_gpu;
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&stop_gpu);

	// Define the size of the grid (block_size = #blocks in the grid)
	//and the size of a block (thread_size = #threads in a block)
	//TODO 1) Change how block_size and thread_size are defined to work with bigger vectors 
	dim3 block_size((arr_size + (NB_THREADS - 1)) / NB_THREADS);
	dim3 thread_size(NB_THREADS);

	cudaSetDevice(0);

	cudaMalloc((void**)&device_a, arr_size * sizeof(int));
	cudaMalloc((void**)&device_b, arr_size * sizeof(int));
	cudaMalloc((void**)&device_c, arr_size * sizeof(int));

	cudaMemcpy(device_a, host_a, arr_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, arr_size * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(start_gpu);
	gpu_multiply_kernel <<<block_size, thread_size>>> (device_a, device_b, device_c, arr_size);
	cudaEventRecord(stop_gpu);

	cudaDeviceSynchronize();


	// Copy output vector from GPU buffer to host memory.
	cudaMemcpy(host_c, device_c, arr_size * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop_gpu);
	float gpu_runtime_ms;
	cudaEventElapsedTime(&gpu_runtime_ms, start_gpu, stop_gpu);

	if (!check_results(cpu_results, host_c, arr_size)) {
		printf("GPU results are invalid !!!\n\n");
	}

	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);

	cudaDeviceReset();

	return gpu_runtime_ms;
}

void initialize_arrays(int* a, int* b, int arr_size) {
	for (int i = 0; i < arr_size; i++) {
		a[i] = i;
		b[i] = arr_size - i;
	}
}

void display_metrics(long long cpu_runtime_microsec, float gpu_runtime_millisec, int arr_size) {
	printCUDADeviceDetails();

	std::cout << "Metrics for an array of " << arr_size << " elements" << std::endl;

	std::cout << "\tCPU time :" << cpu_runtime_microsec << " us" << std::endl;
	std::cout << "\tGPU time : " << gpu_runtime_millisec * 1000 << " us" << std::endl;

	float speedup = cpu_runtime_microsec / (gpu_runtime_millisec * 1000);
	std::cout << "\tSpeedup : " << speedup << " %" << std::endl << std::endl;

	float memoryUsed = 3.0 * arr_size * sizeof(int);
	float memoryThroughput = memoryUsed / gpu_runtime_millisec / 1e+6; //Divide by 1 000 000 to have GB/s

	float numOperation = 1.0 * arr_size;
	float computationThroughput = numOperation / gpu_runtime_millisec / 1e+6; //Divide by 1 000 000 to have GOPS/s

	std::cout << "\tMemory throughput : " << memoryThroughput << " GB/s " << std::endl;
	std::cout << "\tComputation throughput : " << computationThroughput << " GOPS/s " << std::endl;
}

bool check_results(int* cpu, int* gpu, int arr_size) {
	for (int i = 0; i < arr_size; i++) {
		if (cpu[i] != gpu[i]) {
			printf("ERROR : cpu : %d != gpu : %d \n", cpu[i], gpu[i]);
			return false;
		}
	}

	return true;
}

void printCUDADeviceDetails() {
	int device;
	cudaGetDevice(&device);

	struct cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);

	std::cout << "CUDA Device :" << std::endl;

	std::cout << "\tName : " << props.name << std::endl;
	std::cout << "\tTotal Global memory (bytes) : " << props.totalGlobalMem << std::endl;
	std::cout << "\tWarp size : " << props.warpSize << std::endl;
	std::cout << "\tMax. threads/block : " << props.maxThreadsPerBlock << std::endl;
	std::cout << "\tMax. threads/SM : " << props.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "\tMax. blocks/SM : " << props.maxBlocksPerMultiProcessor << std::endl << std::endl;
}
