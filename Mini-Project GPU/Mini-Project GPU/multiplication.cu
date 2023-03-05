#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>

#define NB_THREADS 1024

#define NB_COMPUTATIONS 4
#define NB_ARRAYS_MEMORY 3

using namespace std;

void multiplication_iteration(const int array_size);
void initialize_arrays(int* a, int* b, int arr_size);
long long cpu_computation(int* a, int* b, int* cpu_res, int arr_size);
float gpu_computation(int* host_a, int* host_b, int* host_c, int* cpu_results, int arr_size);
bool check_results(int* cpu, int* gpu, int arr_size);
void display_metrics(long long cpu_runtime_microsec, float gpu_runtime_millisec, int arr_size);
void printCUDADeviceDetails();

__global__ void gpu_multiply_kernel(const int* a, const int* b, int* c, int arr_size) {
	int globalId = blockDim.x * blockIdx.x + threadIdx.x; // 2 operations

	if (globalId >= arr_size) { // 1 operation
		return;
	}

	c[globalId] = a[globalId] * b[globalId]; // 1 operation
}

int main() {
	int iterations_sizes[] = { 10, 100, 1000, 10000, 25000, 50000, 75000, 100000, 250000, 500000, 1000000, 2500000, 5000000, 7500000, 10000000, 50000000, 100000000, 200000000 };
	int nbElements = sizeof(iterations_sizes) / sizeof(int);

	for (int i = 0; i < nbElements; i++) {
		multiplication_iteration(iterations_sizes[i]);
		cout << "===================================================" << endl;
	}

	return 0;
}


void multiplication_iteration(const int array_size) {
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
}

long long cpu_computation(int* a, int* b, int* cpu_res, int arr_size) {
	chrono::steady_clock::time_point start_cpu = chrono::high_resolution_clock::now();

	for (int i = 0; i < arr_size; i++) {
		cpu_res[i] = a[i] * b[i];
	}

	chrono::steady_clock::time_point stop_cpu = chrono::high_resolution_clock::now();
	auto cpu_runtime_us = chrono::duration_cast<chrono::microseconds>(stop_cpu - start_cpu).count();

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

	cout << "Metrics for an array of " << arr_size << " elements" << endl;

	cout << "\tCPU time :" << cpu_runtime_microsec << " us" << endl;
	cout << "\tGPU time : " << gpu_runtime_millisec * 1000 << " us" << endl;

	float speedup = cpu_runtime_microsec / (gpu_runtime_millisec * 1000);
	cout << "\tSpeedup : " << speedup << " %" << endl << endl;

	float memoryUsed = 3.0 * arr_size * sizeof(int);
	float memoryThroughput = memoryUsed / gpu_runtime_millisec / 1e+6; //Divide by 1 000 000 to have GB/s

	float numOperation = 1.0 * arr_size;
	float computationThroughput = numOperation / gpu_runtime_millisec / 1e+6; //Divide by 1 000 000 to have GOPS/s

	cout << "\tMemory throughput : " << memoryThroughput << " GB/s " << endl;
	cout << "\tComputation throughput : " << computationThroughput << " GOPS/s " << endl;

	// CI = C / M
	float c = NB_COMPUTATIONS;
	float m = NB_ARRAYS_MEMORY * sizeof(int);

	cout << "\tCI = " << c << "/" << m << " = " << (c / m) << endl;
}

bool check_results(int* cpu, int* gpu, int arr_size) {
	for (int i = 0; i < arr_size; i++) {
		if (cpu[i] != gpu[i]) {
			printf("ERROR (%dth element): cpu : %d != gpu : %d \n", i, cpu[i], gpu[i]);
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

	cout << "CUDA Device :" << endl;

	cout << "\tName : " << props.name << endl;
	cout << "\tNumber of Multiprocessors : " << props.multiProcessorCount << endl;
	cout << "\tTotal Global memory (bytes) : " << props.totalGlobalMem << endl;
	cout << "\tPeak Memory Bandwidth(GB / s) : "  << (2.0 * props.memoryClockRate * (props.memoryBusWidth / 8)/ 1.0e6) << endl;
	cout << "\tShared memory / SM (kB) : " << (props.sharedMemPerMultiprocessor / 1000) << endl;
	cout << "\tWarp size : " << props.warpSize << endl;
	cout << "\tMax. threads/block : " << props.maxThreadsPerBlock << endl;
	cout << "\tMax. threads/SM : " << props.maxThreadsPerMultiProcessor << endl;
	cout << "\tMax. blocks/SM : " << props.maxBlocksPerMultiProcessor << endl;
	cout << "\tCompute capability version number : " << props.major << "." << props.minor << endl << endl;
}
