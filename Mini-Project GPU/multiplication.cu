#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>

#define CUDA_DEVICE_NUMBER 0 		// The ID of the GPU that is used

#define NB_THREADS_PER_BLOCK 128	// The NB of threads per block
#define NB_ARRAYS_MEMORY 3.0 		// The number of arrays that will be allocated in memory (a, b, c)

#define NB_ELEMENTS_PER_THREAD 1	// The number of multiplications a single thread will do
#define CI_FACTOR 1					// The number of times a thread will do the multiplication operation.
									// Increasing this will increase computational intensity.
									// (Number of operations will grow while memory will stay constant).

#define DATATYPE float				// The datatype of the arrays. Feel free to change this to any type that supports multiplication.

using namespace std;

void multiplication_iteration(const int array_size);
void initialize_arrays(DATATYPE* a, DATATYPE* b, int arr_size);
long long cpu_computation(DATATYPE* a, DATATYPE* b, DATATYPE* cpu_res, int arr_size);
float gpu_computation(DATATYPE* host_a, DATATYPE* host_b, DATATYPE* host_c, DATATYPE* cpu_results, int nb_elements_per_thread, int arr_size);
bool check_results(DATATYPE* cpu, DATATYPE* gpu, int arr_size);
void display_metrics(long long cpu_runtime_microsec, float gpu_runtime_millisec, int arr_size);
void print_CUDA_device_details();

__global__ void gpu_multiply_kernel(const DATATYPE* a, const DATATYPE* b, DATATYPE* c, int nb_elements_per_thread, int arr_size) {
	int globalId = blockDim.x * blockIdx.x + threadIdx.x;

	int idx = globalId * nb_elements_per_thread;

	for (int offset = 0; offset < nb_elements_per_thread; offset++) {

		if (idx + offset >= arr_size) {
			return;
		}

		// Loop on operations to increase CI
		for (int nbOperations = 0; nbOperations < CI_FACTOR; nbOperations++) {
			c[idx + offset] = a[idx + offset] * b[idx + offset];
		}
	}
}

int main() {
	int iterations_sizes[] = { 10, 100, 1000, 10000, 25000, 50000, 75000, 100000, 250000, 500000, 1000000, 2500000 };
	int nbElements = sizeof(iterations_sizes) / sizeof(int);

	print_CUDA_device_details();

	for (int i = 0; i < nbElements; i++) {
		multiplication_iteration(iterations_sizes[i]);
	}

	return 0;
}

void multiplication_iteration(const int array_size) {
	DATATYPE* host_a = new DATATYPE[array_size];
	DATATYPE* host_b = new DATATYPE[array_size];

	DATATYPE* host_c = new DATATYPE[array_size];
	DATATYPE* host_cpu_res = new DATATYPE[array_size];

	initialize_arrays(host_a, host_b, array_size);

	long long cpu_runtime_microsec = cpu_computation(host_a, host_b, host_cpu_res, array_size);
	float gpu_runtime_millisec = gpu_computation(host_a, host_b, host_c, host_cpu_res, NB_ELEMENTS_PER_THREAD, array_size);

	display_metrics(cpu_runtime_microsec, gpu_runtime_millisec, array_size);

	delete[] host_a;
	delete[] host_b;
	delete[] host_c;
	delete[] host_cpu_res;
}

long long cpu_computation(DATATYPE* a, DATATYPE* b, DATATYPE* cpu_res, int arr_size) {
	chrono::steady_clock::time_point start_cpu = chrono::high_resolution_clock::now();

	for (int i = 0; i < arr_size; i++) {
		cpu_res[i] = a[i] * b[i];
	}

	chrono::steady_clock::time_point stop_cpu = chrono::high_resolution_clock::now();
	auto cpu_runtime_us = chrono::duration_cast<chrono::microseconds>(stop_cpu - start_cpu).count();

	return cpu_runtime_us;
}

float gpu_computation(DATATYPE* host_a, DATATYPE* host_b, DATATYPE* host_c, DATATYPE* cpu_results, int nb_elements_per_thread, int arr_size) {
	DATATYPE* device_a = 0;
	DATATYPE* device_b = 0;
	DATATYPE* device_c = 0;

	cudaEvent_t start_gpu, stop_gpu;
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&stop_gpu);
 
	dim3 block_size(((arr_size / nb_elements_per_thread) + (NB_THREADS_PER_BLOCK - 1)) / NB_THREADS_PER_BLOCK);
	dim3 thread_size(NB_THREADS_PER_BLOCK);

	cudaSetDevice(CUDA_DEVICE_NUMBER);

	cudaMalloc((void**)&device_a, arr_size * sizeof(DATATYPE));
	cudaMalloc((void**)&device_b, arr_size * sizeof(DATATYPE));
	cudaMalloc((void**)&device_c, arr_size * sizeof(DATATYPE));

	cudaMemcpy(device_a, host_a, arr_size * sizeof(DATATYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, arr_size * sizeof(DATATYPE), cudaMemcpyHostToDevice);

	cudaEventRecord(start_gpu);
	gpu_multiply_kernel << <block_size, thread_size >> > (device_a, device_b, device_c, nb_elements_per_thread, arr_size);
	cudaEventRecord(stop_gpu);

	cudaDeviceSynchronize();

	// Copy output vector from GPU buffer to host memory.
	cudaMemcpy(host_c, device_c, arr_size * sizeof(DATATYPE), cudaMemcpyDeviceToHost);

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

void initialize_arrays(DATATYPE* a, DATATYPE* b, int arr_size) {
	for (int i = 0; i < arr_size; i++) {
		a[i] = i;
		b[i] = arr_size - i;
	}
}

void display_metrics(long long cpu_runtime_microsec, float gpu_runtime_millisec, int arr_size) {
	cout << "Metrics for an array of " << arr_size << " elements" << endl;

	//cout << "\tCPU time :" << cpu_runtime_microsec << " us" << endl;
	//cout << "\tGPU time : " << gpu_runtime_millisec * 1000 << " us" << endl;

	//float speedup = cpu_runtime_microsec / (gpu_runtime_millisec * 1000);
	//cout << "\tSpeedup : " << speedup << " %" << endl << endl;

	float memoryUsed = NB_ARRAYS_MEMORY * arr_size * sizeof(DATATYPE);
	float memoryThroughput = memoryUsed / gpu_runtime_millisec / 1e+6; //Divide by 1 000 000 to have GB/s

	float numOperation = 1.0 * arr_size;
	float computationThroughput = numOperation / gpu_runtime_millisec / 1e+6; //Divide by 1 000 000 to have GOPS/s

	cout << "\tMemory throughput : " << memoryThroughput << " GB/s " << endl;
	cout << "\tComputation throughput : " << computationThroughput << " GOPS/s " << endl;
}

bool check_results(DATATYPE* cpu, DATATYPE* gpu, int arr_size) {
	for (int i = 0; i < arr_size; i++) {
		if (cpu[i] != gpu[i]) {
			printf("ERROR (%dth element): cpu : %d != gpu : %d \n", i, cpu[i], gpu[i]);
			return false;
		}
	}

	return true;
}

void print_CUDA_device_details() {
	struct cudaDeviceProp props;
	cudaGetDeviceProperties(&props, CUDA_DEVICE_NUMBER);

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
