#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define EPSILON 1e-5

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void fill_matrix(float* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = rand() % 10;
    }
}

void multiply_cpu(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

__global__ void multiply_gpu_kernel(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

bool check_result(float* C_cpu, float* C_gpu, int n) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(C_cpu[i] - C_gpu[i]) > EPSILON) {
            return false;
        }
    }
    return true;
}

int main() {
    int sizes[] = {100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("========================================================================\n");
    printf("Результаты умножения матриц (CPU vs GPU)\n");
    printf("========================================================================\n");
    printf("%8s | %12s | %12s | %10s\n", "N", "CPU (сек)", "GPU (сек)", "Speedup");
    printf("------------------------------------------------------------------------\n");
    
    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        size_t bytes = n * n * sizeof(float);
        
        float* A = (float*)malloc(bytes);
        float* B = (float*)malloc(bytes);
        float* C_cpu = (float*)malloc(bytes);
        float* C_gpu = (float*)malloc(bytes);
        
        srand(42);
        fill_matrix(A, n);
        fill_matrix(B, n);
        
        double start = get_time();
        multiply_cpu(A, B, C_cpu, n);
        double cpu_time = get_time() - start;
        
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);
        
        cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);
        
        dim3 block_size(16, 16);
        dim3 grid_size((n + 15)/16, (n + 15)/16);
        
        cudaEvent_t start_gpu, stop_gpu;
        cudaEventCreate(&start_gpu);
        cudaEventCreate(&stop_gpu);
        
        cudaEventRecord(start_gpu);
        multiply_gpu_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, n);
        cudaEventRecord(stop_gpu);
        cudaEventSynchronize(stop_gpu);
        
        float gpu_time_ms;
        cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu);
        float gpu_time = gpu_time_ms / 1000.0;
        
        cudaMemcpy(C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
        
        bool correct = check_result(C_cpu, C_gpu, n);
        
        printf("%8d | %12.3f | %12.5f | %10.2fx %s\n", 
               n, cpu_time, gpu_time, cpu_time/gpu_time, correct ? "✓" : "✗");
        
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(A); free(B); free(C_cpu); free(C_gpu);
    }
    
    printf("========================================================================\n");
    return 0;
}
