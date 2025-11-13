#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel to mark multiples of a given prime
__global__ void markMultiples(bool *isPrime, unsigned int prime, unsigned int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int multiple = prime * (prime + idx);
    
    if (multiple <= N) {
        isPrime[multiple] = false;
    }
}

// CUDA kernel to sieve primes in parallel
__global__ void sievePrimes(bool *isPrime, unsigned int N, unsigned int limit) {
    unsigned int num = blockIdx.x * blockDim.x + threadIdx.x + 2;
    
    if (num <= limit && isPrime[num]) {
        // Mark all multiples of num starting from num*num
        for (unsigned int multiple = num * num; multiple <= N; multiple += num) {
            isPrime[multiple] = false;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1;
    }
    
    unsigned int N = atoi(argv[1]);
    
    if (N < 2) {
        fprintf(stderr, "N must be at least 2\n");
        return 1;
    }
    
    // Allocate host memory
    bool *h_isPrime = (bool *)malloc((N + 1) * sizeof(bool));
    
    // Initialize all numbers as prime
    for (unsigned int i = 0; i <= N; i++) {
        h_isPrime[i] = true;
    }
    h_isPrime[0] = false;
    h_isPrime[1] = false;
    
    // Allocate device memory
    bool *d_isPrime;
    cudaMalloc((void **)&d_isPrime, (N + 1) * sizeof(bool));
    
    // Copy data to device
    cudaMemcpy(d_isPrime, h_isPrime, (N + 1) * sizeof(bool), cudaMemcpyHostToDevice);
    
    // Calculate the limit for sieving
    unsigned int limit = (unsigned int)floor((N + 1.0) / 2.0);
    
    // Launch kernel to perform sieving
    int threadsPerBlock = 256;
    int numBlocks = (limit - 1 + threadsPerBlock) / threadsPerBlock;
    
    sievePrimes<<<numBlocks, threadsPerBlock>>>(d_isPrime, N, limit);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Copy results back to host
    cudaMemcpy(h_isPrime, d_isPrime, (N + 1) * sizeof(bool), cudaMemcpyDeviceToHost);
    
    // Create output filename
    char filename[256];
    sprintf(filename, "%u.txt", N);
    
    // Write primes to file
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return 1;
    }
    
    bool first = true;
    for (unsigned int i = 2; i <= N; i++) {
        if (h_isPrime[i]) {
            if (!first) {
                fprintf(fp, " ");
            }
            fprintf(fp, "%u", i);
            first = false;
        }
    }
    
    fclose(fp);
    
    // Cleanup
    free(h_isPrime);
    cudaFree(d_isPrime);
    
    return 0;
}

