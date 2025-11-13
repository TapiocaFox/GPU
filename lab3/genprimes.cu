#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

#define BLOCK_SIZE 256
#define SEGMENT_SIZE (1 << 20)  // 1M numbers per segment

// GPU kernel: Mark composites in a segment using all given primes
__global__ void sieveSegment(unsigned char *segment, 
                              unsigned int *primes,
                              int numPrimes,
                              unsigned long long segmentStart,
                              unsigned long long segmentEnd) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= numPrimes) return;
    
    unsigned long long prime = primes[tid];
    
    // Find first multiple of prime in this segment
    unsigned long long start = ((segmentStart + prime - 1) / prime) * prime;
    if (start < prime * prime) start = prime * prime;
    
    // Mark all multiples in this segment
    for (unsigned long long num = start; num <= segmentEnd; num += prime) {
        if (num >= segmentStart) {
            segment[num - segmentStart] = 0;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1;
    }
    
    unsigned long long N = atoll(argv[1]);
    
    if (N < 2) {
        fprintf(stderr, "N must be at least 2\n");
        return 1;
    }
    
    unsigned int limit = (unsigned int)sqrt((double)N);
    
    // Step 1: CPU sieve for small primes up to sqrt(N)
    unsigned char *smallPrimes = (unsigned char *)malloc(limit + 1);
    memset(smallPrimes, 1, limit + 1);
    smallPrimes[0] = smallPrimes[1] = 0;
    
    for (unsigned int p = 2; p * p <= limit; p++) {
        if (smallPrimes[p]) {
            for (unsigned int m = p * p; m <= limit; m += p) {
                smallPrimes[m] = 0;
            }
        }
    }
    
    // Collect base primes
    unsigned int *h_primes = (unsigned int *)malloc((limit + 1) * sizeof(unsigned int));
    int numPrimes = 0;
    for (unsigned int p = 2; p <= limit; p++) {
        if (smallPrimes[p]) {
            h_primes[numPrimes++] = p;
        }
    }
    
    // Allocate result array
    unsigned char *h_isPrime = (unsigned char *)malloc((N + 1) * sizeof(unsigned char));
    memset(h_isPrime, 1, N + 1);
    h_isPrime[0] = h_isPrime[1] = 0;
    
    // Copy small primes to result
    for (unsigned int i = 2; i <= limit; i++) {
        h_isPrime[i] = smallPrimes[i];
    }
    free(smallPrimes);
    
    // Step 2: GPU segmented sieve for large range
    if (N > limit) {
        unsigned char *d_segment;
        unsigned int *d_primes;
        
        unsigned long long segmentSize = SEGMENT_SIZE;
        if (segmentSize > N - limit) segmentSize = N - limit + 1;
        
        cudaMalloc(&d_segment, segmentSize * sizeof(unsigned char));
        cudaMalloc(&d_primes, numPrimes * sizeof(unsigned int));
        
        cudaMemcpy(d_primes, h_primes, numPrimes * sizeof(unsigned int), cudaMemcpyHostToDevice);
        
        // Process in segments
        for (unsigned long long segStart = limit + 1; segStart <= N; segStart += segmentSize) {
            unsigned long long segEnd = segStart + segmentSize - 1;
            if (segEnd > N) segEnd = N;
            
            unsigned long long curSegSize = segEnd - segStart + 1;
            
            // Initialize segment to all prime
            cudaMemset(d_segment, 1, curSegSize);
            
            // Launch kernel
            int numBlocks = (numPrimes + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sieveSegment<<<numBlocks, BLOCK_SIZE>>>(d_segment, d_primes, numPrimes, segStart, segEnd);
            
            cudaDeviceSynchronize();
            
            // Copy segment back
            cudaMemcpy(h_isPrime + segStart, d_segment, curSegSize, cudaMemcpyDeviceToHost);
        }
        
        cudaFree(d_segment);
        cudaFree(d_primes);
    }
    
    free(h_primes);
    
    // Write output
    char filename[256];
    sprintf(filename, "%llu.txt", N);
    
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error opening file\n");
        free(h_isPrime);
        return 1;
    }
    
    int first = 1;
    for (unsigned long long i = 2; i <= N; i++) {
        if (h_isPrime[i]) {
            if (!first) fprintf(fp, " ");
            fprintf(fp, "%llu", i);
            first = 0;
        }
    }
    
    fclose(fp);
    free(h_isPrime);
    
    return 0;
}
