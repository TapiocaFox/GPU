#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

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
    
    // Allocate memory for the sieve array
    bool *isPrime = (bool *)malloc((N + 1) * sizeof(bool));
    if (isPrime == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize all numbers as prime
    for (unsigned int i = 0; i <= N; i++) {
        isPrime[i] = true;
    }
    isPrime[0] = false;
    isPrime[1] = false;
    
    // Calculate the limit for sieving
    unsigned int limit = (unsigned int)floor((N + 1.0) / 2.0);
    
    // Sieve of Eratosthenes
    for (unsigned int p = 2; p <= limit; p++) {
        if (isPrime[p]) {
            // Mark all multiples of p starting from p*p
            for (unsigned int multiple = p * p; multiple <= N; multiple += p) {
                isPrime[multiple] = false;
            }
        }
    }
    
    // Create output filename
    char filename[256];
    sprintf(filename, "%u.txt", N);
    
    // Write primes to file
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        free(isPrime);
        return 1;
    }
    
    bool first = true;
    for (unsigned int i = 2; i <= N; i++) {
        if (isPrime[i]) {
            if (!first) {
                fprintf(fp, " ");
            }
            fprintf(fp, "%u", i);
            first = false;
        }
    }
    
    fclose(fp);
    
    // Cleanup
    free(isPrime);
    
    return 0;
}