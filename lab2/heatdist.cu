/* 
 * This file contains the code for doing the heat distribution problem. 
 * You do not need to modify anything except starting  gpu_heat_dist() at the bottom
 * of this file.
 * In gpu_heat_dist() you can organize your data structure and the call to your
 * kernel(s), memory allocation, data movement, etc. 
 * 
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 

/* To index element (i,j) of a 2D square array of dimension NxN stored as 1D 
   index(i, j, N) means access element at row i, column j, and N is the dimension which is NxN */
#define index(i, j, N)  ((i)*(N)) + (j)

/*****************************************************************/

// Function declarations: Feel free to add any functions you want.
void  seq_heat_dist(float *, unsigned int, unsigned int);
void  gpu_heat_dist(float *, unsigned int, unsigned int);


/*****************************************************************/
/**** Do NOT CHANGE ANYTHING in main() function ******/

int main(int argc, char * argv[])
{
  unsigned int N; /* Dimention of NxN matrix */
  int type_of_device = 0; // CPU or GPU
  int iterations = 0;
  int i;
  
  /* The 2D array of points will be treated as 1D array of NxN elements */
  float * playground; 
  
  // to measure time taken by a specific part of the code 
  double time_taken;
  clock_t start, end;
  
  if(argc != 4)
  {
    fprintf(stderr, "usage: heatdist num  iterations  who\n");
    fprintf(stderr, "num = dimension of the square matrix (50 and up)\n");
    fprintf(stderr, "iterations = number of iterations till stopping (1 and up)\n");
    fprintf(stderr, "who = 0: sequential code on CPU, 1: GPU version\n");
    exit(1);
  }
  
  type_of_device = atoi(argv[3]);
  N = (unsigned int) atoi(argv[1]);
  iterations = (unsigned int) atoi(argv[2]);
 
  
  /* Dynamically allocate NxN array of floats */
  playground = (float *)calloc(N*N, sizeof(float));
  if( !playground )
  {
   fprintf(stderr, " Cannot allocate the %u x %u array\n", N, N);
   exit(1);
  }
  
  /* Initialize it: calloc already initalized everything to 0 */
  // Edge elements  initialization
  for(i = 0; i < N; i++)
    playground[index(0,i,N)] = 120;
  for(i = 0; i < N; i++)
    playground[index(N-1,i,N)] = 130;
  for(i = 1; i < N-1; i++)
    playground[index(i,0,N)] = 70;
  for(i = 1; i < N-1; i++)
    playground[index(i,N-1,N)] = 70;
  

  switch(type_of_device)
  {
	case 0: printf("CPU sequential version:\n");
			start = clock();
			seq_heat_dist(playground, N, iterations);
			end = clock();
			break;
		
	case 1: printf("GPU version:\n");
			start = clock();
			gpu_heat_dist(playground, N, iterations); 
			cudaDeviceSynchronize();
			end = clock();  
			break;
			
	default: printf("Invalid device type\n");
			 exit(1);
  }
  
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  
  printf("Time taken = %lf\n", time_taken);
  
  free(playground);
  
  return 0;

}


/*****************  The CPU sequential version (DO NOT CHANGE THAT) **************/
void  seq_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
  // Loop indices
  int i, j, k;
  int upper = N-1;
  
  // number of bytes to be copied between array temp and array playground
  unsigned int num_bytes = 0;
  
  float * temp; 
  /* Dynamically allocate another array for temp values */
  /* Dynamically allocate NxN array of floats */
  temp = (float *)calloc(N*N, sizeof(float));
  if( !temp )
  {
   fprintf(stderr, " Cannot allocate temp %u x %u array\n", N, N);
   exit(1);
  }
  
  num_bytes = N*N*sizeof(float);
  
  /* Copy initial array in temp */
  memcpy((void *)temp, (void *) playground, num_bytes);
  
  for( k = 0; k < iterations; k++)
  {
    /* Calculate new values and store them in temp */
    for(i = 1; i < upper; i++)
      for(j = 1; j < upper; j++)
	temp[index(i,j,N)] = (playground[index(i-1,j,N)] + 
	                      playground[index(i+1,j,N)] + 
			      playground[index(i,j-1,N)] + 
			      playground[index(i,j+1,N)])/4.0;
  
			      
   			      
    /* Move new values into old values */ 
    memcpy((void *)playground, (void *) temp, num_bytes);
  }
  
}

/***************** The GPU version: Write your code here *********************/
/* This function can call one or more kernels if you want ********************/
// --- Kernel (paste in your .cu file, outside of gpu_heat_dist) ---
__global__ void heat_kernel(const float *in, float *out, unsigned int N) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; // column
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (i > 0 && i < (int)N - 1 && j > 0 && j < (int)N - 1) {
        unsigned int idx = i * N + j;
        out[idx] = 0.25f * (in[idx - N]     /* i-1, j */
                          + in[idx + N]     /* i+1, j */
                          + in[idx - 1]     /* i, j-1 */
                          + in[idx + 1]);   /* i, j+1 */
    }
}

// --- gpu_heat_dist implementation (replace the empty function in your file) ---
void gpu_heat_dist(float *playground, unsigned int N, unsigned int iterations)
{
    if (N < 3 || iterations == 0) return; // nothing to do

    size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    float *d_in = nullptr;
    float *d_out = nullptr;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_in, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_in failed: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc((void**)&d_out, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_out failed: %s\n", cudaGetErrorString(err)); cudaFree(d_in); return; }

    // Copy initial playground 2 device
    err = cudaMemcpy(d_in, playground, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err)); cudaFree(d_in); cudaFree(d_out); return; }

    
    // Kernel launch configuration
    dim3 threads(16, 16);
    dim3 blocks( (N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y );

    // Iteratively run kernel and swap buffers 
    for (unsigned int it = 0; it < iterations; ++it) {

        heat_kernel<<<blocks, threads>>>(d_in, d_out, N); // Actually run the heat kernel.

        // Check kernel launch
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed at iter %u: %s\n", it, cudaGetErrorString(err));
            cudaFree(d_in); cudaFree(d_out);
            return;
        }

        // Ensure kernel finished before swapping 
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize failed at iter %u: %s\n", it, cudaGetErrorString(err));
            cudaFree(d_in); cudaFree(d_out);
            return;
        }


        // Swap pointers
        float *tmp = d_in;
        d_in = d_out;
        d_out = tmp;
    }

    err = cudaMemcpy(playground, d_in, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_in);
    cudaFree(d_out);
}


