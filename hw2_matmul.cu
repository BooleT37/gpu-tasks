#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define EPSILON 0.0001f

bool equals(float a, float b) {
  return fabs(a - b) < EPSILON;
}


int main (int argc, char **argv) {
  const int N = 2;

  float * h_a; 
  float * h_b; 
  float * h_c; 
  float * h_c_cpu;

  h_a = (float *) malloc (N * N * sizeof(float)); 
  h_b = (float *) malloc (N * N * sizeof(float)); 
  h_c = (float *) malloc (N * N * sizeof(float)); 
  h_c_cpu = (float *) malloc (N * N * sizeof(float)); 

  // filling matrices with random values
  for(int i = 0; i < N * N; i++) {
      h_a[i] = -1.0f + (float)rand()/((float)RAND_MAX/2.0f);
      h_b[i] = -1.0f + (float)rand()/((float)RAND_MAX/2.0f);
      h_c[i] = 0.0f;
      h_c_cpu[i] = 0.0f;
  }
  
  // device pointers
  float * d_a; 
  float * d_b;
  float * d_c;
  float * d_c_t;
  cudaMalloc((void **)& d_a, N * N * sizeof(float)); 
  cudaMalloc((void **)& d_b, N * N * sizeof(float)); 
  cudaMalloc((void **)& d_c, N * N * sizeof(float)); 
  cudaMalloc((void **)& d_c_t, N * N * sizeof(float)); 

  cublasHandle_t handle;
  cublasCreate(&handle);
 
  // host pointers
  cublasSetMatrix (N, N, sizeof(float), h_a, N, d_a, N); 
  cublasSetMatrix (N, N, sizeof(float), h_b, N, d_b, N); 
  cublasSetMatrix (N, N, sizeof(float), h_c, N, d_c, N); 
  cublasSetMatrix (N, N, sizeof(float), h_c, N, d_c_t, N); 

  const float alf = 1;
  const float bet = 0;
  const float *alpha = &alf;
  const float *beta = &bet;


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // multiplicate matrices
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, alpha, d_a, N, d_b, N, beta, d_c_t, N);
  // transpose result matrix
  cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, alpha, d_c_t, N, beta, d_c_t, N, d_c, N);
  cublasGetMatrix (N, N, sizeof(float) ,d_c ,N, h_c, N); 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  
  float elapsedTimeGpu;
  cudaEventElapsedTime(&elapsedTimeGpu, start, stop);    
  elapsedTimeGpu /= 100.0f;      


  cudaFree (d_a); 
  cudaFree (d_b); 
  cudaFree (d_c); 
  cublasDestroy (handle);
  float sum;


  cudaEventRecord(start, 0);

  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      sum = 0;
      for (int k=0; k<N; k++) {
        sum += h_a[N*i + k] * h_b[N*k + j];
      }
      h_c_cpu[N*i + j] = sum;
    }
  }
    
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
    
  float elapsedTimeCpu;
  cudaEventElapsedTime(&elapsedTimeCpu, start, stop);
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);


  bool result = true;

  for (int i = 0; i < N * N; i++) {
     if (equals(h_c[i], h_c_cpu[i]) == false) {
       result = false;
     }
  }

  printf("GPU:\n\tProcessing time: %f sec\n\tMatrix slice:\n\t\t[%f %f ... ]\n\t\t[%f %f ... ]\n\n", elapsedTimeGpu, h_c[0], h_c[1], h_c[N], h_c[N + 1]);
  printf("CPU:\n\tProcessing time: %f sec\n\tMatrix slice:\n\t\t[%f %f ... ]\n\t\t[%f %f ... ]\n\n", elapsedTimeCpu, h_c_cpu[0], h_c_cpu[1], h_c_cpu[N], h_c_cpu[N + 1]);
  
  if (result) {
     printf("Matrices are equal\n");
  } else {
     printf("Matrices are not equal\n\n");
     printf("[%f %f]   [%f %f]\n[%f %f] x [%f %f]\n\n", h_a[0], h_a[1], h_b[0], h_b[1], h_a[N], h_a[N + 1], h_b[N], h_b[N + 1]);
     printf("GPU:\n[%f %f]\n[%f %f]\n\n", h_c[0], h_c[1], h_c[N], h_c[N + 1]);
     printf("CPU:\n[%f %f]\n[%f %f]\n\n", h_c_cpu[0], h_c_cpu[1], h_c_cpu[N], h_c_cpu[N + 1]);
  }

  free(h_a); 
  free(h_b); 
  free(h_c); 
  free(h_c_cpu);

  return 0;
}
