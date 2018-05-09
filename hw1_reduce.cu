#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 1024
#define KERNEL_RUN_COUNT 100

__global__ void reduce_kernel(float *, const float *);

float reduce_cpu(float *, int);

int main(int argc, char **argv) {
    const int ARRAY_LENGTH = 1<<20;
    const int ARRAY_SIZE = ARRAY_LENGTH * sizeof(float);

    // declare array to reduce
    float h_in[ARRAY_LENGTH];

    // populate array with random numbers
    for(int i = 0; i < ARRAY_LENGTH; i++) {
        // generate random float in [-1.0f, 1.0f]
        h_in[i] = -1.0f + (float)rand()/((float)RAND_MAX/2.0f);
    }

    // declare GPU memory pointers
    float * d_in, * d_intermediate, * d_out;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_SIZE);
    cudaMalloc((void **) &d_intermediate, ARRAY_SIZE); 
    cudaMalloc((void **) &d_out, sizeof(float));

    // copy the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_SIZE, cudaMemcpyHostToDevice); 
        
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // launch the kernel
    cudaEventRecord(start, 0);
    int blocks = ARRAY_LENGTH / THREADS_PER_BLOCK;
    for (int i = 0; i < KERNEL_RUN_COUNT; i++) {
        //reduce elements in each block
        reduce_kernel<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float)>>>(d_intermediate, d_in);
        
        //reduce all blocks
        reduce_kernel<<<1, blocks, blocks * sizeof(float)>>>(d_out, d_intermediate);
    }

    // copy back the sum from GPU
    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);


    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float elapsedTimeGpu;
    cudaEventElapsedTime(&elapsedTimeGpu, start, stop);    
    elapsedTimeGpu /= (float)KERNEL_RUN_COUNT;      

    // free GPU memory
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);


    // calculate reduced sum on CPU
    cudaEventRecord(start, 0);

    float sum = 0.0f;
    sum = reduce_cpu(h_in, ARRAY_LENGTH);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTimeCpu;
    cudaEventElapsedTime(&elapsedTimeCpu, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("GPU:\n\tProcessing time: %f sec\n\tSum: %f\n\n", elapsedTimeGpu, h_out);
    printf("CPU:\n\tProcessing time: %f sec\n\tSum: %f\n\n", elapsedTimeCpu, sum);
    printf("GPU/CPU speed ratio: %f\n", elapsedTimeCpu/elapsedTimeGpu);
        
    return 0;
}

__global__ void reduce_kernel(float * d_out, const float * d_in) {
    extern __shared__ float sdata[];

    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // copy one value per thread to shared memory
    sdata[tid] = d_in[id];
    __syncthreads();            

    // sum elements tree-wise in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();       
    }

    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

float reduce_cpu(float * a, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += a[i];
    }
    return sum;
}
