#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#define TIME_CHECK_VERBOSE 1

template <typename T>
class GpuArray {
private:
	T *gpuPtr;
	unsigned int arraySize;
public:
	GpuArray():gpuPtr(NULL) {}

	GpuArray(unsigned int size) {
		gpuPtr = NULL;
		size = sizeof(T) * size;
		cutilSafeCall( cudaMalloc(&gpuPtr, size) );
		arraySize = size;
	}

	~GpuArray() {
		if(gpuPtr) {
			cutilSafeCall( cudaFree(gpuPtr) );
			gpuPtr = NULL;
			arraySize = 0;
		}
	}

	T *getPtr() {
		return gpuPtr;
	}

	T *operator()() {
		return gpuPtr;
	}

	void allocate(unsigned int size) {
		if(size * sizeof(T) == arraySize) {
			return;
		}

		if(gpuPtr) {
			cudaFree(gpuPtr);
		}
		size = sizeof(T) * size;
		cutilSafeCall( cudaMalloc(&gpuPtr, size) );
		arraySize = size;
	}

	void writeDataToDevice(T *dataSrc) {
		cutilSafeCall( cudaMemcpy(gpuPtr, dataSrc, arraySize, cudaMemcpyHostToDevice) );
	}

	void readDataToHost(T *dataDes) {
		cutilSafeCall( cudaMemcpy(dataDes, gpuPtr, arraySize, cudaMemcpyDeviceToHost) );
	}

	unsigned int getSize() {
		return arraySize;
	}
};

__global__ void scaleData(float *data, float scale, float offset, int w, int h, int c);
__global__ void imageTranspose_1(float *odata, float *idata, int w, int h);
__global__ void imageTranspose_1_mlayers(float *odata, float *idata, int w, int h, int layer);
__global__ void imageTranspose_3(float *odata, float *idata, int w, int h);
__global__ void imageTranspose_3(unsigned char *odata, unsigned char *idata, int w, int h);
__global__ void normailize(float *pWeight, float *pData, int w, int h, int c);

template <typename TYPE>
void ctmf_helper_gpu(TYPE* src, TYPE* dst,int width, int height, int channels, int r);

void ctmf_gpu(unsigned char * src, unsigned char * dst,int width, int height,int src_step_row, int dst_step_row,int r, int c, unsigned long memsize);
void ctmf_gpu(float* src, float* dst,int width, int height,int src_step_row, int dst_step_row,int r, int c, unsigned long memsize);


void dumpGPUInfo();

float time_check(const char *posStr);

void add2(float t);
void add1(float t);
