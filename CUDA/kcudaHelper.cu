#include <cutil_inline.h>
#include <vector>
#include <ctime>
#include "kcudaHelper.h"

inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
       int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
       int Cores;
    } sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = 
	{ { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
	  { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
	  { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
	  { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
	  { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
	  { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
	  { 0x30, 192}, // Fermi Generation (SM 3.0) GK10x class
	  {   -1, -1 }
	};

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
       if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
          return nGpuArchCoresPerSM[index].Cores;
       }	
       index++;
    }
    printf("MapSMtoCores undefined SM %d.%d is undefined (please update to the latest SDK)!\n", major, minor);
    return -1;
}

static int gpuGetMaxGflopsDeviceId()
{
	int current_device   = 0, sm_per_multiproc = 0;
	int max_compute_perf = 0, max_perf_device  = 0;
	int device_count     = 0, best_SM_arch     = 0;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount( &device_count );
	// Find the best major SM Architecture GPU device
	while ( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major > 0 && deviceProp.major < 9999) {
			best_SM_arch = MAX(best_SM_arch, deviceProp.major);
		}
		current_device++;
	}

	// Find the best CUDA capable GPU device
	current_device = 0;
	while( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
			sm_per_multiproc = 1;
		} else {
			sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
		}

		int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
		if( compute_perf  > max_compute_perf ) {
			// If we find GPU with SM major > 2, search only these
			if ( best_SM_arch > 2 ) {
				// If our device==dest_SM_arch, choose this, or else pass
				if (deviceProp.major == best_SM_arch) {	
					max_compute_perf  = compute_perf;
					max_perf_device   = current_device;
				}
			} else {
				max_compute_perf  = compute_perf;
				max_perf_device   = current_device;
			}
		}
		++current_device;
	}
	return max_perf_device;
}

void dumpGPUInfo() {
	int devID = gpuGetMaxGflopsDeviceId();

	cudaDeviceProp deviceProp;

	// get number of SMs on this GPU
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&deviceProp, devID);
	printf("> Device %d: \"%s\"\n", devID, deviceProp.name);
	printf("> SM Capability %d.%d detected\n\n", deviceProp.major, deviceProp.minor);

	float *pTmp;
	cutilSafeCall( cudaMalloc(&pTmp, sizeof(float)) );
	cutilSafeCall( cudaFree(pTmp) );
}

__global__ void scaleData(float *data, float scale, float offset, int w, int h, int c) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= w || y >= h) return;

	int idx = y * w + x;

	if(c == 3) {
		data[idx * 3] = data[idx * 3] * scale + offset;
		data[idx * 3 + 1] = data[idx * 3 + 1] * scale + offset;
		data[idx * 3 + 2] = data[idx * 3 + 2] * scale + offset;

	} else {
		for(int i = 0; i < c;i++) {
			data[idx*c+i] = data[idx*c+i] * scale + offset;
		}
	}
}

float time_check(const char *posStr) {
	static std::vector<clock_t> check_points;

	if(!posStr) {
		check_points.clear();
	}
	

	clock_t t = clock();
	float time_cost;
	if(check_points.size()) {
		time_cost=(float(t-check_points[check_points.size() - 1]))/(float)CLOCKS_PER_SEC*(float)1000;
		if(TIME_CHECK_VERBOSE) {
			printf("%s: %d\n",posStr,int(time_cost));
		}
	}

	check_points.push_back(t);
	
	return time_cost;
}

#define TILE_DIM  16
__global__ void imageTranspose_1(float *odata, float *idata, int w, int h)
{
	__shared__ float tile[TILE_DIM][TILE_DIM+1];

	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;  
	int index_in = xIndex + (yIndex)*w;

	if(xIndex < w && yIndex < h) {
		tile[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*h;

	if(xIndex < h && yIndex < w) {
		odata[index_out] = tile[threadIdx.x][threadIdx.y];
	}
}

__global__ void imageTranspose_1_mlayers(float *odata, float *idata, int w, int h, int layer)
{
	__shared__ float tile[TILE_DIM][TILE_DIM+1];

	for(int i = 0;i < layer;i++) {
		int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
		int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;  
		int index_in = xIndex + (yIndex)*w;

		float *od = odata + w * h * i;
		float *id = idata + w * h * i;

		if(xIndex < w && yIndex < h) {
			tile[threadIdx.y][threadIdx.x] = id[index_in];
		}

		__syncthreads();

		xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
		yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
		int index_out = xIndex + (yIndex)*h;

		if(xIndex < h && yIndex < w) {
			od[index_out] = tile[threadIdx.x][threadIdx.y];
		}

		__syncthreads();
	}
}

__global__ void imageTranspose_3(float *odata, float *idata, int w, int h) {
	__shared__ float tile[3][TILE_DIM][TILE_DIM+1];

	int xIndex;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;  
	int index_in = (yIndex * w) * 3;
	
	for(int i = 0;i < TILE_DIM * 3;i += TILE_DIM) {
		xIndex = (blockIdx.x * TILE_DIM)*3 + i + threadIdx.x;
		if(xIndex < 3 * w && yIndex < h) {
			tile[(i + threadIdx.x) % 3][threadIdx.y][(i + threadIdx.x)/3] = idata[index_in + xIndex];
		}
	}
	
	__syncthreads();

	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	int index_out = (yIndex * h) * 3;
	for(int i = 0;i < TILE_DIM * 3;i += TILE_DIM) {
		xIndex = (blockIdx.y * TILE_DIM)*3 + i + threadIdx.x;
		if(xIndex < 3 * h && yIndex < w) {
			odata[index_out + xIndex] = tile[(i + threadIdx.x) % 3][(i + threadIdx.x)/3][threadIdx.y];
		}
	}
}

__global__ void imageTranspose_3(unsigned char *odata, unsigned char *idata, int w, int h) {
	__shared__ unsigned char tile[3][TILE_DIM][TILE_DIM+1];

	int xIndex;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;  
	int index_in = (yIndex * w) * 3;
	
	for(int i = 0;i < TILE_DIM * 3;i += TILE_DIM) {
		xIndex = (blockIdx.x * TILE_DIM)*3 + i + threadIdx.x;
		if(xIndex < 3 * w && yIndex < h) {
			tile[(i + threadIdx.x) % 3][threadIdx.y][(i + threadIdx.x)/3] = idata[index_in + xIndex];
		}
	}
	
	__syncthreads();

	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	int index_out = (yIndex * h) * 3;
	for(int i = 0;i < TILE_DIM * 3;i += TILE_DIM) {
		xIndex = (blockIdx.y * TILE_DIM)*3 + i + threadIdx.x;
		if(xIndex < 3 * h && yIndex < w) {
			odata[index_out + xIndex] = tile[(i + threadIdx.x) % 3][(i + threadIdx.x)/3][threadIdx.y];
		}
	}
}

__global__ void normailize(float *pWeight, float *pData, int w, int h, int c) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= w || y >= h) return;

	int idx = y * w + x;
	float scale = 1.0f / pWeight[idx];
	
	if(c == 3) {
		pData[idx * c] *= scale;
		pData[idx * c+1] *= scale;
		pData[idx * c+2] *= scale;
	} else {
		for(int i = 0;i < c;i++) {
			pData[i*w*h+idx] *= scale;
		}
	}
}

#define BLOCK 8
template <typename TYPE>
__global__ void medianFilter3(TYPE *input, TYPE *output, int w, int h, int c) {
	__shared__ TYPE window[BLOCK*BLOCK][9];

	int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
 
    int tid=threadIdx.y*blockDim.y+threadIdx.x;

	if(x < w && y < h) {
		for(int i = 0;i < c;i++) {
			window[tid][0] = (y==0||x==0)? 0 : input[((y-1)*w+x-1)*c+i];
			window[tid][1] = (y==0) ? 0 : input[((y-1)*w+x)*c+i];
			window[tid][2] = (y==0||x==w-1) ? 0 : input[((y-1)*w+x+1)*c+i];
			window[tid][3] = (x==0) ? 0 : input[(y*w+x-1)*c+i];
			window[tid][4] = input[(y*w+x)*c+i];
			window[tid][5] = (x==w-1) ? 0 : input[(y*w+x+1)*c+i];
			window[tid][6] = (y==h-1||x==0) ? 0 : input[((y+1)*w+x-1)*c+i];
			window[tid][7] = (y==h-1) ? 0 : input[((y+1)*w+x)*c+i];
			window[tid][8] = (y==h-1||x==w-1) ? 0 : input[((y+1)*w+x+1)*c+i];

			int min=0;TYPE temp;
			if (window[tid][1] < window[tid][min]) min=1;
			if (window[tid][2] < window[tid][min]) min=2;
			if (window[tid][3] < window[tid][min]) min=3;
			if (window[tid][4] < window[tid][min]) min=4;
			if (window[tid][5] < window[tid][min]) min=5;
			if (window[tid][6] < window[tid][min]) min=6;
			if (window[tid][7] < window[tid][min]) min=7;
			if (window[tid][8] < window[tid][min]) min=8;
			temp=window[tid][0];window[tid][0]=window[tid][min];window[tid][min]=temp;
			min=1;
			if (window[tid][2] < window[tid][min]) min=2;
			if (window[tid][3] < window[tid][min]) min=3;
			if (window[tid][4] < window[tid][min]) min=4;
			if (window[tid][5] < window[tid][min]) min=5;
			if (window[tid][6] < window[tid][min]) min=6;
			if (window[tid][7] < window[tid][min]) min=7;
			if (window[tid][8] < window[tid][min]) min=8;
			temp=window[tid][1];window[tid][1]=window[tid][min];window[tid][min]=temp;
			min=2;
			if (window[tid][3] < window[tid][min]) min=3;
			if (window[tid][4] < window[tid][min]) min=4;
			if (window[tid][5] < window[tid][min]) min=5;
			if (window[tid][6] < window[tid][min]) min=6;
			if (window[tid][7] < window[tid][min]) min=7;
			if (window[tid][8] < window[tid][min]) min=8;
			temp=window[tid][2];window[tid][2]=window[tid][min];window[tid][min]=temp;
			min=3;
			if (window[tid][4] < window[tid][min]) min=4;
			if (window[tid][5] < window[tid][min]) min=5;
			if (window[tid][6] < window[tid][min]) min=6;
			if (window[tid][7] < window[tid][min]) min=7;
			if (window[tid][8] < window[tid][min]) min=8;
			temp=window[tid][3];window[tid][3]=window[tid][min];window[tid][min]=temp;
			min=4;
			if (window[tid][5] < window[tid][min]) min=5;
			if (window[tid][6] < window[tid][min]) min=6;
			if (window[tid][7] < window[tid][min]) min=7;
			if (window[tid][8] < window[tid][min]) min=8;
			temp=window[tid][4];window[tid][4]=window[tid][min];window[tid][min]=temp;

			output[(y*w+x)*c+i] = window[tid][4];
		}
	}
}

template <typename TYPE>
__global__ void medianFilter5(TYPE *input, TYPE *output, int w, int h, int c) {
	__shared__ TYPE window[BLOCK*BLOCK][25];

	int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
 
    int tid=threadIdx.y*blockDim.y+threadIdx.x;

	if(x < w && y < h) {
		for(int i = 0;i < c;i++) {
			if(x >= 2 && y >= 2 && x < w - 2 && y < h - 2) {
				window[tid][0] = input[((y+(-2))*w+x+(-2))*c+i];
				window[tid][1] = input[((y+(-1))*w+x+(-2))*c+i];
				window[tid][2] = input[((y+(0))*w+x+(-2))*c+i];
				window[tid][3] = input[((y+(1))*w+x+(-2))*c+i];
				window[tid][4] = input[((y+(2))*w+x+(-2))*c+i];
				window[tid][5] = input[((y+(-2))*w+x+(-1))*c+i];
				window[tid][6] = input[((y+(-1))*w+x+(-1))*c+i];
				window[tid][7] = input[((y+(0))*w+x+(-1))*c+i];
				window[tid][8] = input[((y+(1))*w+x+(-1))*c+i];
				window[tid][9] = input[((y+(2))*w+x+(-1))*c+i];
				window[tid][10] = input[((y+(-2))*w+x+(0))*c+i];
				window[tid][11] = input[((y+(-1))*w+x+(0))*c+i];
				window[tid][12] = input[((y+(0))*w+x+(0))*c+i];
				window[tid][13] = input[((y+(1))*w+x+(0))*c+i];
				window[tid][14] = input[((y+(2))*w+x+(0))*c+i];
				window[tid][15] = input[((y+(-2))*w+x+(1))*c+i];
				window[tid][16] = input[((y+(-1))*w+x+(1))*c+i];
				window[tid][17] = input[((y+(0))*w+x+(1))*c+i];
				window[tid][18] = input[((y+(1))*w+x+(1))*c+i];
				window[tid][19] = input[((y+(2))*w+x+(1))*c+i];
				window[tid][20] = input[((y+(-2))*w+x+(2))*c+i];
				window[tid][21] = input[((y+(-1))*w+x+(2))*c+i];
				window[tid][22] = input[((y+(0))*w+x+(2))*c+i];
				window[tid][23] = input[((y+(1))*w+x+(2))*c+i];
				window[tid][24] = input[((y+(2))*w+x+(2))*c+i];
			} else {
				for(int i = 0;i < 25;i++) {
					int ox = i % 5 - 2;
					int oy = i / 5 - 2;
					ox += x;
					oy += y;
					if(ox < 0 || oy < 0 || ox >= w || oy >= h) {
						window[tid][i] = 0;
					} else {
						window[tid][i] = input[(oy*w+ox)*c+i];
					}
				}

			}


			int min; TYPE temp;
			min=0;
			if (window[tid][1] < window[tid][min]) min=1;
			if (window[tid][2] < window[tid][min]) min=2;
			if (window[tid][3] < window[tid][min]) min=3;
			if (window[tid][4] < window[tid][min]) min=4;
			if (window[tid][5] < window[tid][min]) min=5;
			if (window[tid][6] < window[tid][min]) min=6;
			if (window[tid][7] < window[tid][min]) min=7;
			if (window[tid][8] < window[tid][min]) min=8;
			if (window[tid][9] < window[tid][min]) min=9;
			if (window[tid][10] < window[tid][min]) min=10;
			if (window[tid][11] < window[tid][min]) min=11;
			if (window[tid][12] < window[tid][min]) min=12;
			if (window[tid][13] < window[tid][min]) min=13;
			if (window[tid][14] < window[tid][min]) min=14;
			if (window[tid][15] < window[tid][min]) min=15;
			if (window[tid][16] < window[tid][min]) min=16;
			if (window[tid][17] < window[tid][min]) min=17;
			if (window[tid][18] < window[tid][min]) min=18;
			if (window[tid][19] < window[tid][min]) min=19;
			if (window[tid][20] < window[tid][min]) min=20;
			if (window[tid][21] < window[tid][min]) min=21;
			if (window[tid][22] < window[tid][min]) min=22;
			if (window[tid][23] < window[tid][min]) min=23;
			if (window[tid][24] < window[tid][min]) min=24;
			temp=window[tid][0];window[tid][0]=window[tid][min];window[tid][min]=temp;
			min=1;
			if (window[tid][2] < window[tid][min]) min=2;
			if (window[tid][3] < window[tid][min]) min=3;
			if (window[tid][4] < window[tid][min]) min=4;
			if (window[tid][5] < window[tid][min]) min=5;
			if (window[tid][6] < window[tid][min]) min=6;
			if (window[tid][7] < window[tid][min]) min=7;
			if (window[tid][8] < window[tid][min]) min=8;
			if (window[tid][9] < window[tid][min]) min=9;
			if (window[tid][10] < window[tid][min]) min=10;
			if (window[tid][11] < window[tid][min]) min=11;
			if (window[tid][12] < window[tid][min]) min=12;
			if (window[tid][13] < window[tid][min]) min=13;
			if (window[tid][14] < window[tid][min]) min=14;
			if (window[tid][15] < window[tid][min]) min=15;
			if (window[tid][16] < window[tid][min]) min=16;
			if (window[tid][17] < window[tid][min]) min=17;
			if (window[tid][18] < window[tid][min]) min=18;
			if (window[tid][19] < window[tid][min]) min=19;
			if (window[tid][20] < window[tid][min]) min=20;
			if (window[tid][21] < window[tid][min]) min=21;
			if (window[tid][22] < window[tid][min]) min=22;
			if (window[tid][23] < window[tid][min]) min=23;
			if (window[tid][24] < window[tid][min]) min=24;
			temp=window[tid][1];window[tid][1]=window[tid][min];window[tid][min]=temp;
			min=2;
			if (window[tid][3] < window[tid][min]) min=3;
			if (window[tid][4] < window[tid][min]) min=4;
			if (window[tid][5] < window[tid][min]) min=5;
			if (window[tid][6] < window[tid][min]) min=6;
			if (window[tid][7] < window[tid][min]) min=7;
			if (window[tid][8] < window[tid][min]) min=8;
			if (window[tid][9] < window[tid][min]) min=9;
			if (window[tid][10] < window[tid][min]) min=10;
			if (window[tid][11] < window[tid][min]) min=11;
			if (window[tid][12] < window[tid][min]) min=12;
			if (window[tid][13] < window[tid][min]) min=13;
			if (window[tid][14] < window[tid][min]) min=14;
			if (window[tid][15] < window[tid][min]) min=15;
			if (window[tid][16] < window[tid][min]) min=16;
			if (window[tid][17] < window[tid][min]) min=17;
			if (window[tid][18] < window[tid][min]) min=18;
			if (window[tid][19] < window[tid][min]) min=19;
			if (window[tid][20] < window[tid][min]) min=20;
			if (window[tid][21] < window[tid][min]) min=21;
			if (window[tid][22] < window[tid][min]) min=22;
			if (window[tid][23] < window[tid][min]) min=23;
			if (window[tid][24] < window[tid][min]) min=24;
			temp=window[tid][2];window[tid][2]=window[tid][min];window[tid][min]=temp;
			min=3;
			if (window[tid][4] < window[tid][min]) min=4;
			if (window[tid][5] < window[tid][min]) min=5;
			if (window[tid][6] < window[tid][min]) min=6;
			if (window[tid][7] < window[tid][min]) min=7;
			if (window[tid][8] < window[tid][min]) min=8;
			if (window[tid][9] < window[tid][min]) min=9;
			if (window[tid][10] < window[tid][min]) min=10;
			if (window[tid][11] < window[tid][min]) min=11;
			if (window[tid][12] < window[tid][min]) min=12;
			if (window[tid][13] < window[tid][min]) min=13;
			if (window[tid][14] < window[tid][min]) min=14;
			if (window[tid][15] < window[tid][min]) min=15;
			if (window[tid][16] < window[tid][min]) min=16;
			if (window[tid][17] < window[tid][min]) min=17;
			if (window[tid][18] < window[tid][min]) min=18;
			if (window[tid][19] < window[tid][min]) min=19;
			if (window[tid][20] < window[tid][min]) min=20;
			if (window[tid][21] < window[tid][min]) min=21;
			if (window[tid][22] < window[tid][min]) min=22;
			if (window[tid][23] < window[tid][min]) min=23;
			if (window[tid][24] < window[tid][min]) min=24;
			temp=window[tid][3];window[tid][3]=window[tid][min];window[tid][min]=temp;
			min=4;
			if (window[tid][5] < window[tid][min]) min=5;
			if (window[tid][6] < window[tid][min]) min=6;
			if (window[tid][7] < window[tid][min]) min=7;
			if (window[tid][8] < window[tid][min]) min=8;
			if (window[tid][9] < window[tid][min]) min=9;
			if (window[tid][10] < window[tid][min]) min=10;
			if (window[tid][11] < window[tid][min]) min=11;
			if (window[tid][12] < window[tid][min]) min=12;
			if (window[tid][13] < window[tid][min]) min=13;
			if (window[tid][14] < window[tid][min]) min=14;
			if (window[tid][15] < window[tid][min]) min=15;
			if (window[tid][16] < window[tid][min]) min=16;
			if (window[tid][17] < window[tid][min]) min=17;
			if (window[tid][18] < window[tid][min]) min=18;
			if (window[tid][19] < window[tid][min]) min=19;
			if (window[tid][20] < window[tid][min]) min=20;
			if (window[tid][21] < window[tid][min]) min=21;
			if (window[tid][22] < window[tid][min]) min=22;
			if (window[tid][23] < window[tid][min]) min=23;
			if (window[tid][24] < window[tid][min]) min=24;
			temp=window[tid][4];window[tid][4]=window[tid][min];window[tid][min]=temp;
			min=5;
			if (window[tid][6] < window[tid][min]) min=6;
			if (window[tid][7] < window[tid][min]) min=7;
			if (window[tid][8] < window[tid][min]) min=8;
			if (window[tid][9] < window[tid][min]) min=9;
			if (window[tid][10] < window[tid][min]) min=10;
			if (window[tid][11] < window[tid][min]) min=11;
			if (window[tid][12] < window[tid][min]) min=12;
			if (window[tid][13] < window[tid][min]) min=13;
			if (window[tid][14] < window[tid][min]) min=14;
			if (window[tid][15] < window[tid][min]) min=15;
			if (window[tid][16] < window[tid][min]) min=16;
			if (window[tid][17] < window[tid][min]) min=17;
			if (window[tid][18] < window[tid][min]) min=18;
			if (window[tid][19] < window[tid][min]) min=19;
			if (window[tid][20] < window[tid][min]) min=20;
			if (window[tid][21] < window[tid][min]) min=21;
			if (window[tid][22] < window[tid][min]) min=22;
			if (window[tid][23] < window[tid][min]) min=23;
			if (window[tid][24] < window[tid][min]) min=24;
			temp=window[tid][5];window[tid][5]=window[tid][min];window[tid][min]=temp;
			min=6;
			if (window[tid][7] < window[tid][min]) min=7;
			if (window[tid][8] < window[tid][min]) min=8;
			if (window[tid][9] < window[tid][min]) min=9;
			if (window[tid][10] < window[tid][min]) min=10;
			if (window[tid][11] < window[tid][min]) min=11;
			if (window[tid][12] < window[tid][min]) min=12;
			if (window[tid][13] < window[tid][min]) min=13;
			if (window[tid][14] < window[tid][min]) min=14;
			if (window[tid][15] < window[tid][min]) min=15;
			if (window[tid][16] < window[tid][min]) min=16;
			if (window[tid][17] < window[tid][min]) min=17;
			if (window[tid][18] < window[tid][min]) min=18;
			if (window[tid][19] < window[tid][min]) min=19;
			if (window[tid][20] < window[tid][min]) min=20;
			if (window[tid][21] < window[tid][min]) min=21;
			if (window[tid][22] < window[tid][min]) min=22;
			if (window[tid][23] < window[tid][min]) min=23;
			if (window[tid][24] < window[tid][min]) min=24;
			temp=window[tid][6];window[tid][6]=window[tid][min];window[tid][min]=temp;
			min=7;
			if (window[tid][8] < window[tid][min]) min=8;
			if (window[tid][9] < window[tid][min]) min=9;
			if (window[tid][10] < window[tid][min]) min=10;
			if (window[tid][11] < window[tid][min]) min=11;
			if (window[tid][12] < window[tid][min]) min=12;
			if (window[tid][13] < window[tid][min]) min=13;
			if (window[tid][14] < window[tid][min]) min=14;
			if (window[tid][15] < window[tid][min]) min=15;
			if (window[tid][16] < window[tid][min]) min=16;
			if (window[tid][17] < window[tid][min]) min=17;
			if (window[tid][18] < window[tid][min]) min=18;
			if (window[tid][19] < window[tid][min]) min=19;
			if (window[tid][20] < window[tid][min]) min=20;
			if (window[tid][21] < window[tid][min]) min=21;
			if (window[tid][22] < window[tid][min]) min=22;
			if (window[tid][23] < window[tid][min]) min=23;
			if (window[tid][24] < window[tid][min]) min=24;
			temp=window[tid][7];window[tid][7]=window[tid][min];window[tid][min]=temp;
			min=8;
			if (window[tid][9] < window[tid][min]) min=9;
			if (window[tid][10] < window[tid][min]) min=10;
			if (window[tid][11] < window[tid][min]) min=11;
			if (window[tid][12] < window[tid][min]) min=12;
			if (window[tid][13] < window[tid][min]) min=13;
			if (window[tid][14] < window[tid][min]) min=14;
			if (window[tid][15] < window[tid][min]) min=15;
			if (window[tid][16] < window[tid][min]) min=16;
			if (window[tid][17] < window[tid][min]) min=17;
			if (window[tid][18] < window[tid][min]) min=18;
			if (window[tid][19] < window[tid][min]) min=19;
			if (window[tid][20] < window[tid][min]) min=20;
			if (window[tid][21] < window[tid][min]) min=21;
			if (window[tid][22] < window[tid][min]) min=22;
			if (window[tid][23] < window[tid][min]) min=23;
			if (window[tid][24] < window[tid][min]) min=24;
			temp=window[tid][8];window[tid][8]=window[tid][min];window[tid][min]=temp;
			min=9;
			if (window[tid][10] < window[tid][min]) min=10;
			if (window[tid][11] < window[tid][min]) min=11;
			if (window[tid][12] < window[tid][min]) min=12;
			if (window[tid][13] < window[tid][min]) min=13;
			if (window[tid][14] < window[tid][min]) min=14;
			if (window[tid][15] < window[tid][min]) min=15;
			if (window[tid][16] < window[tid][min]) min=16;
			if (window[tid][17] < window[tid][min]) min=17;
			if (window[tid][18] < window[tid][min]) min=18;
			if (window[tid][19] < window[tid][min]) min=19;
			if (window[tid][20] < window[tid][min]) min=20;
			if (window[tid][21] < window[tid][min]) min=21;
			if (window[tid][22] < window[tid][min]) min=22;
			if (window[tid][23] < window[tid][min]) min=23;
			if (window[tid][24] < window[tid][min]) min=24;
			temp=window[tid][9];window[tid][9]=window[tid][min];window[tid][min]=temp;
			min=10;
			if (window[tid][11] < window[tid][min]) min=11;
			if (window[tid][12] < window[tid][min]) min=12;
			if (window[tid][13] < window[tid][min]) min=13;
			if (window[tid][14] < window[tid][min]) min=14;
			if (window[tid][15] < window[tid][min]) min=15;
			if (window[tid][16] < window[tid][min]) min=16;
			if (window[tid][17] < window[tid][min]) min=17;
			if (window[tid][18] < window[tid][min]) min=18;
			if (window[tid][19] < window[tid][min]) min=19;
			if (window[tid][20] < window[tid][min]) min=20;
			if (window[tid][21] < window[tid][min]) min=21;
			if (window[tid][22] < window[tid][min]) min=22;
			if (window[tid][23] < window[tid][min]) min=23;
			if (window[tid][24] < window[tid][min]) min=24;
			temp=window[tid][10];window[tid][10]=window[tid][min];window[tid][min]=temp;
			min=11;
			if (window[tid][12] < window[tid][min]) min=12;
			if (window[tid][13] < window[tid][min]) min=13;
			if (window[tid][14] < window[tid][min]) min=14;
			if (window[tid][15] < window[tid][min]) min=15;
			if (window[tid][16] < window[tid][min]) min=16;
			if (window[tid][17] < window[tid][min]) min=17;
			if (window[tid][18] < window[tid][min]) min=18;
			if (window[tid][19] < window[tid][min]) min=19;
			if (window[tid][20] < window[tid][min]) min=20;
			if (window[tid][21] < window[tid][min]) min=21;
			if (window[tid][22] < window[tid][min]) min=22;
			if (window[tid][23] < window[tid][min]) min=23;
			if (window[tid][24] < window[tid][min]) min=24;
			temp=window[tid][11];window[tid][11]=window[tid][min];window[tid][min]=temp;
			min=12;
			if (window[tid][13] < window[tid][min]) min=13;
			if (window[tid][14] < window[tid][min]) min=14;
			if (window[tid][15] < window[tid][min]) min=15;
			if (window[tid][16] < window[tid][min]) min=16;
			if (window[tid][17] < window[tid][min]) min=17;
			if (window[tid][18] < window[tid][min]) min=18;
			if (window[tid][19] < window[tid][min]) min=19;
			if (window[tid][20] < window[tid][min]) min=20;
			if (window[tid][21] < window[tid][min]) min=21;
			if (window[tid][22] < window[tid][min]) min=22;
			if (window[tid][23] < window[tid][min]) min=23;
			if (window[tid][24] < window[tid][min]) min=24;
			temp=window[tid][12];window[tid][12]=window[tid][min];window[tid][min]=temp;

			output[(y*w+x)*c+i] = window[tid][12];
		}
	}
}


template <typename TYPE>
void ctmf_helper_gpu(TYPE* src, TYPE* dst,int width, int height, int channels, int r) {
	dim3 blocks((width+BLOCK-1)/BLOCK, (height+BLOCK-1)/BLOCK);
	if(r == 1) {
		medianFilter3<<<blocks, dim3(BLOCK,BLOCK)>>>(src, dst, width, height, channels);
	} else if(r == 2) {
		medianFilter5<<<blocks, dim3(BLOCK,BLOCK)>>>(src, dst, width, height, channels);
	}
}


void ctmf_gpu(float* src, float* dst,int width, int height,int src_step_row, int dst_step_row,int r, int c, unsigned long memsize) {
	
	GpuArray<float> src_gpu(width*height*c);
	src_gpu.writeDataToDevice(src);

	GpuArray<float> dst_gpu(width*height*c);
	
	ctmf_helper_gpu(src_gpu.getPtr(), dst_gpu.getPtr(), width, height, c, r);

	dst_gpu.readDataToHost(dst);
}

void ctmf_gpu(unsigned char * src, unsigned char * dst,int width, int height,int src_step_row, int dst_step_row,int r, int c, unsigned long memsize) {
	
	GpuArray<unsigned char > src_gpu(width*height*c);
	src_gpu.writeDataToDevice(src);

	GpuArray<unsigned char > dst_gpu(width*height*c);
	
	ctmf_helper_gpu(src_gpu.getPtr(), dst_gpu.getPtr(), width, height, c, r);

	dst_gpu.readDataToHost(dst);
}




