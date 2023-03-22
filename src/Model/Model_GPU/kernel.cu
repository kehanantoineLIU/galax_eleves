#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	// unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	accelerationsGPU[i].x = 0.0f;
	accelerationsGPU[i].y = 0.0f;
	accelerationsGPU[i].z = 0.0f;
	for (int j = 0; j < n_particles; j++)
	{
		if(i != j)
		{
			const float diffx = positionsGPU[j].x - positionsGPU[i].x;
			const float diffy = positionsGPU[j].y - positionsGPU[i].y;
			const float diffz = positionsGPU[j].z - positionsGPU[i].z;

			float dij = diffx * diffx + diffy * diffy + diffz * diffz;

			if (dij < 1.0)
			{
				dij = 10.0;
			}
			else
			{
				dij = sqrt(dij);
				dij = 10.0 / (dij * dij * dij);
			}

			accelerationsGPU[i].x += diffx * dij * massesGPU[j];
			accelerationsGPU[i].y += diffy * dij * massesGPU[j];
			accelerationsGPU[i].z += diffz * dij * massesGPU[j];
		}
	}
	
	
}

__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
	velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
	velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;
	positionsGPU[i].x += velocitiesGPU   [i].x * DIFF_T;
	positionsGPU[i].y += velocitiesGPU   [i].y * DIFF_T;
	positionsGPU[i].z += velocitiesGPU   [i].z * DIFF_T;

}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU);
}

#endif // GALAX_MODEL_GPU