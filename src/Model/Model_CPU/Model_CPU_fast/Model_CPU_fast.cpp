#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>
#include <immintrin.h>

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;

#define ROTATION

Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
	// int size_ex = n_particles + b_type::size*2 -2;
	// posx_ex = std::vector<float>(size_ex, 0.0f);
	// posy_ex = std::vector<float>(size_ex, 0.0f);
	// posz_ex = std::vector<float>(size_ex, 0.0f);
	// mass_ex = std::vector<float>(size_ex, 0.0f);
}

void Model_CPU_fast
::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

#ifdef OMP
    #pragma omp parallel for // collapse(2)
	for (int i = 0; i < n_particles; i++)
	{
		for (int j = 0; j < n_particles; j++)
		{
			if(i != j)
			{
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;

				if (dij < 1.0)
				{
					dij = 10.0;
				}
				else
				{
					dij = std::sqrt(dij);
					dij = 10.0 / (dij * dij * dij);
				}

				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];
			}
		}
	}

    #pragma omp parallel for
	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;
	}
#endif

// OMP + xsimd version
#ifdef SLIDING_WINDOW
// Sliding window
	const b_type dij_threshold = b_type::broadcast(1.0);
	const b_type dij_max = b_type::broadcast(10.0);
	auto vec_size = n_particles-n_particles%b_type::size;

	#pragma omp parallel for
		for (int i = 0; i < vec_size; i += b_type::size)
		{
			b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
			b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
			b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

			// load registers body i
			const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
			const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
			const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);

			for (int j = 0; j < n_particles; j++)
			{
				if(i != j)
				{
					const b_type rposx_j = b_type::load_unaligned(&particles.x[j]);
					const b_type rposy_j = b_type::load_unaligned(&particles.y[j]);
					const b_type rposz_j = b_type::load_unaligned(&particles.z[j]);
					const b_type rmass_j = b_type::load_unaligned(&initstate.masses[j]);

					b_type diffx = rposx_j - rposx_i;
					b_type diffy = rposy_j - rposy_i;
					b_type diffz = rposz_j - rposz_i;
					b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;

					auto mask_threshold = xs::lt(dij, dij_threshold);
					
					dij =  _mm256_isqrt_ps(dij);
					dij  = 10.0 * (dij * dij * dij);
					dij = xs::select(mask_threshold, dij_max, dij);

					raccx_i += diffx * dij * rmass_j;
					raccy_i += diffy * dij * rmass_j;
					raccz_i += diffz * dij * rmass_j;
				}
			}

			raccx_i.store_unaligned(&accelerationsx[i]);
			raccy_i.store_unaligned(&accelerationsy[i]);
			raccz_i.store_unaligned(&accelerationsz[i]);
		}
#endif

#ifdef IMPROVED_SLIDING_WINDOW
//Improved sliding window
	auto vec_size = n_particles-n_particles%b_type::size;

	int size_ex = n_particles + b_type::size*2 -2;
	std::vector<float> posx_ex(size_ex, 0.0f);
	std::vector<float> posy_ex(size_ex, 0.0f);
	std::vector<float> posz_ex(size_ex, 0.0f);
	std::vector<float> mass_ex(size_ex, 0.0f);

	std::copy(particles.x.begin(), particles.x.end(), posx_ex.begin()+b_type::size-1);
	std::copy(particles.y.begin(), particles.y.end(), posy_ex.begin()+b_type::size-1);
	std::copy(particles.z.begin(), particles.z.end(), posz_ex.begin()+b_type::size-1);
	std::copy(initstate.masses.begin(), initstate.masses.end(), mass_ex.begin()+b_type::size-1);

	b_type raccx_i = b_type::load_unaligned(&accelerationsx[0]);
	b_type raccy_i = b_type::load_unaligned(&accelerationsy[0]);
	b_type raccz_i = b_type::load_unaligned(&accelerationsz[0]);

	#pragma omp parallel for
		for (int i = 0; i < vec_size; i += b_type::size)
		{
			b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
			b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
			b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

			// load registers body i
			const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
			const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
			const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);

			for (int j = 0; j < n_particles + b_type::size - 1; j++)
			// for (int j = 0; j <  vec_size+ b_type::size - 1; j += b_type::size)
			{
				if((i + b_type::size - 1) != j)
				{
					const b_type rposx_j = b_type::load_unaligned(&posx_ex[j]);
					const b_type rposy_j = b_type::load_unaligned(&posy_ex[j]);
					const b_type rposz_j = b_type::load_unaligned(&posz_ex[j]);
					const b_type rmass_j = b_type::load_unaligned(&mass_ex[j]);

					b_type diffx = rposx_j - rposx_i;
					b_type diffy = rposy_j - rposy_i;
					b_type diffz = rposz_j - rposz_i;
					b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;

					// auto mask_threshold = xs::lt(dij, dij_threshold);
					
					dij =  _mm256_rsqrt_ps(dij);
					dij = xs::clip(dij, b_type::broadcast(0.0), b_type::broadcast(1.0));
					dij =  10.0 * (dij * dij * dij);
					// dij = xs::select(mask_threshold, dij_max, dij);

					raccx_i += diffx * dij * rmass_j;
					raccy_i += diffy * dij * rmass_j;
					raccz_i += diffz * dij * rmass_j;
				}
			}
			raccx_i.store_unaligned(&accelerationsx[i]);
			raccy_i.store_unaligned(&accelerationsy[i]);
			raccz_i.store_unaligned(&accelerationsz[i]);
		}
#endif












#ifdef ROTATION
//methode 2.0
	// Rotation
	auto vec_size = n_particles-n_particles%b_type::size;
	struct Rot {
		static constexpr unsigned get(unsigned i, unsigned n) {
			return (i + n - 1) % n;
		}
	};

	#pragma omp parallel for
	for (int i = 0; i < vec_size; i += b_type::size)
	{
         // load registers body i
         const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
         const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
         const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
               b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
               b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
               b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

        // calculate force
		for (int j = 0; j <  vec_size; j += b_type::size)
		{
			 b_type rposx_j = b_type();
			 b_type rposy_j = b_type();
			 b_type rposz_j = b_type();
			
			for (int k = 0; k < b_type::size; k++) {
				
				if(k == 0) {
					 // load registers body j
					rposx_j = b_type::load_unaligned(&particles.x[j]);
					rposy_j = b_type::load_unaligned(&particles.y[j]);
					rposz_j = b_type::load_unaligned(&particles.z[j]);
					if(i==j)
					{
						continue;
					}
				} else {

        			using index_type = xsimd::as_unsigned_integer_t<b_type>;
					const auto mask = xs::make_batch_constant<index_type, Rot>();
					rposx_j = xs::swizzle(rposx_j , mask);
					rposy_j = xs::swizzle(rposy_j, mask);
					rposz_j = xs::swizzle(rposz_j, mask);
				}

				const b_type diffx = rposx_j - rposx_i;
				const b_type diffy = rposy_j - rposy_i;
				const b_type diffz = rposz_j - rposz_i;

				b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;
				const auto comp = xs::lt(dij, b_type(1.0f));
				//dij = xs::select(comp, b_type(10.0f), b_type(10.0f) / xs::pow(dij, b_type(3.0f/2.0f)));
				dij = xs::select(comp, b_type(10.0f), b_type(10.0f) * xs::rsqrt((dij * dij * dij)));

				
				raccx_i += diffx * dij * initstate.masses[j];
				raccy_i += diffy * dij * initstate.masses[j];
				raccz_i += diffz * dij * initstate.masses[j];
			}
		}
		
		// load register into memory
		raccx_i.store_unaligned(&accelerationsx[i]);
        raccy_i.store_unaligned(&accelerationsy[i]);
        raccz_i.store_unaligned(&accelerationsz[i]);
    }
#endif






//method 3
	// #pragma omp parallel for // collapse(2)
	// for (int i = vec_size; i < n_particles; i++)
	// {
	// 	for (int j = 0; j < n_particles; j++)
	// 	{
	// 		if(i != j)
	// 		{
	// 			const float diffx = particles.x[j] - particles.x[i];
	// 			const float diffy = particles.y[j] - particles.y[i];
	// 			const float diffz = particles.z[j] - particles.z[i];

	// 			float dij = diffx * diffx + diffy * diffy + diffz * diffz;

	// 			if (dij < 1.0)
	// 			{
	// 				dij = 10.0;
	// 			}
	// 			else
	// 			{
	// 				dij = std::sqrt(dij);
	// 				dij = 10.0 / (dij * dij * dij);
	// 			}

	// 			accelerationsx[i] += diffx * dij * initstate.masses[j];
	// 			accelerationsy[i] += diffy * dij * initstate.masses[j];
	// 			accelerationsz[i] += diffz * dij * initstate.masses[j];
	// 		}
	// 	}
	// }

	// #pragma omp parallel for
	// 	for (int i = 0; i < n_particles; i += b_type::size)
	// 	{	
	// 		// load registers body i
	// 		const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
	// 		const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
	// 		const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
	// 		b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
	// 		b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
	// 		b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);
	// 		for (int j = 0; j < n_particles; j++)
	// 		{
	// 			if(j>i-1 && j<i+b_type::size){
	// 				for(int x=i; x<i+b_type::size; x++){
	// 					if(x != j){
	// 						const float diffx = particles.x[j] - particles.x[x];
	// 						const float diffy = particles.y[j] - particles.y[x];
	// 						const float diffz = particles.z[j] - particles.z[x];

	// 						float dij = diffx * diffx + diffy * diffy + diffz * diffz;

	// 						if (dij < 1.0)
	// 						{
	// 							dij = 10.0;
	// 						}
	// 						else
	// 						{
	// 							dij = std::sqrt(dij);
	// 							dij = 10.0 / (dij * dij * dij);
	// 						}

	// 						accelerationsx[x] += diffx * dij * initstate.masses[j];
	// 						accelerationsy[x] += diffy * dij * initstate.masses[j];
	// 						accelerationsz[x] += diffz * dij * initstate.masses[j];
	// 					}
	// 				}
	// 				raccx_i = b_type::load_unaligned(&accelerationsx[i]);
	// 				raccy_i = b_type::load_unaligned(&accelerationsy[i]);
	// 				raccz_i = b_type::load_unaligned(&accelerationsz[i]);	
	// 			}else{
	// 				// load registers body j
	// 				const b_type rposx_j = b_type::broadcast(particles.x[j]);
	// 				const b_type rposy_j = b_type::broadcast(particles.y[j]);
	// 				const b_type rposz_j = b_type::broadcast(particles.z[j]);
	// 				const b_type rmas_j = b_type::broadcast(initstate.masses[j]);

	// 				b_type diffx = rposx_j - rposx_i;
	// 				b_type diffy = rposy_j - rposy_i;
	// 				b_type diffz = rposz_j - rposz_i;

	// 				b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;

	// 				b_type dij_sqrt = xs::sqrt(dij);
	// 				b_type dij_real = 10.0 / (dij_sqrt * dij_sqrt * dij_sqrt);

	// 				dij = select(lt(dij, b_type::broadcast(1.0)), b_type::broadcast(10.0), dij_real);

	// 				raccx_i += diffx * dij * rmas_j;
	// 				raccy_i += diffy * dij * rmas_j;
	// 				raccz_i += diffz * dij * rmas_j;

	// 				raccx_i.store_unaligned(&accelerationsx[i]);
	// 				raccy_i.store_unaligned(&accelerationsy[i]);
	// 				raccz_i.store_unaligned(&accelerationsz[i]);
	// 			}
	// 		}    
	// 	}


#ifndef OMP
	#pragma omp parallel for
	for (int i = 0; i < vec_size; i += b_type::size)
	{
		// Load
		b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
		b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
		b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

		b_type rvelx_i = b_type::load_unaligned(&velocitiesx[i]);
		b_type rvely_i = b_type::load_unaligned(&velocitiesy[i]);
		b_type rvelz_i = b_type::load_unaligned(&velocitiesz[i]);

		b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
		b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
		b_type rposz_i = b_type::load_unaligned(&particles.z[i]);

		//Calculate
		rvelx_i += raccx_i * b_type::broadcast(2.0f);
		rvely_i += raccy_i * b_type::broadcast(2.0f);
		rvelz_i += raccz_i * b_type::broadcast(2.0f);

		rposx_i += rvelx_i * b_type::broadcast(0.1f);
		rposy_i += rvely_i * b_type::broadcast(0.1f);
		rposz_i += rvelz_i * b_type::broadcast(0.1f);

		//Store
		rvelx_i.store_unaligned(&velocitiesx[i]);
		rvely_i.store_unaligned(&velocitiesy[i]);
		rvelz_i.store_unaligned(&velocitiesz[i]);

		rposx_i.store_unaligned(&particles.x[i]);
		rposy_i.store_unaligned(&particles.y[i]);
		rposz_i.store_unaligned(&particles.z[i]);
	}

	#pragma omp parallel for
	for (int i = vec_size; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;
	}
#endif







}

#endif // GALAX_MODEL_CPU_FAST