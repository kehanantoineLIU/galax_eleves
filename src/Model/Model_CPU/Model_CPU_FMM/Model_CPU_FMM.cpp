#ifdef GALAX_MODEL_CPU_FMM

#include <cmath>

#include "Model_CPU_FMM.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;

Model_CPU_FMM
::Model_CPU_FMM(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
	root = new Cell(R_ROOT);
	r_min = R_ROOT;
}

Model_CPU_FMM::~Model_CPU_FMM()
{
	delete root;
}

void Model_CPU_FMM
::step()
{
	std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
	std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
	std::fill(accelerationsz.begin(), accelerationsz.end(), 0);
	// root = new Cell(R_ROOT);
	BuildTree();
	// MeasureTreeDepth(root);
	//std::cout << "Depth: " << r_min << "\n";
	GetMultipole(root);

	// #pragma omp parallel for
    // for (int i = 0; i < n_particles; i++)
    // {
    //     Evaluate(root, i);
    // }
    // for (const auto &item : accelerationsx) 
    // {
    //     std::cout << item << "\n";
    // }
	// std::cout << "\n" << std::endl;
    // #pragma omp parallel for
	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;	
	}

	ResetTree(root);
	// delete root;
}

void Model_CPU_FMM::Evaluate(Cell* entry, int index_target)
{
	if(entry->child)
	{
		for(int octant=0; octant<8; octant++)
		{
			if(entry->nchild & (1 << octant))
			{
				Cell* c = entry->child[octant];
                float diffx = particles.x[index_target] - c->x;
                float diffy = particles.y[index_target] - c->y;
                float diffz = particles.z[index_target] - c->z;
				float distance = std::sqrt(diffx * diffx + diffy * diffy + diffz * diffz);
				if(c->r > THETA*distance)
				{
					Evaluate(c, index_target);
				}
				else
				{
                    // Using FMM
                    float weightx[10] = {0.0f};
                    float weighty[10] = {0.0f};
					float weightz[10] = {0.0f};
                    GetWeights(diffx, diffy, diffz, distance, weightx, weighty, weightz);

					accelerationsx[index_target] += 10 * DotProduct<10>(weightx, c->multipole);
					accelerationsy[index_target] += 10 * DotProduct<10>(weighty, c->multipole);
					accelerationsz[index_target] += 10 * DotProduct<10>(weightz, c->multipole);
				}
			}    
		}
	}
	else
	{
		for(int i=0; i<entry->nleaf; i++)
		{
			int index_source = entry->leaf[i];
			if(index_source != index_target)
			{
                //i: target     j: source
				const float diffx = particles.x[index_source] - particles.x[index_target];
				const float diffy = particles.y[index_source] - particles.y[index_target];
				const float diffz = particles.z[index_source] - particles.z[index_target];

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

				accelerationsx[index_target] += diffx * dij * initstate.masses[index_source];
				accelerationsy[index_target] += diffy * dij * initstate.masses[index_source];
				accelerationsz[index_target] += diffz * dij * initstate.masses[index_source];
			}
		}
	}
}
#endif // GALAX_MODEL_CPU_FMM