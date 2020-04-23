

#ifndef LAMMPS_PAIR_LJ_SPH_GPU_H
#define LAMMPS_PAIR_LJ_SPH_GPU_H

#ifdef PAIR_CLASS

PairStyle(lj/sph/gpu,PairSPHLJGPU)

#else

#ifndef LMP_PAIR_LJ_SPH_GPU_H
#define LMP_PAIR_LJ_SPH_GPU_H

#include "USER-SPH/pair_sph_lj.h"

namespace LAMMPS_NS {

class PairSPHLJGPU : public PairSPHLJ {
public:
    PairSPHLJGPU(LAMMPS *lmp);
    ~PairSPHLJGPU();
    void cpu_compute(int, int, int, int, int *, int *, int **);
    void compute(int, int);
    void init_style();
    void reinit();
    void memory_usage();

    enum {GPU_FORCE, GPU_NEIGH, GPU_HYB_NEIGH};

private:
    int gpu_mode;
    double cpu_time;
};
}

#endif //PAIR_CLASS
#endif //LAMMPS_PAIR_LJ_SPH_GPU_H
