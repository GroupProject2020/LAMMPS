/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LAMMPS_PAIR_LJ_SPH_GPU_H
#define LAMMPS_PAIR_LJ_SPH_GPU_H

#ifdef PAIR_CLASS

PairStyle(lj/sph/gpu,PairSPHLJGPU)

#else

#include "USER-SPH/pair_sph_lj.h"

namespace LAMMPS_NS {

class PairSPHLJGPU : public PairSPHLJ {
public:
    PairSPHLJGPU(LAMMPS *lmp);
    ~PairSPHLJGPU();
    void cpu_compute(int, int, int, int, int *, int *, int **);
    void compute(int, int);
    void memory_usage();

    enum {GPU_FORCE, GPU_NEIGH, GPU_HYB_NEIGH};

private:
    int gpu_mode;
    double cpu_time;
};
}

#endif //PAIR_CLASS
#endif //LAMMPS_PAIR_LJ_SPH_GPU_H
