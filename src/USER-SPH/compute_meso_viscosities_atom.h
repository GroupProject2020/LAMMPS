#ifdef COMPUTE_CLASS

ComputeStyle(meso/viscosities/atom,ComputeMesoViscositiesAtom)

#else

#ifndef LMP_COMPUTE_MESO_VISCOSITIES_ATOM_H
#define LMP_COMPUTE_MESO_VISCOSITIES_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

    class ComputeMesoViscositiesAtom : public Compute {
    public:
        ComputeMesoViscositiesAtom(class LAMMPS *, int, char **);
        ~ComputeMesoViscositiesAtom();
        void init();
        void compute_peratom();
        double memory_usage();

    private:
        int nmax;
        double *viscositiesVector;
    };

}

#endif //LAMMPS_COMPUTE_MESO_VISCOSITIES_ATOM_H
#endif
