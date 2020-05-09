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

#ifdef COMPUTE_CLASS

ComputeStyle(meso/viscosities/atom,ComputeMesoViscositiesAtom)

#else

#ifndef LMP_COMPUTE_MESO_VISCOSITIES_ATOM_H
#define LMP_COMPUTE_MESO_VISCOSITIES_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

    class ComputeMesoViscositiesAtom : public Compute {
        /**
         * Implementation of the compute class
         * adapted to compute the viscosities attribute.
         * The viscosities attribute stores the dynamic viscosity.
         */
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
