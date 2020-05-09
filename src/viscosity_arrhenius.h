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

#ifndef LAMMPS_VISCOSITY_ARRHENIUS_H
#define LAMMPS_VISCOSITY_ARRHENIUS_H

#include "math.h"
#include "viscosity.h"

namespace LAMMPS_NS{

    class ViscosityArrhenius : public Viscosity {
        /**
         * Implementation of the Arrhenius viscosity.
         * This viscosity has two attributes.
         */
    private:
        double C1;
        double C2;
    public:
        ViscosityArrhenius(double C1, double C2);

        double compute_visc(double temperature) override final;

    };
};


#endif //LAMMPS_VISCOSITY_ARRHENIUS_H
