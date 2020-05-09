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

#ifndef LAMMPS_VISCOSITY_H
#define LAMMPS_VISCOSITY_H

namespace LAMMPS_NS {
    class Viscosity {
        /**
         * Abstract base class for the viscosity attribute.
         * All viscosity types should inherit from this class.
         */
    public:
        Viscosity();
        /**
         * Virtual function.
         * Returns the viscosity, given the temperature.
         */
        virtual double compute_visc(double temperature) = 0;
    };

}
#endif //LAMMPS_VISCOSITY_H
