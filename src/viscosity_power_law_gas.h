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

#ifndef LAMMPS_VISCOSITY_POWERLAWGAS_H
#define LAMMPS_VISCOSITY_POWERLAWGAS_H

#include "math.h"
#include "viscosity.h"
namespace LAMMPS_NS{

class PowerLawGas  : public Viscosity{
    /**
     * Implementation of the Power law gas viscosity.
     * This viscosity has one attribute.
     */
private:
    double B;
public:
    PowerLawGas(double B);

    double compute_visc(double temperature) override final;

};

};

#endif //LAMMPS_VISCOSITY_POWERLAWGAS_H