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

#include "viscosity_arrhenius.h"

using namespace LAMMPS_NS;

ViscosityArrhenius::ViscosityArrhenius(double C1, double C2){
    this->C1 = C1;
    this->C2 = C2;
}

double ViscosityArrhenius::compute_visc(double temperature) {
    return  C1*exp(C2/temperature);
}