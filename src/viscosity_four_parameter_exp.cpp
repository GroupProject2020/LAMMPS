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

#include "viscosity_four_parameter_exp.h"

using namespace LAMMPS_NS;

ViscosityFourParameterExp::ViscosityFourParameterExp(double A, double B, double C, double D) {
    this->A = A;
    this->B = B;
    this->C = C;
    this->D = D;
}

double ViscosityFourParameterExp::compute_visc(double temperature) {
    return  A*exp(B/temperature +C*temperature + D *temperature*temperature);
}