//
// Created by wall-e on 23/03/2020.
//

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