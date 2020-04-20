#include "viscosity_sutherland_law.h"

using namespace LAMMPS_NS;

SutherlandViscosityLaw::SutherlandViscosityLaw(double A, double B) {
    this->A = A;
    this->B = B;
}

double SutherlandViscosityLaw::compute_visc(double temperature) {
    return (A*pow(temperature,3/2))/(temperature + B);
}