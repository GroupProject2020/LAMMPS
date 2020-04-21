
#include "viscosity_arrhenius.h"

using namespace LAMMPS_NS;

ViscosityArrhenius::ViscosityArrhenius(double C1, double C2){
    this->C1 = C1;
    this->C2 = C2;
}

double ViscosityArrhenius::compute_visc(double temperature) {
    return  C1*exp(C2/temperature);
}