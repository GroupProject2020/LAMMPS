
#include "viscosity_constant.h"

using namespace LAMMPS_NS;

ViscosityConstant::ViscosityConstant(double A){
    this->A = A;
}

double ViscosityConstant::compute_visc(double temperature) {
    return  A;
}
