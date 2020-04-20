#include "viscosity_power_law_gas.h"

using namespace LAMMPS_NS;

PowerLawGas::PowerLawGas(double B) {
    this->B = B;
}

double PowerLawGas::compute_visc(double temperature) {
    return B*pow(temperature,2/3);
}