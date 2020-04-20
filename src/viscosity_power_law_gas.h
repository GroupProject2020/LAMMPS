#ifndef LAMMPS_VISCOSITY_POWERLAWGAS_H
#define LAMMPS_VISCOSITY_POWERLAWGAS_H

#include "math.h"
#include "viscosity.h"
namespace LAMMPS_NS{

class PowerLawGas  : public Viscosity{
private:
    double B;
public:
    PowerLawGas(double B);

    double compute_visc(double temperature) override;

};

};

#endif //LAMMPS_VISCOSITY_POWERLAWGAS_H