#ifndef LAMMPS_VISCOSITY_SUTHERLANDVISCOSITYLAW_H
#define LAMMPS_VISCOSITY_SUTHERLANDVISCOSITYLAW_H

#include "math.h"
#include "viscosity.h"
namespace LAMMPS_NS{

class SutherlandViscosityLaw  : public Viscosity{
private:
    double A;
    double B;

public:
    SutherlandViscosityLaw(double A, double B);

    double compute_visc(double temperature) override final;

};

};

#endif // LAMMPS_VISCOSITY_SUTHERLANDVISCOSITYLAW_H