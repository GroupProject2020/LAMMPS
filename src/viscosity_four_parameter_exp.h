#ifndef LAMMPS_VISCOSITY_FOURPARAMETEREXP_H
#define LAMMPS_VISCOSITY_FOURPARAMETEREXP_H

#include "math.h"
#include "viscosity.h"
namespace LAMMPS_NS{

class ViscosityFourParameterExp  : public Viscosity{
private:
    double A;
    double B;
    double C;
    double D;
public:
    ViscosityFourParameterExp(double A, double B, double C, double D);

    double compute_visc(double temperature) override final;

};

};

#endif //LAMMPS_VISCOSITY_FOURPARAMETEREXP_H
