#ifndef LAMMPS_VISCOSITY_CONSTANT_H
#define LAMMPS_VISCOSITY_CONSTANT_H


#include "math.h"
#include "viscosity.h"

namespace LAMMPS_NS{

    class ViscosityConstant : public Viscosity {
    private:
        double A;
    public:
        ViscosityConstant(double A);

        double compute_visc(double temperature) override final;

    };
};


#endif //LAMMPS_VISCOSITY_CONSTANT_H
