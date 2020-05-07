#ifndef LAMMPS_VISCOSITY_ARRHENIUS_H
#define LAMMPS_VISCOSITY_ARRHENIUS_H

#include "math.h"
#include "viscosity.h"

namespace LAMMPS_NS{

    class ViscosityArrhenius : public Viscosity {
    private:
        double C1;
        double C2;
    public:
        ViscosityArrhenius(double C1, double C2);

        double compute_visc(double temperature) override final;

    };
};


#endif //LAMMPS_VISCOSITY_ARRHENIUS_H
