//
// Created by wall-e on 23/03/2020.
//

#ifndef LAMMPS_VISCOSITY_H
#define LAMMPS_VISCOSITY_H

namespace LAMMPS_NS {
    class Viscosity {
    public:
        Viscosity();

        virtual double compute_visc(double temperature) = 0;
    };

}
#endif //LAMMPS_VISCOSITY_H
