#include "compute_meso_viscosities_atom.h"
#include <cstring>
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeMesoViscositiesAtom::ComputeMesoViscositiesAtom(LAMMPS *lmp, int narg, char **arg) :
        Compute(lmp, narg, arg)
{
    if (narg != 3) error->all(FLERR,"Number of arguments for compute meso/viscosities/atom command != 3");
    if (atom->e_flag != 1) error->all(FLERR,"compute meso/viscosities/atom command requires atom_style with energy (e.g. meso)");

    peratom_flag = 1;
    size_peratom_cols = 0;

    nmax = 0;
    viscositiesVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeMesoViscositiesAtom::~ComputeMesoViscositiesAtom()
{
    memory->sfree(viscositiesVector);
}

/* ---------------------------------------------------------------------- */

void ComputeMesoViscositiesAtom::init()
{

    int count = 0;
    for (int i = 0; i < modify->ncompute; i++)
        if (strcmp(modify->compute[i]->style,"viscositiesVector/atom") == 0) count++;
    if (count > 1 && comm->me == 0)
        error->warning(FLERR,"More than one compute viscositiesVector/atom");
}

/* ---------------------------------------------------------------------- */

void ComputeMesoViscositiesAtom::compute_peratom()
{
    invoked_peratom = update->ntimestep;

    // grow viscositiesVector array if necessary

    if (atom->nmax > nmax) {
        memory->sfree(viscositiesVector);
        nmax = atom->nmax;
        viscositiesVector = (double *) memory->smalloc(nmax*sizeof(double),"viscositiesVector/atom:viscositiesVector");
        vector_atom = viscositiesVector;
    }

    double *viscosities = atom->viscosities;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            viscositiesVector[i] = viscosities[i];
        }
        else {
            viscositiesVector[i] = 0.0;
        }
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeMesoViscositiesAtom::memory_usage()
{
    double bytes = nmax * sizeof(double);
    return bytes;
}
