

#include "pair_lj_sph_gpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "atom.h"
#include "atom_vec.h"
#include "USER-SPH/atom_vec_meso.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "integrate.h"
#include "memory.h"
#include "error.h"
#include "neigh_request.h"
#include "universe.h"
#include "update.h"
#include "domain.h"
#include "gpu_extra.h"
#include "suffix.h"

using namespace LAMMPS_NS;

// External functions from cuda library for atom decomposition

int lj_sph_gpu_init(const int ntypes, double **cutsq, double **host_cutsq,
                 double **host_cut, double **host_mass, const int nlocal,
                 const int nall, const int max_nbors, const int maxspecial,
                 const double cell_size, int &gpu_mode, FILE *screen, int domainDim);

void lj_sph_gpu_clear();
int ** ljl_gpu_compute_n(const int ago, const int inum_full,
                         const int nall, double **host_x, double **host_v,
                         double **host_cv, double **host_e, double **host_rho,
                         double **host_de, double **host_drho, int *host_type,
                         double *sublo, double *subhi, tagint *tag, int **nspecial,
                         tagint **special, const bool eflag, const bool vflag,
                         const bool eatom, const bool vatom, int &host_start,
                         int **ilist, int **jnum, const double cpu_time,
                         bool &success);
void ljl_gpu_compute(const int ago, const int inum_full, const int nall,
                     double **host_x, double **host_v, double **host_cv,
                     double **host_e, double **host_rho, double **host_de,
                     double **host_drho, int *host_type, int *ilist, int *numj,
                     int **firstneigh, const bool eflag, const bool vflag,
                     const bool eatom, const bool vatom, int &host_start,
                     const double cpu_time, bool &success, tagint *tag);
double lj_sph_gpu_bytes();


/* ---------------------------------------------------------------------- */

PairSPHLJGPU::PairSPHLJGPU(LAMMPS *lmp) : PairSPHLJ(lmp), gpu_mode(GPU_FORCE)
{
    respa_enable = 0;
    cpu_time = 0.0;
    suffix_flag |= Suffix::GPU;
    GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}


/* ----------------------------------------------------------------------
   free all arrays
------------------------------------------------------------------------- */

PairSPHLJGPU::~PairSPHLJGPU()
{
    lj_sph_gpu_clear();
}

/* ---------------------------------------------------------------------- */

void PairSPHLJGPU::compute(int eflag, int vflag)
{
    ev_init(eflag,vflag);

    int nall = atom->nlocal + atom->nghost;
    int inum, host_start;

    bool success = true;
    int *ilist, *numneigh, **firstneigh;
    if (gpu_mode != GPU_FORCE) {
        inum = atom->nlocal;
        firstneigh = lj_sph_gpu_compute_n(neighbor->ago, inum, nall,
                                       atom->x, atom->v, atom->cv, atom->e, atom->rho,
                                       atom->de, atom->drho, atom->type, domain->sublo,
                                       domain->subhi, atom->tag, atom->nspecial,
                                       atom->special, eflag, vflag, eflag_atom,
                                       vflag_atom, host_start,
                                       &ilist, &numneigh, cpu_time, success, domain->dimension, atom->tag);
    } else {
        inum = list->inum;
        ilist = list->ilist;
        numneigh = list->numneigh;
        firstneigh = list->firstneigh;
        lj_sph_gpu_compute(neighbor->ago, inum, nall, atom->x, atom->v,
                           atom->cv, atom->e, atom->rho,
                           atom->de, atom->drho, atom->type,
                           ilist, numneigh, firstneigh, eflag, vflag, eflag_atom,
                           vflag_atom, host_start, cpu_time, success, domain->dimension, atom->tag);
    }
    if (!success)
        error->one(FLERR,"Insufficient memory on accelerator");

    if (host_start<inum) {
        cpu_time = MPI_Wtime();
        cpu_compute(host_start, inum, eflag, vflag, ilist, numneigh, firstneigh);
        cpu_time = MPI_Wtime() - cpu_time;
    }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSPHLJGPU::init_style()
{
    cut_respa = NULL;

    if (force->newton_pair)
        error->all(FLERR,"Cannot use newton pair with lj/cut/gpu pair style");

    // Repeat cutsq calculation because done after call to init_style
    double maxcut = -1.0;
    double cut;
    for (int i = 1; i <= atom->ntypes; i++) {
        for (int j = i; j <= atom->ntypes; j++) {
            if (setflag[i][j] != 0 || (setflag[i][i] != 0 && setflag[j][j] != 0)) {
                cut = init_one(i,j);
                cut *= cut;
                if (cut > maxcut)
                    maxcut = cut;
                cutsq[i][j] = cutsq[j][i] = cut;
            } else
                cutsq[i][j] = cutsq[j][i] = 0.0;
        }
    }
    double cell_size = sqrt(maxcut) + neighbor->skin;

    int maxspecial=0;
    if (atom->molecular)
        maxspecial=atom->maxspecial;
    int success = lj_sph_gpu_init(atom->ntypes+1, cutsq, cut, mass, atom->nlocal,
                               atom->nlocal+atom->nghost, 300, maxspecial,
                               cell_size, gpu_mode, screen);
    GPU_EXTRA::check_flag(success,error,world);

    if (gpu_mode == GPU_FORCE) {
        int irequest = neighbor->request(this,instance_me);
        neighbor->requests[irequest]->half = 0;
        neighbor->requests[irequest]->full = 1;
    }
}

/* ---------------------------------------------------------------------- */

double PairSPHLJGPU::memory_usage()
{
    double bytes = Pair::memory_usage();
    return bytes + lj_sph_gpu_bytes();
}

/* ---------------------------------------------------------------------- */

void PairSPHLJGPU::cpu_compute(int start, int inum, int eflag, int /* vflag */,
                               int *ilist, int *numneigh, int **firstneigh) {
    int i,j,ii,jj,jnum,itype,jtype;
    double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
    double rsq,r2inv,r6inv,forcelj,factor_lj;
    int *jlist;

    double vxtmp, vytmp, vztmp, imass, jmass, fi, fj, fvisc, h, ih, ihsq, ihcub;
    double rsq, wfd, delVdotDelR, mu, deltaE, ci, cj, lrc;

    double **x = atom->x;
    double **f = atom->f;
    int *type = atom->type;
    double *special_lj = force->special_lj;

    double **v = atom->vest;
    double *rho = atom->rho;
    double *mass = atom->mass;
    double *de = atom->de;
    double *e = atom->e;
    double *cv = atom->cv;
    double *drho = atom->drho;

    // loop over neighbors of my atoms

    for (ii = start; ii < inum; ii++) {
        i = ilist[ii];
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        vxtmp = v[i][0];
        vytmp = v[i][1];
        vztmp = v[i][2];
        itype = type[i];
        jlist = firstneigh[i];
        jnum = numneigh[i];

        imass = mass[itype];

        // compute pressure of particle i with LJ EOS
        LJEOS2(rho[i], e[i], cv[i], &fi, &ci); // TODO: check this LJEOS2
        fi /= (rho[i] * rho[i]);
        //printf("fi = %f\n", fi);

        for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            j &= NEIGHMASK;

            delx = xtmp - x[j][0];
            dely = ytmp - x[j][1];
            delz = ztmp - x[j][2];
            rsq = delx*delx + dely*dely + delz*delz;
            jtype = type[j];
            jmass = mass[jtype];

            if (rsq < cutsq[itype][jtype]) {
                h = cut[itype][jtype];
                ih = 1.0 / h;
                ihsq = ih * ih;
                ihcub = ihsq * ih;

                wfd = h - sqrt(rsq);
                if (domain->dimension == 3) {
                    // Lucy Kernel, 3d
                    // Note that wfd, the derivative of the weight function with respect to r,
                    // is lacking a factor of r.
                    // The missing factor of r is recovered by
                    // (1) using delV . delX instead of delV . (delX/r) and
                    // (2) using f[i][0] += delx * fpair instead of f[i][0] += (delx/r) * fpair
                    wfd = -25.066903536973515383e0 * wfd * wfd * ihsq * ihsq * ihsq * ih;
                } else {
                    // Lucy Kernel, 2d
                    wfd = -19.098593171027440292e0 * wfd * wfd * ihsq * ihsq * ihsq;
                }

                // function call to LJ EOS
                LJEOS2(rho[j], e[j], cv[j], &fj, &cj);
                fj /= (rho[j] * rho[j]);

                // apply long-range correction to model a LJ fluid with cutoff
                // this implies that the modelled LJ fluid has cutoff == SPH cutoff
                lrc = - 11.1701 * (ihcub * ihcub * ihcub - 1.5 * ihcub);
                fi += lrc;
                fj += lrc;

                // dot product of velocity delta and distance vector
                delVdotDelR = delx * (vxtmp - v[j][0]) + dely * (vytmp - v[j][1])
                              + delz * (vztmp - v[j][2]);

                // artificial viscosity (Monaghan 1992)
                if (delVdotDelR < 0.) {
                    mu = h * delVdotDelR / (rsq + 0.01 * h * h);
                    fvisc = -viscosity[itype][jtype] * (ci + cj) * mu / (rho[i] + rho[j]);
                } else {
                    fvisc = 0.;
                }

                // total pair force & thermal energy increment
                fpair = -imass * jmass * (fi + fj + fvisc) * wfd;
                deltaE = -0.5 * fpair * delVdotDelR;

                f[i][0] += delx * fpair;
                f[i][1] += dely * fpair;
                f[i][2] += delz * fpair;

                // and change in density
                drho[i] += jmass * delVdotDelR * wfd;

                // change in thermal energy
                de[i] += deltaE;
                /*
                if (newton_pair || j < nlocal) {
                    f[j][0] -= delx * fpair;
                    f[j][1] -= dely * fpair;
                    f[j][2] -= delz * fpair;
                    de[j] += deltaE;
                    drho[j] += imass * delVdotDelR * wfd;
                }*/
                if (evflag) ev_tally_full(i,evdwl,0.0,fpair,delx,dely,delz);
            }
        }
    }
}
