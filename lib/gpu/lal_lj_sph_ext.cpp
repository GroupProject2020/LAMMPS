
#include <iostream>
#include <cassert>
#include <cmath>

#include "lal_lj_sph.h"

using namespace std;
using namespace LAMMPS_AL;

static LJ_SPH<PRECISION,ACC_PRECISION> LJSPHMF; //TODO: adapted template


// ---------------------------------------------------------------------------
// Allocate memory on host and device and copy constants to device
// ---------------------------------------------------------------------------
int lj_sph_gpu_init(const int ntypes, double **host_cutsq,
                    double **host_cut, double **host_mass, const int inum,
                    const int nall, const int max_nbors, const int maxspecial,
                    const double cell_size, int &gpu_mode, FILE *screen, int domainDim) {
    LJSPHMF.clear();
    gpu_mode=LJSPHMF.device->gpu_mode();
    double gpu_split=LJSPHMF.device->particle_split();
    int first_gpu=LJSPHMF.device->first_device();
    int last_gpu=LJSPHMF.device->last_device();
    int world_me=LJSPHMF.device->world_me();
    int gpu_rank=LJSPHMF.device->gpu_rank();
    int procs_per_gpu=LJSPHMF.device->procs_per_gpu();

    LJSPHMF.device->init_message(screen,"lj/sph",first_gpu,last_gpu);

    bool message=false;
    if (LJSPHMF.device->replica_me()==0 && screen)
        message=true;

    if (message) {
        fprintf(screen,"Initializing Device and compiling on process 0...");
        fflush(screen);
    }

    int init_ok=0;
    if (world_me==0)
        init_ok=LJSPHMF.init(ntypes, host_cutsq, host_cut, host_mass, inum, nall, 300,
                           maxspecial, cell_size, gpu_split, screen, domainDim);

    LJSPHMF.device->world_barrier();
    if (message)
        fprintf(screen,"Done.\n");

    for (int i=0; i<procs_per_gpu; i++) {
        if (message) {
            if (last_gpu-first_gpu==0)
                fprintf(screen,"Initializing Device %d on core %d...",first_gpu,i);
            else
                fprintf(screen,"Initializing Devices %d-%d on core %d...",first_gpu,
                        last_gpu,i);
            fflush(screen);
        }
        if (gpu_rank==i && world_me!=0)
            init_ok=LJSPHMF.init(ntypes, host_cutsq, host_cut, host_mass, inum, nall, 300,
                                  maxspecial, cell_size, gpu_split, screen, domainDim);

        LJSPHMF.device->gpu_barrier();
        if (message)
            fprintf(screen,"Done.\n");
    }
    if (message)
        fprintf(screen,"\n");

    if (init_ok==0)
        LJSPHMF.estimate_gpu_overhead();
    return init_ok;
}

// ---------------------------------------------------------------------------
// Copy updated coeffs from host to device
// ---------------------------------------------------------------------------
void ljl_gpu_reinit(const int ntypes, double **host_cutsq,
                    double **host_cut, double **host_mass) {
    int world_me=LJSPHMF.device->world_me();
    int gpu_rank=LJSPHMF.device->gpu_rank();
    int procs_per_gpu=LJSPHMF.device->procs_per_gpu();

    if (world_me==0)
        LJSPHMF.reinit(ntypes, host_cutsq, host_cut, host_mass);
    LJSPHMF.device->world_barrier();

    for (int i=0; i<procs_per_gpu; i++) {
        if (gpu_rank==i && world_me!=0)
            LJSPHMF.reinit(ntypes, host_cutsq, host_cut, host_mass);
        LJSPHMF.device->gpu_barrier();
    }
}

void ljl_gpu_clear() {
    LJSPHMF.clear();
}

int ** ljl_gpu_compute_n(const int ago, const int inum_full,
                         const int nall, double **host_x, double **host_v,
                         double **host_cv, double **host_e, double **host_rho,
                         double **host_de, double **host_drho, int *host_type,
                         double *sublo, double *subhi, tagint *tag, int **nspecial,
                         tagint **special, const bool eflag, const bool vflag,
                         const bool eatom, const bool vatom, int &host_start,
                         int **ilist, int **jnum, const double cpu_time,
                         bool &success) {
    return LJSPHMF.compute(ago, inum_full, nall, host_x, host_v,
                           host_cv, host_e, host_rho,
                           host_de, host_drho, host_type, sublo,
                           subhi, tag, nspecial, special, eflag, vflag, eatom,
                           vatom, host_start, ilist, jnum, cpu_time, success);
}

void ljl_gpu_compute(const int ago, const int inum_full, const int nall,
                     double **host_x, double **host_v, double **host_cv,
                     double **host_e, double **host_rho, double **host_de,
                     double **host_drho, int *host_type, int *ilist, int *numj,
                     int **firstneigh, const bool eflag, const bool vflag,
                     const bool eatom, const bool vatom, int &host_start,
                     const double cpu_time, bool &success, tagint* tag) {
    LJSPHMF.compute(ago, inum_full, nall, host_x, host_v, host_cv, host_e,
                    host_rho, host_de, host_drho, host_type,ilist,numj,
                    firstneigh,eflag,vflag,eatom,vatom,host_start,cpu_time,success, tag);
}

double ljl_gpu_bytes() {
    return LJSPHMF.host_memory_usage();
}
