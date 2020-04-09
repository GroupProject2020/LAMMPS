#if defined(USE_OPENCL)
#include "lj_cl.h"
#elif defined(USE_CUDART)
const char *lj=0;
#else
#include "lj_cubin.h"
#endif

#include "lal_lj_sph.h"

#include <cassert>
namespace LAMMPS_AL {
#define LJ_SPHT LJ_SPH<numtyp, acctyp>

extern Device<PRECISION,ACC_PRECISION> device;

template <class numtyp, class acctyp>
LJ_SPHT::LJ_SPH() : BaseAtomicSPH<numtyp,acctyp>(), _allocated(false) {
}

template <class numtyp, class acctyp>
LJ_SPHT::~LJ_SPH() {
    clear();
}

template <class numtyp, class acctyp>
int LJ_SPHT::bytes_per_atom(const int max_nbors) const {
    return this->bytes_per_atom_atomic(max_nbors);
}

template <class numtyp, class acctyp>
int LJ_SPHT::init(const int ntypes, double **host_cutsq,
              double **host_cut, double **host_mass,
              const int nlocal, const int nall, const int max_nbors,
              const int maxspecial, const double cell_size,
              const double gpu_split, FILE *screen) {
    int success;
    success=this->init_atomic(nlocal,nall,max_nbors,maxspecial,cell_size,gpu_split,
                              _screen,lj,"k_lj_sph");
    if (success!=0)
        return success;

    // If atom type constants fit in shared memory use fast kernel
    int lj_types=ntypes;
    shared_types=false;
    int max_shared_types=this->device->max_shared_types();
    if (lj_types<=max_shared_types && this->_block_size>=max_shared_types) {
        lj_types=max_shared_types;
        shared_types=true;
    }
    _lj_types=lj_types;

    // Allocate a host write buffer for data initialization
    UCL_H_Vec<numtyp> host_write(lj_types*lj_types*32,*(this->ucl_device),
                                 UCL_WRITE_ONLY);

    for (int i=0; i<lj_types*lj_types; i++)
        host_write[i]=0.0;

    cuts.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
    this->atom->type_pack4(ntypes,lj_types,cuts,host_write,host_cutsq,host_cut,
                           host_mass);


    _allocated=true;
    this->_max_bytes=cuts.row_bytes();
    return 0;
}

template <class numtyp, class acctyp>
void LJ_SPHT::reinit(const int ntypes, double **host_cutsq,
                 double **host_cut, double **host_mass) {
    // Allocate a host write buffer for data initialization
    UCL_H_Vec<numtyp> host_write(_lj_types*_lj_types*32,*(this->ucl_device),
            UCL_WRITE_ONLY);

    for (int i=0; i<_lj_types*_lj_types; i++)
        host_write[i]=0.0;

    this->atom->type_pack4(ntypes,lj_types,cuts,host_write,host_cutsq,host_cut,
                           host_mass);
}

template <class numtyp, class acctyp>
void LJ_SPHT::clear() {
    if (!_allocated)
        return;
    _allocated=false;

    cuts.clear();
    this->clear_atomic();
}

template <class numtyp, class acctyp>
double LJ_SPHT::host_memory_usage() const {
    return this->host_memory_usage_atomic()+sizeof(LJ_SPH<numtyp,acctyp>);
}

// ---------------------------------------------------------------------------
// Calculate energies, forces, and torques
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
void LJ_SPHT::loop(const bool _eflag, const bool _vflag) {
    // Compute the block size and grid size to keep all cores busy
    const int BX=this->block_size();
    int eflag, vflag;
    if (_eflag)
        eflag=1;
    else
        eflag=0;

    if (_vflag)
        vflag=1;
    else
        vflag=0;

    int GX=static_cast<int>(ceil(static_cast<double>(this->ans->inum())/
                                 (BX/this->_threads_per_atom)));

    int ainum=this->ans->inum();
    int nbor_pitch=this->nbor->nbor_pitch();
    this->time_pair.start();
    if (shared_types) {
        this->k_pair_fast.set_size(GX,BX);
        this->k_pair_fast.run(&this->atom->x, &this->atom->v, &this->atom->cv,
                              &this->atom->e, &this->atom->rho, &this->atom->de,
                              &this->atom->drho, &this->atom->cuts,
                              &this->nbor->dev_nbor, &this->_nbor_data->begin(),
                              &this->ans->force, &this->ans->engv, &eflag, &vflag,
                              &ainum, &nbor_pitch, &this->_threads_per_atom);
    } else {
        this->k_pair.set_size(GX,BX);
        this->k_pair.run(&this->atom->x, &this->atom->v, &this->atom->cv,
                         &this->atom->e, &this->atom->rho, &this->atom->de,
                         &this->atom->drho, &this->atom->cuts, &_lj_types,
                         &this->nbor->dev_nbor, &this->_nbor_data->begin(),
                         &this->ans->force, &this->ans->engv, &eflag, &vflag,
                         &ainum, &nbor_pitch, &this->_threads_per_atom);
    }
    this->time_pair.stop();
}

template class LJ_SPH<PRECISION,ACC_PRECISION>;
}