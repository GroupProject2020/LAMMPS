/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#if defined(USE_OPENCL)
#include "sph_lj_cl.h"
#elif defined(USE_CUDART)
const char *lj=0;
#else
#include "sph_lj_cubin.h"
#endif

#include "lal_lj_sph.h"

#include <cassert>
namespace LAMMPS_AL {
#define LJ_SPHT LJ_SPH<numtyp, acctyp>

extern Device<PRECISION,ACC_PRECISION> device;

template <class numtyp, class acctyp>
LJ_SPHT::LJ_SPH() : BaseAtomic<numtyp,acctyp>(), _allocated(false) {
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
              const double gpu_split, FILE *screen, int domainDim) {
    int success;
    success=this->init_atomic(nlocal,nall,max_nbors,maxspecial,cell_size,gpu_split,
                              screen,lj_sph,"k_lj_sph");
    if (success!=0)
        return success;
    this->k_pair.set_function(*(this->pair_program),"k_lj_sph");
    this->k_pair_fast.set_function(*(this->pair_program),"k_lj_sph_fast");

    this->domainDim = domainDim;
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

    cv.alloc(nlocal, *(this->ucl_device),UCL_READ_ONLY);
    cv_tex.get_texture(*(this->pair_program), "cv_tex");
    cv_tex.bind_float(cv,1);

    vel_tex.get_texture(*(this->pair_program), "vel_tex");
    vel_tex.bind_float(atom->v,4);

    rho.alloc(nlocal, *(this->ucl_device),UCL_READ_ONLY);
    rho_tex.get_texture(*(this->pair_program), "rho_tex");
    rho_tex.bind_float(rho,1);

    e.alloc(nlocal, *(this->ucl_device),UCL_READ_ONLY);
    e_tex.get_texture(*(this->pair_program), "e_tex");
    e_tex.bind_float(e,1);

    de.alloc(nlocal, *(this->ucl_device),UCL_READ_ONLY);
    de_tex.get_texture(*(this->pair_program), "de_tex");
    de_tex.bind_float(de,1);

    drho.alloc(nlocal, *(this->ucl_device),UCL_READ_ONLY);
    drho_tex.get_texture(*(this->pair_program), "drho_tex");
    drho_tex.bind_float(drho,1);

    _allocated=true;
    this->_max_bytes=cuts.row_bytes();
    return 0;
}

template <class numtyp, class acctyp>
void LJ_SPHT::clear() {
    if (!_allocated)
        return;
    _allocated=false;

    cuts.clear();
    cv.clear();
    e.clear();
    rho.clear();
    de.clear();
    drho.clear();

    this->clear_atomic();
}

template <class numtyp, class acctyp>
double LJ_SPHT::host_memory_usage() const {
    return this->host_memory_usage_atomic()+sizeof(LJ_SPH<numtyp,acctyp>);
}

template <class numtyp, class acctyp>
void LJ_SPHT::compute(const int f_ago, const int inum_full, const int nall,
                      double **host_x, double **host_v, double *host_cv,
                      double *host_e, double *host_rho, double *host_de,
                      double *host_drho, int *host_type, int *ilist, int *numj,
                      int **firstneigh, const bool eflag, const bool vflag,
                      const bool eatom, const bool vatom, int &host_start,
                      const double cpu_time, bool &success, tagint* tag){
    this->acc_timers();
    if (inum_full==0) {
        host_start=0;
        // Make sure textures are correct if realloc by a different hybrid style
        this->resize_atom(0,nall,success);
        this->zero_timers();
        return;
    }

    int ago=this->hd_balancer.ago_first(f_ago);
    int inum=this->hd_balancer.balance(ago,inum_full,cpu_time);
    this->ans->inum(inum);
    host_start=inum;

    if (ago==0) {
        this->reset_nbors(nall, inum, ilist, numj, firstneigh, success);
        if (!success)
            return;
    }

    this->atom->cast_x_data(host_x,host_type);
    this->atom->cast_v_data(host_v, tag);
    this->cast_cv_data(host_cv);
    this->cast_e_data(host_e);
    this->cast_rho_data(host_rho);
    this->cast_de_data(host_de);
    this->cast_drho_data(host_drho);

    this->hd_balancer.start_timer();
    this->atom->add_x_data(host_x,host_type);
    this->atom->add_v_data(host_v,tag);
    this->add_cv_data();
    this->add_e_data();
    this->add_rho_data();
    this->add_de_data();
    this->add_drho_data();

    this->loop(eflag,vflag);
    this->ans->copy_answers(eflag,vflag,eatom,vatom,ilist);
    this->device->add_ans_object(this->ans);
    this->hd_balancer.stop_timer();
}

template <class numtyp, class acctyp>
int ** LJ_SPHT::compute(const int ago, const int inum_full,
                   const int nall, double **host_x, double **host_v,
                   double *host_cv, double *host_e, double *host_rho,
                   double *host_de, double *host_drho, int *host_type,
                   double *sublo, double *subhi, tagint *tag, int **nspecial,
                   tagint **special, const bool eflag, const bool vflag,
                   const bool eatom, const bool vatom, int &host_start,
                   int **ilist, int **jnum, const double cpu_time,
                   bool &success){
    this->acc_timers();
    if (inum_full==0) {
        host_start=0;
        // Make sure textures are correct if realloc by a different hybrid style
        this->resize_atom(0,nall,success);
        this->zero_timers();
        return NULL;
    }

    this->hd_balancer.balance(cpu_time);
    int inum=this->hd_balancer.get_gpu_count(ago,inum_full);
    this->ans->inum(inum);
    host_start=inum;

    // Build neighbor list on GPU if necessary
    if (ago==0) {
        this->build_nbor_list(inum, inum_full-inum, nall, host_x, host_type,
                        sublo, subhi, tag, nspecial, special, success);
        if (!success)
            return NULL;
        this->atom->cast_v_data(host_v,tag);
        this->cast_cv_data(host_cv);
        this->cast_e_data(host_e);
        this->cast_rho_data(host_rho);
        this->cast_de_data(host_de);
        this->cast_drho_data(host_drho);
        this->hd_balancer.start_timer();
    } else {
        this->atom->cast_x_data(host_x,host_type);
        this->atom->cast_v_data(host_v,tag);
        this->hd_balancer.start_timer();
        this->atom->add_x_data(host_x,host_type);
    }
    this->add_cv_data();
    this->add_e_data();
    this->add_rho_data();
    this->add_de_data();
    this->add_drho_data();
    *ilist=this->nbor->host_ilist.begin();
    *jnum=this->nbor->host_acc.begin();

    this->loop(eflag,vflag);
    this->ans->copy_answers(eflag,vflag,eatom,vatom);
    this->device->add_ans_object(this->ans);
    this->hd_balancer.stop_timer();

    return this->nbor->host_jlist.begin()-host_start;
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
        this->k_pair_fast.run(&this->atom->x, &this->atom->v, &cv,
                              &e, &rho, &de,
                              &drho, &this->cuts,
                              &this->nbor->dev_nbor, &this->_nbor_data->begin(),
                              &this->ans->force, &this->ans->engv, &eflag, &vflag,
                              &ainum, &nbor_pitch, &this->_threads_per_atom, &this->domainDim);
    } else {
        this->k_pair.set_size(GX,BX);
        this->k_pair.run(&this->atom->x, &this->atom->v, &cv,
                         &e, &rho, &de,
                         &drho, &this->cuts, &_lj_types,
                         &this->nbor->dev_nbor, &this->_nbor_data->begin(),
                         &this->ans->force, &this->ans->engv, &eflag, &vflag,
                         &ainum, &nbor_pitch, &this->_threads_per_atom, &this->domainDim);
    }
    this->time_pair.stop();
}

template class LJ_SPH<PRECISION,ACC_PRECISION>;
}
