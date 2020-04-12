
#ifndef LAL_LJ_SPH_H
#define LAL_LJ_SPH_H


#include "lal_base_atomic.h"

namespace LAMMPS_AL {

    template <class numtyp, class acctyp>
    class LJ_SPH : public BaseAtomic<numtyp, acctyp> {
    public:
        LJ_SPH();
        ~LJ_SPH();

        /// Clear any previous data and set up for a new LAMMPS run
        /** \param max_nbors initial number of rows in the neighbor matrix
          * \param cell_size cutoff + skin
          * \param gpu_split fraction of particles handled by device
          *
          * Returns:
          * -  0 if successful
          * - -1 if fix gpu not found
          * - -3 if there is an out of memory error
          * - -4 if the GPU library was not compiled for GPU
          * - -5 Double precision is not supported on card **/
        int init(const int ntypes, double **host_cutsq,
                 double **host_cut, double **host_mass,
                 const int nlocal, const int nall, const int max_nbors,
                 const int maxspecial, const double cell_size,
                 const double gpu_split, FILE *screen, int domainDim);

        /// Send updated coeffs from host to device (to be compatible with fix adapt)
        void reinit(const int ntypes, double **host_cutsq,
                    double **host_cut, double **host_mass);

        /// Clear all host and device data
        /** \note This is called at the beginning of the init() routine **/
        void clear();

        /// Returns memory usage on device per atom
        int bytes_per_atom(const int max_nbors) const;

        /// Total host memory used by library for pair style
        double host_memory_usage() const;

        /// Pair loop with host neighboring
        void compute(const int ago, const int inum_full, const int nall,
                             double **host_x, double **host_v, double **host_cv,
                             double **host_e, double **host_rho, double **host_de,
                             double **host_drho, int *host_type, int *ilist, int *numj,
                             int **firstneigh, const bool eflag, const bool vflag,
                             const bool eatom, const bool vatom, int &host_start,
                             const double cpu_time, bool &success, tagint* tag);

        /// Pair loop with device neighboring
        int ** compute(const int ago, const int inum_full,
                       const int nall, double **host_x, double **host_v,
                       double **host_cv, double **host_e, double **host_rho,
                       double **host_de, double **host_drho, int *host_type,
                       double *sublo, double *subhi, tagint *tag, int **nspecial,
                       tagint **special, const bool eflag, const bool vflag,
                       const bool eatom, const bool vatom, int &host_start,
                       int **ilist, int **jnum, const double cpu_time,
                       bool &success);
        // --------------------------- TEXTURES -----------------------------

        UCL_Texture cv_tex, e_tex, rho_tex, de_tex, drho_tex;
        // --------------------------- TYPE DATA --------------------------

        /// cuts.x = cutsq, cuts.y = cut, cuts.z = mass
        UCL_D_Vec<numtyp4> cuts;

        UCL_Vector<numtyp,numtyp> cv, e, rho, de, drho;

        /// If atom type constants fit in shared memory, use fast kernels
        bool shared_types;

        /// Number of atom types
        int _lj_types;

    private:
	int domainDim;
        bool _allocated;
        void loop(const bool _eflag, const bool _vflag);
    };

}

#endif //LAL_LJ_SPH_H
