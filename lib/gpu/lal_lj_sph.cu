#ifdef NV_KERNEL
#include "lal_aux_fun1.h"
#ifndef _DOUBLE_DOUBLE
texture<float4> pos_tex;
texture<float4> vel_tex;
texture<float> cv_tex;
texture<float> e_tex;
texture<float> rho_tex;
texture<float> de_tex;
texture<float> drho_tex;
#else
texture<int4,1> pos_tex;
texture<int4,1> vel_tex;
texture<int2> cv_tex;
texture<int2> e_tex;
texture<int2> rho_tex;
texture<int2> de_tex;
texture<int2> drho_tex;
#endif
#else
#define pos_tex x_
#define vel_tex v_
#define cv_tex cv_
#define e_tex e_
#define rho_tex rho_
#define de_tex de_
#define drho_tex drho_
#endif

__kernel void k_lj_sph(const __global numtyp4 *restrict x_,
                       const __global numtyp4 *restrict v_,
                       const __global numtyp *restrict cv_,
                       const __global numtyp *restrict e_,
                       const __global numtyp *restrict rho_,
                       const __global numtyp *restrict de_,
                       const __global numtyp *restrict drho_,
                       const __global numtyp4 *restrict cuts,
                       const int lj_types,
                       const __global int * dev_nbor,
                       const __global int * dev_packed,
                       __global acctyp4 *restrict ans,
                       __global acctyp *restrict engv,
                       const int eflag, const int vflag, const int inum,
                       const int nbor_pitch,
                       const int t_per_atom){ //TODO: arguments?
  int tid, ii, offset;
  atom_info(t_per_atom,ii,tid,offset);
  double h, ih, ihsq, ihcub, wfd, fi, ci, fj, cj;
  double delVdotDelR, mu, fvisc, fpair, deltaE, imass, jmass;

  acctyp energy=(acctyp)0;
  acctyp4 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp virial[6];
  for (int i=0; i<6; i++)
    virial[i]=(acctyp)0;

  if (ii<inum) {
    int i, numj, nbor, nbor_end;
    __local int n_stride;
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,
              n_stride,nbor_end,nbor);

    numtyp4 ix; fetch4(ix,i,pos_tex); //x_[i];
    numtyp4 iv; fetch4(iv,i,vel_tex); //v_[i];
    double cvi; fetch(cvi, i, cv_tex);
    double ei; fetch(ei, i, cv_tex);
    double rhoi; fetch(rhoi, i, cv_tex);

    LJEOS2(rhoi,ei,cvi, &fi; &ci);
    fi /= (rhoi * rhoi);

    int itype=ix.w;
    imass = cuts[itype].z;
    for ( ; nbor<nbor_end; nbor+=n_stride) {

      int j=dev_packed[nbor];
      j &= NEIGHMASK;

      numtyp4 jx; fetch4(jx,j,pos_tex); //x_[j];
      numtyp4 jv; fetch4(jv,j,vel_tex); //v_[j];
      int jtype=jx.w;


      // Compute r12
      numtyp delx = ix.x-jx.x;
      numtyp dely = ix.y-jx.y;
      numtyp delz = ix.z-jx.z;
      numtyp rsq = delx*delx+dely*dely+delz*delz;
      jmass = cuts[jtype].z;

      int mtype=itype*lj_types+jtype;
      if (rsq<cuts[mtype].z) {
        h = cuts[mtype].y;
        ih = 1.0/h;
        ihsq = ih * ih;
        ihcub = ihsq * ih;

        wfd = h - sqrt(rsq);

      if (domainDim == 3) {
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

        double cvj; fetch(cvj, j, cv_tex);
        double ej; fetch(ej, j, cv_tex);
        double rhoj; fetch(rhoj, j, cv_tex);

        // function call to LJ EOS
        LJEOS2(rhoj, ej, cvj, &fj, &cj);
        fj /= (rhoj * rhoj);

        // apply long-range correction to model a LJ fluid with cutoff
        // this implies that the modelled LJ fluid has cutoff == SPH cutoff
        lrc = - 11.1701 * (ihcub * ihcub * ihcub - 1.5 * ihcub);
        fi += lrc;
        fj += lrc;


        // dot product of velocity delta and distance vector
        delVdotDelR = delx * (iv.x - jv.x) + dely * (iv.y -jv.y)
            + delz * (iv.z - jv.z);

        // artificial viscosity (Monaghan 1992)
        if (delVdotDelR < 0.) {
          mu = h * delVdotDelR / (rsq + 0.01 * h * h);
          fvisc = -0.04 * (ci + cj) * mu / (rhoi + rhoj); //TODO: implement the viscosity
        } else {
          fvisc = 0.;
        }

        // total pair force & thermal energy increment
        fpair = -imass * jmass * (fi + fj + fvisc) * wfd;
        deltaE = -0.5 * fpair * delVdotDelR;

        f.x+=delx*force;
        f.y+=dely*force;
        f.z+=delz*force;

        double dei; fetch(dei, i, dcv_tex);
        double drhoi; fetch(drhoi, i, dcv_tex);

        // and change in density
        drhoi += jmass * delVdotDelR * wfd;

        // change in thermal energy
        dei += deltaE;

        if (vflag>0) {
          virial[0] += delx*delx*force;
          virial[1] += dely*dely*force;
          virial[2] += delz*delz*force;
          virial[3] += delx*dely*force;
          virial[4] += delx*delz*force;
          virial[5] += dely*delz*force;
        }
      }

    } // for nbor TODO: write an adapted store_answer function
    store_answers(f,energy,virial,ii,inum,tid,t_per_atom,offset,eflag,vflag,
                  ans,engv);
  } // if ii
}

__kernel void k_lj_sph_fast(const __global numtyp4 *restrict x_,
                       const __global numtyp4 *restrict v_,
                       const __global numtyp *restrict cv_,
                       const __global numtyp *restrict e_,
                       const __global numtyp *restrict rho_,
                       const __global numtyp *restrict de_,
                       const __global numtyp *restrict drho_,
                       const __global numtyp4 *restrict cuts_in,
                       const __global int * dev_nbor,
                       const __global int * dev_packed,
                       __global acctyp4 *restrict ans,
                       __global acctyp *restrict engv,
                       const int eflag, const int vflag, const int inum,
                       const int nbor_pitch,
                       const int t_per_atom){ //TODO: arguments?
  int tid, ii, offset;
  atom_info(t_per_atom,ii,tid,offset);
  double h, ih, ihsq, ihcub, wfd, fi, ci, fj, cj;
  double delVdotDelR, mu, fvisc, fpair, deltaE, imass, jmass;

  __local numtyp4 cuts[MAX_SHARED_TYPES*MAX_SHARED_TYPES];
  if (tid<MAX_SHARED_TYPES*MAX_SHARED_TYPES) {
    cuts[tid]=cuts_in[tid];
  }

  acctyp energy=(acctyp)0;
  acctyp4 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp virial[6];
  for (int i=0; i<6; i++)
    virial[i]=(acctyp)0;

  __syncthreads();

  if (ii<inum) {
    int i, numj, nbor, nbor_end;
    __local int n_stride;
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,
              n_stride,nbor_end,nbor);

    numtyp4 ix; fetch4(ix,i,pos_tex); //x_[i];
    numtyp4 iv; fetch4(iv,i,vel_tex); //v_[i];
    double cvi; fetch(cvi, i, cv_tex);
    double ei; fetch(ei, i, cv_tex);
    double rhoi; fetch(rhoi, i, cv_tex);

    LJEOS2(rhoi,ei,cvi, &fi; &ci);
    fi /= (rhoi * rhoi);
    int iw=ix.w;
    int itype=fast_mul((int)MAX_SHARED_TYPES,iw);

    imass = cuts[itype].z;
    for ( ; nbor<nbor_end; nbor+=n_stride) {

      int j=dev_packed[nbor];
      j &= NEIGHMASK;

      numtyp4 jx; fetch4(jx,j,pos_tex); //x_[j];
      numtyp4 jv; fetch4(jv,j,vel_tex); //v_[j];
      int jtype=jx.w;


      // Compute r12
      numtyp delx = ix.x-jx.x;
      numtyp dely = ix.y-jx.y;
      numtyp delz = ix.z-jx.z;
      numtyp rsq = delx*delx+dely*dely+delz*delz;
      jmass = cuts[jtype].z;

      int mtype=itype+jtype;
      if (rsq<cuts[mtype].z) {
        h = cuts[mtype].y;
        ih = 1.0/h;
        ihsq = ih * ih;
        ihcub = ihsq * ih;

        wfd = h - sqrt(rsq);

      if (domainDim == 3) {
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

        double cvj; fetch(cvj, j, cv_tex);
        double ej; fetch(ej, j, cv_tex);
        double rhoj; fetch(rhoj, j, cv_tex);

        // function call to LJ EOS
        LJEOS2(rhoj, ej, cvj, &fj, &cj);
        fj /= (rhoj * rhoj);

        // apply long-range correction to model a LJ fluid with cutoff
        // this implies that the modelled LJ fluid has cutoff == SPH cutoff
        lrc = - 11.1701 * (ihcub * ihcub * ihcub - 1.5 * ihcub);
        fi += lrc;
        fj += lrc;


        // dot product of velocity delta and distance vector
        delVdotDelR = delx * (iv.x - jv.x) + dely * (iv.y -jv.y)
            + delz * (iv.z - jv.z);

        // artificial viscosity (Monaghan 1992)
        if (delVdotDelR < 0.) {
          mu = h * delVdotDelR / (rsq + 0.01 * h * h);
          fvisc = -0.04 * (ci + cj) * mu / (rhoi + rhoj); //TODO: implement the viscosity
        } else {
          fvisc = 0.;
        }

        // total pair force & thermal energy increment
        fpair = -imass * jmass * (fi + fj + fvisc) * wfd;
        deltaE = -0.5 * fpair * delVdotDelR;

        f.x+=delx*force;
        f.y+=dely*force;
        f.z+=delz*force;

        double dei; //fetch(dei, i, dcv_tex);
        double drhoi; //fetch(drhoi, i, dcv_tex);

        // and change in density
        drho_[i] += jmass * delVdotDelR * wfd;

        // change in thermal energy
        de_[i] += deltaE;

        if (vflag>0) {
          virial[0] += delx*delx*force;
          virial[1] += dely*dely*force;
          virial[2] += delz*delz*force;
          virial[3] += delx*dely*force;
          virial[4] += delx*delz*force;
          virial[5] += dely*delz*force;
        }
      }

    } // for nbor TODO: write an adapted store_answer function
    store_answers(f,energy,virial,ii,inum,tid,t_per_atom,offset,eflag,vflag,
                  ans,engv);
  } // if ii
}
