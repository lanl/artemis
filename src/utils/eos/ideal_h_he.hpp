//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
#ifndef UTILS_EOS_IDEAL_H_HE_HPP_
#define UTILS_EOS_IDEAL_H_HE_HPP_

#include <cstdio>
#ifdef SINGULARITY_USE_SPINER_WITH_HDF5
#include <hdf5.h>
#include <hdf5_hl.h>
#endif
#include <singularity-eos/base/robust_utils.hpp>
#include <singularity-eos/base/root-finding-1d/root_finding.hpp>
#include <singularity-eos/base/spiner_table_utils.hpp>
#include <singularity-eos/eos/eos.hpp>

namespace ArtemisEOS {

struct Mixture {
  Real x, y, z1, z2;
  Real dxdt, dydt, dz1dt, dz2dt;
  Real dxdr, dydr, dz1dr, dz2dr;
};
struct H2Partition {
  Real Z, dZ, d2Z;
};

class IdealHHe : public singularity::eos_base::EosBase<IdealHHe> {
  friend class singularity::table_utils::SpinerTricks<IdealHHe>;

 public:
  using DataBox = Spiner::DataBox<Real>;
  IdealHHe() = default;
  IdealHHe(Real X, Real Y, Real ltmin, Real ltmax, int nt, Real ldmin, Real ldmax, int nd,
           const std::string &save_to_file, bool use_table = true, Real dlnT = 1e-6,
           const singularity::MeanAtomicProperties &AZbar =
               singularity::MeanAtomicProperties())
      : _X(X), _Y(Y), lTmin(ltmin), lTmax(ltmax), nt(nt), lDmin(ldmin), lDmax(ldmax),
        nd(nd), use_table(use_table), _dlnT(dlnT), _AZbar(AZbar) {
    _fp = 0.25;
    _fo = 1. - _fp;
    CheckParams();
    FillTable(save_to_file);
  }
  IdealHHe(const std::string &filename) {
    // Load from file
    Load(filename);
  }
  inline void FillTable(const std::string &filename);
  inline void Save(const std::string &filename);
  inline void Load(const std::string &filename);
  IdealHHe GetOnDevice();

  PORTABLE_INLINE_FUNCTION void CheckParams() const {
    PORTABLE_ALWAYS_REQUIRE(_X >= 0, "X must be positive");
    PORTABLE_ALWAYS_REQUIRE(_Y >= 0, "Y must be positive");
    _AZbar.CheckParams();
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real TemperatureFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {

    const Real T = TofRE(rho, sie, lambda);
    return T;
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real InternalEnergyFromDensityTemperature(
      const Real rho, const Real temperature,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    const auto &r = GetMassFractions(rho, temperature);
    const Real kT = _eV * temperature;
    const Real henorm = 1. + r.z2 * (r.z1 - 1.0);
    Real EH = 1.5 * _X * (1. + r.x) * r.y;                        // H
    Real EHe = 0.375 * _Y * (1. + (r.z1 + r.z1 * r.z2) / henorm); // He
    Real EHpH = _ye * _X * r.y * 0.5;
    Real EHp = _xe * _X * r.x * r.y;
    Real EHep = _z1e * 0.25 * _Y * r.z1 / henorm;
    Real EHepp = _z2e * 0.25 * _Y * (r.z1 * r.z2 / henorm);

    // molecular hydrogen
    // NOTE: yes, the coefficients are in K
    const auto &[xp, xo] = H2PartitionFunctions(temperature);
    const Real dlno = singularity::robust::ratio(xo.dZ, xo.Z);
    const Real dlnp = singularity::robust::ratio(xp.dZ, xp.Z);
    const Real lnx = 6140. / temperature;
    const Real efac = singularity::robust::safe_arg_exp(-lnx);
    const Real EH2 = 0.5 * _X * (1. - r.y) *
                     (1.5 + lnx * singularity::robust::ratio(efac, 1. - efac) +
                      _fp * dlnp + _fo * (dlno - 2. * 85.5 / temperature));

    Real sie = EH2 + EH + EHe + (EHpH + EHp + EHep + EHepp) / temperature;
    sie = std::max(_small, sie * temperature * _kb / _mp);
    return sie;
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real PressureFromDensityTemperature(
      const Real rho, const Real temperature,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    const auto &[mu, dlmut, dlmur] = MeanMass(rho, temperature);
    const Real P = _kb / (mu * _mp) * rho * temperature;
    return std::max(_small, P);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real PressureFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    const Real ld = std::log10(rho);
    const Real lE = std::log10(sie);
    if (use_table && ((ld >= lDmin) && (ld <= lDmax) && (lE >= lEmin) && (lE <= lEmax))) {
      return std::pow(10., lP_.interpToReal(ld, lE));
    }
    // fall back to inline
    const Real T = TofRE(rho, sie);
    return PressureFromDensityTemperature(rho, T);
  }

  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real
  EntropyFromDensityTemperature(const Real rho, const Real temperature,
                                Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    // ? kb log(T)?
    // ln(P/rho^g) ?
    return 0.0;
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real EntropyFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    return 0.0;
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real SpecificHeatFromDensityTemperature(
      const Real rho, const Real temperature,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {

    // const auto &r = GetMassFractions(rho, temperature);
    // const Real kT = _eV * temperature;
    // const Real henorm = 1. + r.z2 * (r.z1 - 1.0);
    // const Real dhenorm = r.z2 * r.dz1dt + (r.z1 - 1.0) * r.dz2dt;

    // Real EH = 1.5 * _X * (1. + r.x) * r.y; // H
    // Real cH = EH + 1.5 * _X * ((1. + r.x) * r.dydt + r.y * r.dxdt);

    // Real EHe = 0.375 * _Y * (1. + (r.z1 + r.z1 * r.z2) / henorm); // He
    // Real cHe = EHe + 0.375 * _Y *
    //                      (((1. + r.z2) * r.dz1dt + r.z1 * r.dz2dt) / henorm -
    //                       r.z1 * (1. + r.z2) * dhenorm / (henorm * henorm));

    // Real EHpH = _ye * _X * r.y * 0.5;
    // Real cHpH = _ye * _X * r.dydt * 0.5;

    // Real EHp = _xe * _X * r.x * r.y;
    // Real cHp = _xe * _X * (r.x * r.dydt + r.y * r.dxdt);

    // Real EHep = _z1e * 0.25 * _Y * r.z1 / henorm;
    // Real cHep =
    //     _z1e * 0.25 * _Y * (r.dz1dt / henorm - r.z1 * dhenorm / (henorm * henorm));

    // Real EHepp = _z2e * 0.25 * _Y * (r.z1 * r.z2 / henorm);
    // Real cHepp = _z2e * 0.25 * _Y *
    //              ((r.z1 * r.dz2dt + r.z2 * r.dz1dt) / henorm -
    //               r.z1 * r.z2 * dhenorm / (henorm * henorm));

    // // molecular hydrogen
    // // NOTE: yes, the coefficients are in K
    // constexpr Real tiny = std::numeric_limits<Real>::min();
    // const auto &[xp, xo] = H2PartitionFunctions(temperature);
    // const Real dlno = (xo.Z > tiny) ? singularity::robust::ratio(xo.dZ, xo.Z) : 0.0;
    // const Real dlnp = singularity::robust::ratio(xp.dZ, xp.Z);
    // const Real d2lno = (xo.Z > tiny) ? singularity::robust::ratio(xo.d2Z, xo.Z) : 0.0;
    // const Real d2lnp = singularity::robust::ratio(xp.d2Z, xp.Z);
    // const Real lnx = 6140. / temperature;
    // const Real xv = std::min(dlno * 0.5, 85.5 / temperature);
    // const Real efac = singularity::robust::safe_arg_exp(-lnx);
    // const Real evf = lnx * singularity::robust::ratio(efac, 1. - efac);
    // // d1= 0.25 * (58.4792 / 58.6465) + 0.997148 * (0.00855 - 2*0)
    // const Real d1 = _fp * dlnp + _fo * (dlno - 2. * xv);
    // Real d2 = _fp * d2lnp + _fo * d2lno - 4. * _fo * (_fo * dlno + _fp * dlnp) * xv -
    //           _fp * _fo * SQR(dlnp - dlno) + 4. * _fo * (1. + _fo * xv) * xv;

    // // d2 = 0.0;
    // Real EH2 = 0.5 * (1.5 + evf + _fp * dlnp + _fo * (dlno - 2. * xv));
    // Real cH2 = (1.5 + 2. * d1 + d2 - d1 * d1 +
    //             SQR(lnx) * singularity::robust::ratio(efac, SQR(1. - efac))) *
    //            0.5;
    // cH2 = _X * ((1. - r.y) * cH2 - EH2 * r.dydt);

    // // const Real EH2 = 0.5 * _X * (1. - r.y) *
    // //                  (1.5 + lnx * singularity::robust::ratio(efac, 1. - efac) +
    // //                   _fp * dlnp + _fo * (dlno - 2. * 85.5 / temperature));

    // EH2 *= _X * (1. - r.y);
    // Real cv = cH2 + cH + cHe + (cHpH + cHp + cHep + cHepp) / temperature;

    // cv = std::max(_small, cv * _kb / _mp);
    // return cv;

    Real tp = temperature * (1.0 + _dlnT);
    Real tm = temperature * (1.0 - _dlnT);

    Real ep = InternalEnergyFromDensityTemperature(rho, tp);
    Real em = InternalEnergyFromDensityTemperature(rho, tm);
    Real cv = singularity::robust::ratio(ep - em, 2 * _dlnT * temperature);
    return std::max(_small, cv);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real SpecificHeatFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    const Real ld = std::log10(rho);
    const Real lE = std::log10(sie);
    if (use_table && ((ld >= lDmin) && (ld <= lDmax) && (lE >= lEmin) && (lE <= lEmax))) {
      return Cv_.interpToReal(ld, lE);
    }
    // fall back to inline
    const Real T = TofRE(rho, sie);
    return SpecificHeatFromDensityTemperature(rho, T);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real BulkModulusFromDensityTemperature(
      const Real rho, const Real temperature,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    const Real P = PressureFromDensityTemperature(rho, temperature);
    const Real G1 = GruneisenParamFromDensityTemperature(rho, temperature);
    return std::max(_small, G1 * P);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real BulkModulusFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    const Real ld = std::log10(rho);
    const Real lE = std::log10(sie);
    if (use_table && ((ld >= lDmin) && (ld <= lDmax) && (lE >= lEmin) && (lE <= lEmax))) {
      return std::pow(10., lB_.interpToReal(ld, lE));
    }
    // fall back to inline
    const Real T = TofRE(rho, sie);
    return BulkModulusFromDensityTemperature(rho, T);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real GruneisenParamFromDensityTemperature(
      const Real rho, const Real temperature,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    // Gamma*rho  = (dP/dT) / Cv
    const Real Cv = SpecificHeatFromDensityTemperature(rho, temperature);
    const Real P = PressureFromDensityTemperature(rho, temperature);

    // log(T) - log(T*(1-eps))
    // -log(1-eps)
    // const Real tp = temperature * (1. + 0);
    // const Real tm = temperature * (1. - _dlnT);
    // const Real dp = rho * (1. + 0);
    // const Real dm = rho * (1. - _dlnT);
    // const Real dlmudt =
    //     std::log(MeanMass(rho, tp) / MeanMass(rho, tm)) / std::log(tp / tm);
    // const Real dlmudr = std::log(MeanMass(dp, temperature) / MeanMass(dm, temperature))
    // /
    //                     std::log(dp / dm);
    // const Real dlpt = std::log(PressureFromDensityTemperature(rho, tp) /
    //                            PressureFromDensityTemperature(rho, tm)) /
    //                   std::log(tp / tm);
    // const Real dlpr = std::log(PressureFromDensityTemperature(dp, temperature) /
    //                            PressureFromDensityTemperature(dm, temperature)) /
    //                   std::log(dp / dm);
    const auto &[mu, dlmudt, dlmudr] = MeanMass(rho, temperature);
    // dln(P)/dln(T)
    // dln(P)/dln(rho)
    const Real xt = std::max(0.0, 1. - dlmudt);
    const Real xd = std::max(0.0, 1. - dlmudr);
    const Real denom = Cv * rho * temperature;
    const Real fac1 = singularity::robust::ratio(P, Cv * rho * temperature);
    const Real G1 = fac1 * xt * xt + xd;
    return singularity::robust::ratio(P, Cv * rho * temperature) * xt * xt + xd;
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real GruneisenParamFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    const Real ld = std::log10(rho);
    const Real lE = std::log10(sie);
    if (use_table && ((ld >= lDmin) && (ld <= lDmax) && (lE >= lEmin) && (lE <= lEmax))) {
      return Gm_.interpToReal(ld, lE);
    }
    // fall back to inline
    const Real T = TofRE(rho, sie);
    return GruneisenParamFromDensityTemperature(rho, T);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION void
  FillEos(Real &rho, Real &temp, Real &energy, Real &press, Real &cv, Real &bmod,
          const unsigned long output,
          Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION void
  ValuesAtReferenceState(Real &rho, Real &temp, Real &sie, Real &press, Real &cv,
                         Real &bmod, Real &dpde, Real &dvdt,
                         Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    // TODO
  }
  // Generic functions provided by the base class. These contain e.g. the vector
  // overloads that use the scalar versions declared here
  SG_ADD_BASE_CLASS_USINGS(IdealHHe)
  SG_ADD_DEFAULT_MEAN_ATOMIC_FUNCTIONS(_AZbar)

  static constexpr unsigned long PreferredInput() { return _preferred_input; }
  PORTABLE_INLINE_FUNCTION void PrintParams() const {
    printf("Ideal H-He Parameters:\nX = %g\nY    = %g\n", _X, _Y);
    _AZbar.PrintParams();
  }
  //   template <typename Indexer_t>
  //   DensityEnergyFromPressureTemperature(const Real press, const Real temp,
  //                                        Indexer_t &&lambda, Real &rho, Real &sie)
  //                                        const {
  //     sie = std::max(
  //         _qq, singularity::robust::ratio(press + (_gm1 + 1.0) * _Pinf, press + _Pinf)
  //         *
  //                      _Cv * temp +
  //                  _qq);
  //     rho = std::max(singularity::robust::SMALL(),
  //                    singularity::robust::ratio(press + _Pinf, _gm1 * _Cv * temp));
  //   }
  inline void Finalize();
  static std::string EosType() { return std::string("IdealHHe"); }
  static std::string EosPyType() { return EosType(); }
  std::size_t DynamicMemorySizeInBytes() const;
  std::size_t DumpDynamicMemory(char *dst);
  std::size_t SetDynamicMemory(char *src, const singularity::SharedMemSettings &stngs =
                                              singularity::DEFAULT_SHMEM_STNGS);

 private:
  bool use_table;
  Real _X, _Y, _fp, _fo;
  Real lTmin, lTmax, lDmin, lDmax, _dlnT, lEmin, lEmax;
  int nd, nt;
  DataBox lP_, lB_, lT_, Cv_, Gm_;
  Real _small = 1e-15;
  Real _na = 6.02214129e23;
  Real _hbar = 1.0546e-27; // cm^2 g/s
  Real _kb = 1.3807e-16;   // cm^2 g/(s^2 K)
  Real _eV = 8.6173e-5;    // eV/K
  Real _me = 9.1094e-28;   // g
  Real _mp = 1.6726e-24;   // g
  Real _Tp = 4.0 * M_PI * _hbar * _hbar / (_mp * _kb);
  Real _Te = 2.0 * M_PI * _hbar * _hbar / (_me * _kb);
  Real _ye = 4.478069 / _eV;
  Real _xe = 13.598433 / _eV;
  Real _z1e = 24.587387 / _eV;
  Real _z2e = 54.417760 / _eV;

  singularity::MeanAtomicProperties _AZbar;
  static constexpr const unsigned long _preferred_input =
      singularity::thermalqs::density | singularity::thermalqs::temperature;
#define DBLIST &lP_, &lB_, &lT_, &Cv_, &Gm_
  auto GetDataBoxPointers_() const { return std::vector<const DataBox *>{DBLIST}; }
  auto GetDataBoxPointers_() { return std::vector<DataBox *>{DBLIST}; }
#undef DBLIST

  singularity::DataStatus memoryStatus_ = singularity::DataStatus::Deallocated;
  // Internal functions

  PORTABLE_INLINE_FUNCTION
  Real root_solve(const Real x, const Real a, const Real b, const Real c) const {
    // Newton-Raphson on the polynomial (b + c*y)*y - a*(1-y)
    constexpr int ITER_MAX = 100;
    constexpr Real tol = 1e-15;
    // return x;
    Real f = (a + b + c * x) * x - a;
    if (std::abs(f) <= tol) return x;

    int iter = 0;
    Real xk = x;
    do {
      Real df = b + a + 2. * c * xk;
      xk -= f / df;
      f = (a + b + c * xk) * xk - a;
      iter++;
    } while ((std::abs(f) > tol) && (iter < ITER_MAX));
    return xk;
  }
  PORTABLE_INLINE_FUNCTION
  Real quadratic_root(const Real a, const Real b, const Real c) const {
    // Solving quadratic equations of the form
    //   (b + c*y)*y = (1 - y) * a == 0
    // solution is  (-(a+b) + sqrt( (a+b)^2 + 4*a*c)) /(2*c)
    // = c*y^2 + (b + a)*y - a == 0
    // = c/a y^2 + (b/a + 1) * y - 1 == 0
    // const Real disc = 1.0 + (2 * (b + 2.0 * c) / a + (b / a) * (b / a));
    // return singularity::robust::ratio(2.0, 1.0 + (b / a + std::sqrt(disc)));
    const Real a_ = std::abs(a);
    if (a_ <= _small) {
      // Treat as 0 and pick the zero root
      // Usually this means T is so low that nothing is ionized
      return 0.0;
    }
    // a is large
    if (a_ > 1e8) {
      // use the small y^2 coefficient formula
      const Real disc = 1.0 + (2 * (b + 2.0 * c) / a + (b / a) * (b / a));
      return root_solve(singularity::robust::ratio(2.0, 1.0 + (b / a + std::sqrt(disc))),
                        a, b, c);
    }
    // a is small, use the normal formula
    const Real bp = b + a;
    return root_solve(
        singularity::robust::ratio((-bp + std::sqrt(bp * bp + 4. * a * c)), 2 * c), a, b,
        c);
  }
  PORTABLE_INLINE_FUNCTION Mixture GetMassFractions(const Real rho, const Real T) const {
    Mixture res{0.0};
    // y
    Real f1 = _mp / rho;
    const Real ppfac = _mp * std::pow(_Tp, -1.5);
    Real f2p = std::pow(T / _Tp, 1.5);
    Real f2e = std::pow(T / _Te, 1.5);
    const Real kT = _eV * T;
    auto snap = [](Real x) {
      //   if (std::abs(x - 1.0) <= 1e-10) return 1.0;
      //   if (std::abs(x) <= 1e-10) return 0.0;
      return x;
    };
    // x^2 = (1-x)*a
    // x
    Real a = f1 / _X * f2e * singularity::robust::safe_arg_exp(-_xe / (T));
    Real dlat = _xe / (T) + 1.5;
    res.x = snap(quadratic_root(a, 0., 1.0));
    if ((std::abs(a) > _small) && (res.x > 0.0) && (res.x < 1.0)) {
      res.dxdt = singularity::robust::ratio(dlat * a * (1. - res.x), 2 * res.x + a);
      res.dxdr = singularity::robust::ratio(-a * (1. - res.x), (2 * res.x + a));
    }

    // y^2 = (1- y)*
    Real efac_ = singularity::robust::safe_arg_exp(-_ye / (T));
    Real fac1_ = f1 * f2p;
    a = 0.5 * f1 / _X * f2p * singularity::robust::safe_arg_exp(-_ye / (T));
    res.y = snap(quadratic_root(a, 0.0, 1.0));
    dlat = _ye / (T) + 1.5;
    if ((std::abs(a) > _small) || (res.y > 0.0) && (res.y < 1.0)) {
      res.dydt = singularity::robust::ratio(dlat * a * (1. - res.y), 2 * res.y + a);
      res.dydr = singularity::robust::ratio(-a * (1. - res.y), (2 * res.y + a));
    }

    // (X + 0.25*Y * z)*z = (1-z)*a
    a = 4.0 * f1 * f2e * singularity::robust::safe_arg_exp(-_z1e / (T));
    dlat = _z1e / (T) + 1.5;
    res.z1 = snap(quadratic_root(a, _X, 0.25 * _Y));
    if ((std::abs(a) > _small) && (res.z1 > 0.0) && (res.z1 < 1.0)) {
      res.dz1dt = singularity::robust::ratio(dlat * a * (1. - res.z1),
                                             a + 0.5 * _Y * res.z1 + _X);
      res.dz1dr =
          singularity::robust::ratio(-a * (1. - res.z1), (a + 0.5 * _Y * res.z1 + _X));
    }

    // (X + 0.25*Y  + 0.25*Y * z)*z = (1-z)*a
    a = f1 * f2e * singularity::robust::safe_arg_exp(-_z2e / (T));
    dlat = _z2e / (T) + 1.5;
    res.z2 = snap(quadratic_root(a, _X + 0.25 * _Y, 0.25 * _Y));
    if ((std::abs(a) > _small) && (res.z2 > 0.0) && (res.z2 < 1.0)) {
      res.dz2dt = singularity::robust::ratio(dlat * a * (1. - res.z2),
                                             a + 0.5 * _Y * res.z2 + _X + 0.25 * _Y);
      res.dz2dr = singularity::robust::ratio(-a * (1. - res.z2),
                                             (a + 0.5 * _Y * res.z2 + _X + 0.25 * _Y));
    }

    return res;
  }

  PORTABLE_INLINE_FUNCTION std::tuple<H2Partition, H2Partition>
  H2PartitionFunctions(const Real T) const {
    const Real x = 85.5 / T;

    // para
    H2Partition para{0.0};
    for (int j = 0; j < 1000; j += 2) {
      const int jj = j * (j + 1);
      const Real dr = (2 * j + 1) * singularity::robust::safe_arg_exp(-jj * x);
      para.Z += dr;
      if (j >= 2) {
        para.dZ += jj * dr;
        para.d2Z += jj * (jj * x - 2.0) * dr;
        if (singularity::robust::ratio(dr * std::abs(jj * (jj * x - 2.0)),
                                       std::abs(para.d2Z)) < 1e-12) {
          break;
        }
      }
    }
    para.dZ *= x;
    para.d2Z *= x;

    // ortho
    H2Partition ortho{0.0};
    for (int j = 1; j < 1000; j += 2) {
      const int jj = j * (j + 1);
      const Real dr = (2 * j + 1) * singularity::robust::safe_arg_exp(-jj * x);
      ortho.Z += dr;
      ortho.dZ += jj * dr;
      // if (jj * x > 2.0) {
      ortho.d2Z += jj * (jj * x - 2.0) * dr;
      /// }
      if (singularity::robust::ratio(dr * std::abs(jj * (jj * x - 2.0)),
                                     std::abs(ortho.d2Z)) < 3e-16) {
        break;
      }
    }
    ortho.dZ *= x;
    ortho.d2Z *= x;

    return {para, ortho};
  }
  PORTABLE_INLINE_FUNCTION std::tuple<Real, Real, Real> MeanMass(const Real rho,
                                                                 const Real T) const {
    auto get_mu = [](Real X, Real Y, Mixture r) {
      return singularity::robust::ratio(
          1.0,
          0.25 * (2. * X * (1. + r.y * (1. + 2. * r.x)) + Y * (1. + r.z1 * (1. + r.z2))));
    };
    const auto &r = GetMassFractions(rho, T);
    Real mu = get_mu(_X, _Y, r);

    const auto &rtp = GetMassFractions(rho, T * (1. + _dlnT));
    const auto &rtm = GetMassFractions(rho, T * (1. - _dlnT));
    const auto &rdp = GetMassFractions(rho * (1. + _dlnT), T);
    const auto &rdm = GetMassFractions(rho * (1. - _dlnT), T);
    Real dlmut = (get_mu(_X, _Y, rtp) - get_mu(_X, _Y, rtm)) / (2. * _dlnT * mu);
    Real dlmur = (get_mu(_X, _Y, rdp) - get_mu(_X, _Y, rdm)) / (2. * _dlnT * mu);

    // const auto &r = GetMassFractions(rho, T);
    // Real imu =
    //     0.25 * (2. * _X * (1. + r.y * (1. + 2. * r.x)) + _Y * (1. + r.z1 * (1. +
    //     r.z2)));
    // const Real mu = singularity::robust::ratio(1.0, imu);
    // const Real dlmut = -0.25 * mu *
    //                    (2 * _X * (r.dydt * (1.0 + 2 * r.x) + 2 * r.dxdt * r.y) +
    //                     _Y * (r.dz1dt * (1.0 + r.z2) + r.dz2dt * r.z1));
    // const Real dlmur = -0.25 * mu *
    //                    (2 * _X * (r.dydr * (1.0 + 2 * r.x) + 2 * r.dxdr * r.y) +
    //                     _Y * (r.dz1dr * (1.0 + r.z2) + r.dz2dr * r.z1));
    return {mu, dlmut, dlmur};
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real
  TofRE(const Real rho, const Real sie,
        Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    // Root finding version.

    // T_k+1 = T_k - (E(T) - E0)/cv(T)

    // Use ideal gas as guess
    // kb/(2.*mp) * T = E
    Real T = sie * _mp / _kb;
    int iter = 0;
    Real sie_new = 0.0;
    bool conv = false;
    Real lower = std::pow(10., lTmin) * .01;
    Real upper = std::pow(10., lTmax) * 100;
    Real E_low = InternalEnergyFromDensityTemperature(rho, lower);
    ;
    Real E_high = InternalEnergyFromDensityTemperature(rho, upper);
    Real dE_low = E_low - sie;
    Real dE_high = E_high - sie;

    if (dE_low * dE_high > 0.0) {
      printf("Initial sie %lg not bracketed at density %lg by temperature bounds [%lg, "
             "%lg], [%lg, %lg]\n",
             sie, rho, lower, upper, E_low, E_high);
    }

    for (iter = 0; iter < 100; iter++) {
      T = std::sqrt(lower * upper);
      sie_new = InternalEnergyFromDensityTemperature(rho, T);
      const Real dE = sie_new - sie;
      if (std::abs(dE) <= 1e-8 * sie) {
        conv = true;
        break;
      }
      if (dE * dE_low < 0.0) {
        upper = T;
        dE_high = dE;
      } else if (dE * dE_high < 0.0) {
        lower = T;
        dE_low = dE;
      } else {
        printf("Failed to converge %lg %lg %lg %lg\n", rho, sie, T,
               InternalEnergyFromDensityTemperature(rho, T, lambda));
        break;
      }
      //   sie_new = InternalEnergyFromDensityTemperature(rho, T, lambda);
      //   const Real dE = sie_new - sie;
      //   const Real Cv = SpecificHeatFromDensityTemperature(rho, T, lambda);
      //   Real dT = singularity::robust::ratio(-dE, Cv);
      //   if ((T + dT <= 0.0) || (std::abs(dT) > .5 * T)) {
      //     dT = ((dT > 0) ? 1 : -1) * 0.1 * T;
      //   }
      //   T += dT;
      //   if ((std::abs(dT) <= 1e-8 * T) && (std::abs(dE) <= 1e-8 * sie)) {
      //     conv = true;
      //     break;
      //   }
    }
    if (!conv) {
      printf("Failed to converge %lg %lg %lg %lg\n", rho, sie, T,
             InternalEnergyFromDensityTemperature(rho, T, lambda));
    }
    return std::max(_small, T);
  }
};

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION void
IdealHHe::FillEos(Real &rho, Real &temp, Real &sie, Real &press, Real &cv, Real &bmod,
                  const unsigned long output, Indexer_t &&lambda) const {
  if (output & singularity::thermalqs::density &&
      output & singularity::thermalqs::specific_internal_energy) {
    if (output & singularity::thermalqs::pressure ||
        output & singularity::thermalqs::temperature) {
      UNDEFINED_ERROR;
    }
    DensityEnergyFromPressureTemperature(press, temp, lambda, rho, sie);
  }
  if (output & singularity::thermalqs::pressure &&
      output & singularity::thermalqs::specific_internal_energy) {
    if (output & singularity::thermalqs::density ||
        output & singularity::thermalqs::temperature) {
      UNDEFINED_ERROR;
    }
    sie = InternalEnergyFromDensityTemperature(rho, temp, lambda);
  }
  if (output & singularity::thermalqs::temperature &&
      output & singularity::thermalqs::specific_internal_energy) {
    sie = InternalEnergyFromDensityTemperature(rho, temp, lambda);
  }
  if (output & singularity::thermalqs::pressure)
    press = PressureFromDensityInternalEnergy(rho, sie);
  if (output & singularity::thermalqs::temperature)
    temp = TemperatureFromDensityInternalEnergy(rho, sie);
  if (output & singularity::thermalqs::bulk_modulus)
    bmod = BulkModulusFromDensityInternalEnergy(rho, sie);
  if (output & singularity::thermalqs::specific_heat)
    cv = SpecificHeatFromDensityInternalEnergy(rho, sie);
}

inline void IdealHHe::FillTable(const std::string &filename) {
  lT_.resize(nd, nt);
  lT_.setRange(0, lTmin, lTmax, nt);
  lT_.setRange(1, lDmin, lDmax, nd);
  // Determine the energy grid
  lEmin = std::numeric_limits<Real>::max();
  lEmax = std::numeric_limits<Real>::min();

  for (int j = 0; j < nd; j++) {
    const Real d = std::pow(10., lT_.range(1).x(j));
    for (int i = 0; i < nt; i++) {
      const Real T = std::pow(10., lT_.range(0).x(i));
      const auto &[mu, dlmut, dlmur] = MeanMass(d, T);
      const auto &r = GetMassFractions(d, T);
      const Real E = InternalEnergyFromDensityTemperature(d, T);
      const Real P = PressureFromDensityTemperature(d, T);
      const Real cv = SpecificHeatFromDensityTemperature(d, T);
      const Real G = GruneisenParamFromDensityTemperature(d, T);
      const Real B = BulkModulusFromDensityTemperature(d, T);
      lEmin = std::min(lEmin, E);
      lEmax = std::max(lEmax, E);
    }
  }
  if (lEmin <= 0.0 || (lEmax <= 0.0) || std::isnan(lEmin) || std::isnan(lEmax)) {
    PORTABLE_THROW_OR_ABORT("Failed to find positive or real energy values from given "
                            "temperature and density grid.");
  }
  lEmin = std::log10(lEmin);
  lEmax = std::log10(lEmax);
  lT_.setRange(0, lEmin, lEmax, nt);
  lP_.copyMetadata(lT_);
  lB_.copyMetadata(lT_);
  Cv_.copyMetadata(lT_);
  Gm_.copyMetadata(lT_);
  // Fill table

  for (int j = 0; j < nd; j++) {
    const Real d = std::pow(10., lT_.range(1).x(j));
    for (int i = 0; i < nt; i++) {
      const Real e = std::pow(10., lT_.range(0).x(i));
      const Real T = TofRE(d, e);
      lT_(j, i) = std::log10(T);
      lP_(j, i) = std::log10(PressureFromDensityTemperature(d, T));
      lB_(j, i) = std::log10(BulkModulusFromDensityTemperature(d, T));
      Cv_(j, i) = SpecificHeatFromDensityTemperature(d, T);
      Gm_(j, i) = GruneisenParamFromDensityTemperature(d, T);
    }
  }

  // Checking table inversion
  for (int j = 0; j < nd; j++) {
    const Real ld = lT_.range(1).x(j);
    const Real d = std::pow(10., lT_.range(1).x(j));
    for (int i = 0; i < nt; i++) {
      const Real lE = lT_.range(0).x(i);
      const Real e = std::pow(10., lT_.range(0).x(i));
      const Real T = TofRE(d, e);
      assert(std::abs(std::pow(10., lT_.interpToReal(ld, lE)) / T - 1) <= 1e-4);
    }
  }

  //   // Save table

  if (filename != "") {
    if (Globals::my_rank == 0) {
      Save(filename);
    }
  }
}

constexpr char METADATA_NAME[] = "Params";
inline void IdealHHe::Save(const std::string &filename) {
  herr_t status = H5_SUCCESS;
  hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  hid_t metadata = H5Gcreate(file, METADATA_NAME, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status += H5LTset_attribute_double(file, METADATA_NAME, "x", &_X, 1);
  status += H5LTset_attribute_double(file, METADATA_NAME, "y", &_Y, 1);
  status += H5LTset_attribute_double(file, METADATA_NAME, "ltmin", &lTmin, 1);
  status += H5LTset_attribute_double(file, METADATA_NAME, "ltmax", &lTmax, 1);
  status += H5LTset_attribute_double(file, METADATA_NAME, "ldmin", &lDmin, 1);
  status += H5LTset_attribute_double(file, METADATA_NAME, "ldmax", &lDmax, 1);
  status += H5LTset_attribute_double(file, METADATA_NAME, "dlnT", &_dlnT, 1);
  status += H5LTset_attribute_double(file, METADATA_NAME, "fp", &_fp, 1);
  status += H5LTset_attribute_double(file, METADATA_NAME, "fm", &_fo, 1);
  status += H5LTset_attribute_int(file, METADATA_NAME, "nt", &nt, 1);
  status += H5LTset_attribute_int(file, METADATA_NAME, "nd", &nd, 1);
  H5Gclose(metadata);

  status += lP_.saveHDF(file, "logpress");
  status += lT_.saveHDF(file, "logtemp");
  status += Cv_.saveHDF(file, "cv");
  status += lB_.saveHDF(file, "logbulkmodulus");
  status += Gm_.saveHDF(file, "grun");

  status += H5Fclose(file);
  if (status != H5_SUCCESS) {
    EOS_ERROR("[IdealHHe::Save]: There was a problem with HDF5\n");
  }
}
inline void IdealHHe::Load(const std::string &filename) {
  herr_t status = H5_SUCCESS;
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  hid_t metadata = H5Gopen(file, METADATA_NAME, H5P_DEFAULT);
  status += H5LTget_attribute_double(file, METADATA_NAME, "x", &_X);
  status += H5LTget_attribute_double(file, METADATA_NAME, "y", &_Y);
  status += H5LTget_attribute_double(file, METADATA_NAME, "ltmin", &lTmin);
  status += H5LTget_attribute_double(file, METADATA_NAME, "ltmax", &lTmax);
  status += H5LTget_attribute_double(file, METADATA_NAME, "ldmin", &lDmin);
  status += H5LTget_attribute_double(file, METADATA_NAME, "ldmax", &lDmax);
  status += H5LTget_attribute_double(file, METADATA_NAME, "dlnT", &_dlnT);
  status += H5LTget_attribute_double(file, METADATA_NAME, "fp", &_fp);
  status += H5LTget_attribute_double(file, METADATA_NAME, "fm", &_fo);
  status += H5LTget_attribute_int(file, METADATA_NAME, "nt", &nt);
  status += H5LTget_attribute_int(file, METADATA_NAME, "nd", &nd);
  H5Gclose(metadata);

  status += lP_.loadHDF(file, "logpress");
  status += lT_.loadHDF(file, "logtemp");
  status += Cv_.loadHDF(file, "cv");
  status += lB_.loadHDF(file, "logbulkmodulus");
  status += Gm_.loadHDF(file, "grun");
  status += H5Fclose(file);
  if (status != H5_SUCCESS) {
    EOS_ERROR("[IdealHHe::Save]: There was a problem with HDF5\n");
  }
}

inline IdealHHe IdealHHe::GetOnDevice() {
  return singularity::table_utils::SpinerTricks<IdealHHe>::GetOnDevice(this);
}
inline void IdealHHe::Finalize() {
  return singularity::table_utils::SpinerTricks<IdealHHe>::Finalize(this);
}

} // namespace ArtemisEOS

#endif // UTILS_EOS_IDEAL_H_HE_HPP_
