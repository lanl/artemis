//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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

// Parthenon includes
#include <globals.hpp>
#include <parameter_input.hpp>

// Artemis includes
#include "artemis.hpp"
#include "nbody.hpp"
#include "utils/artemis_utils.hpp"

namespace NBody {
//----------------------------------------------------------------------------------------
//! \fn  SoftType NBody::ReturnSoft
//! \brief
SoftType ReturnSoft(std::string styp) {
  if (styp.compare("none") == 0) {
    return SoftType::plummer;
  } else if (styp.compare("plummer") == 0) {
    return SoftType::plummer;
  } else if (styp.compare("spline") == 0) {
    return SoftType::spline;
  } else {
    std::stringstream msg;
    msg << styp << " is not a valid softening type!";
    PARTHENON_FAIL(msg);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  std::vector<std::string> NBody::split
//! \brief
std::vector<std::string> split(const std::string &str, char delim) {
  std::vector<std::string> res;
  std::stringstream ss(str);
  std::string item;
  while (getline(ss, item, delim)) {
    res.push_back(item);
  }
  return res;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::PrintParticle
//! \brief
void PrintParticle(const int id, const ParticleParams &part) {
  std::cout << "===============\n"
            << "id: " << part.id << "\n"
            << "Mass: " << part.m << "\n"
            << "Radius: " << part.radius << "\n"
            << "Soft: " << part.rs << "\n"
            << "type: " << ((part.stype == SoftType::plummer) ? "plummer" : "spline")
            << "\n"
            << "Sink: " << part.racc << "\n"
            << "gamma: " << part.gamma << "\n"
            << "beta: " << part.beta << "\n"
            << "couple: " << part.couple << "\n"
            << "target_rad: " << part.target_rad << "\n"
            << "live: " << part.live << "\n"
            << "live_after: " << part.live_after << "\n"
            << "init: " << part.init << "\n"
            << "x=(" << part.x << "," << part.y << "," << part.z << ")\n"
            << "v=(" << part.vx << "," << part.vy << "," << part.vz << ")\n"
            << "===============\n"
            << std::endl;
  if (part.init == 0) {
    PARTHENON_WARN("This particle is not initialized!");
  }
}

//----------------------------------------------------------------------------------------
//! Versions of sin/cos meant to handle the edge cases wherein exactly 0.0 or 1.0 should
//! be returned (e.g., std::sin does not return exactly 0.0 for 180.0 degrees)
//! NOTE(ADM): these are not called from kernels
Real excos(const Real ang) {
  if ((ang == 0.0) || (ang == 2.0 * M_PI)) return 1.0;
  if (ang == M_PI) return -1.0;
  if ((2.0 * ang == M_PI) || (2.0 * ang == 3.0 * M_PI)) return 0.0;
  return std::cos(ang);
}
Real exsin(const Real ang) {
  if ((ang == 0.0) || (ang == 2.0 * M_PI) || (ang == M_PI)) return 0.0;
  if (2.0 * ang == M_PI) return 1.0;
  if (2.0 * ang == 3.0 * M_PI) return -1.0;
  return std::sin(ang);
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::init_orbit
//! \brief
void init_orbit(const Real m, const struct Orbit orb, Real *pos, Real *vel) {
  const Real a = orb.a;
  const Real e = orb.e;
  const Real I = orb.i;
  const Real f = orb.f;
  const Real om = orb.o;
  const Real Om = orb.O;
  Real sinf = exsin(f);
  Real cosf = excos(f);
  const Real n = std::sqrt(m / (a * a * a));
  const Real vb = a * n / std::sqrt(1 - SQR(e));
  const Real rb = a * (1 - e * e) / (1. + e * cosf);
  const Real xb = rb * cosf;
  const Real yb = rb * sinf;
  const Real vxb = -sinf * vb;
  const Real vyb = (cosf + e) * vb;

  // In lab frame
  // See eg Murray&Dermott Section 2.8
  const Real cosO = excos(Om);
  const Real sinO = exsin(Om);
  const Real cosI = excos(I);
  const Real sinI = exsin(I);
  const Real coso = excos(om);
  const Real sino = exsin(om);
  cosf = xb * coso - sino * yb;
  sinf = xb * sino + coso * yb;
  pos[0] = (cosO * cosf - sinO * sinf * cosI);
  pos[1] = (sinO * cosf + cosO * sinf * cosI);
  pos[2] = sinf * sinI;
  cosf = vxb * coso - sino * vyb;
  sinf = vxb * sino + coso * vyb;
  vel[0] = (cosO * cosf - sinO * sinf * cosI);
  vel[1] = (sinO * cosf + cosO * sinf * cosI);
  vel[2] = sinf * sinI;
}

//----------------------------------------------------------------------------------------
//! \fn  ParticleParams NBody::CreateNewParticle
//! \brief
ParticleParams CreateNewParticle(Real m, Real radius, Real rs, std::string stype,
                                 Real racc, Real gamma, Real beta, Real target_rad,
                                 int live, Real live_after) {
  ParticleParams part = {0};
  part.m = m;
  part.rs = rs;
  part.stype = ReturnSoft(stype);
  part.racc = racc;
  part.gamma = gamma;
  part.beta = beta;
  part.target_rad = target_rad;
  part.live = live;
  part.live_after = live_after;
  part.radius = radius;
  return part;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::ReadParticleBlock
//! \brief
void ReadParticleBlock(ParameterInput *pin, parthenon::InputBlock *pib,
                       ParticleParams &part) {
  std::vector<std::string> subs = split(pib->block_name, '/');
  if (subs.size() <= 2) {
    // <nbody/particle1>
    part.m = pin->GetReal(pib->block_name, "mass");
    part.radius = pin->GetOrAddReal(pib->block_name, "radius", 0.0);
    part.couple = pin->GetOrAddInteger(pib->block_name, "couple", 1);
    part.live = pin->GetOrAddInteger(pib->block_name, "live", 0);
    part.live_after = pin->GetOrAddReal(pib->block_name, "live_after", 0.0);
    part.target_rad = pin->GetOrAddReal(pib->block_name, "refine_distance", 0.0);
  } else {
    if (subs[2] == "soft") {
      // <nbody/particle1/soft>
      std::string ityp = pin->GetString(pib->block_name, "type");
      if (ityp == "none") {
        part.rs = 0.0;
        part.stype = SoftType::plummer;
      } else if (ityp == "plummer") {
        part.rs = pin->GetReal(pib->block_name, "radius");
        part.stype = SoftType::plummer;
      } else if (ityp == "spline") {
        part.rs = pin->GetReal(pib->block_name, "radius");
        part.stype = SoftType::spline;
      } else {
        std::stringstream msg;
        msg << "Unknown particle softening type " << ityp;
        PARTHENON_FAIL(msg);
      }
    } else if (subs[2] == "sink") {
      // <nbody/particle1/sink>
      part.racc = pin->GetReal(pib->block_name, "radius");
      part.gamma = pin->GetReal(pib->block_name, "gamma");
      part.beta = pin->GetOrAddReal(pib->block_name, "beta", 0.0);
    } else if (subs[2] == "initialize") {
      // <nbody/particle1/init>
      part.x = pin->GetOrAddReal(pib->block_name, "x", 0.0);
      part.y = pin->GetOrAddReal(pib->block_name, "y", 0.0);
      part.z = pin->GetOrAddReal(pib->block_name, "z", 0.0);
      part.vx = pin->GetOrAddReal(pib->block_name, "vx", 0.0);
      part.vy = pin->GetOrAddReal(pib->block_name, "vy", 0.0);
      part.vz = pin->GetOrAddReal(pib->block_name, "vz", 0.0);
      part.init = 1;
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  int NBody::ReadBinaryBlock
//! \brief
int ReadBinaryBlock(ParameterInput *pin, parthenon::InputBlock *pib,
                    std::map<int, ParticleParams> &parts) {
  int new_parts = 0;
  std::vector<std::string> subs = split(pib->block_name, '/');
  if (subs.size() <= 2) {
    // <nbody/binary1>
    Real mass = pin->GetOrAddReal(pib->block_name, "mass", -1.0);
    struct Orbit orb = {0};
    orb.a = pin->GetReal(pib->block_name, "a");
    orb.e = pin->GetOrAddReal(pib->block_name, "e", 0.0);
    orb.i = (pin->GetOrAddReal(pib->block_name, "i", 0.0) / 180.0) * M_PI;
    orb.o = (pin->GetOrAddReal(pib->block_name, "o", 0.0) / 180.0) * M_PI;
    orb.O = (pin->GetOrAddReal(pib->block_name, "O", 0.0) / 180.0) * M_PI;
    orb.f = (pin->GetOrAddReal(pib->block_name, "f", 180.0) / 180.0) * M_PI;
    Real Rb[3] = {Null<Real>()}, Vb[3] = {Null<Real>()};
    Rb[0] = pin->GetOrAddReal(pib->block_name, "x", 0.0);
    Rb[1] = pin->GetOrAddReal(pib->block_name, "y", 0.0);
    Rb[2] = pin->GetOrAddReal(pib->block_name, "z", 0.0);
    Vb[0] = pin->GetOrAddReal(pib->block_name, "vx", 0.0);
    Vb[1] = pin->GetOrAddReal(pib->block_name, "vy", 0.0);
    Vb[2] = pin->GetOrAddReal(pib->block_name, "vz", 0.0);

    // Particles designation
    int pp = pin->GetOrAddInteger(pib->block_name, "primary", -1);
    int ss = pin->GetOrAddInteger(pib->block_name, "secondary", -1);
    auto itp = parts.find(pp);
    auto its = parts.find(ss);

    if ((itp == parts.end()) || (its == parts.end()) || (pp == -1) || (ss == -1)) {
      const Real qb = pin->GetReal(pib->block_name, "q");
      const Real radius = pin->GetOrAddReal(pib->block_name, "radius", 0.0);
      const Real rs = pin->GetOrAddReal(pib->block_name, "rsoft", 0.0);
      const Real racc = pin->GetOrAddReal(pib->block_name, "rsink", 0.0);
      const Real gamma = pin->GetOrAddReal(pib->block_name, "gamma", 0.0);
      const Real beta = pin->GetOrAddReal(pib->block_name, "beta", 0.0);
      const Real target_rad = pin->GetOrAddReal(pib->block_name, "refine_distance", 0.0);
      const int live = pin->GetOrAddInteger(pib->block_name, "live", 0);
      const Real live_after = pin->GetOrAddReal(pib->block_name, "live_after", 0);
      const std::string stype = pin->GetOrAddString(pib->block_name, "stype", "spline");
      const int couple = pin->GetOrAddInteger(pib->block_name, "couple", 0);
      if (mass < 0.0) {
        std::stringstream msg;
        msg << "mass < 0 for " << pib->block_name << ". Please set the mass.";
        PARTHENON_FAIL(msg);
      }

      // Create new particles
      const Real m1 = mass / (1.0 + qb);
      const Real m2 = qb * m1;
      ParticleParams part1 = CreateNewParticle(m1, radius, rs, stype, racc, gamma, beta,
                                               target_rad, live, live_after);
      ParticleParams part2 = CreateNewParticle(m2, radius, rs, stype, racc, gamma, beta,
                                               target_rad, live, live_after);
      part1.couple = couple;
      part2.couple = couple;

      int maxid = 0;
      for (auto const &[id, p] : parts) {
        maxid = std::max(maxid, id);
      }
      if (pp == -1) {
        pp = maxid + 1;
        part1.id = pp;
        parts[pp] = part1;
        maxid++;
        new_parts++;
      }
      if (ss == -1) {
        ss = maxid + 1;
        part2.id = ss;
        parts[ss] = part2;
        new_parts++;
      }
    }

    // Correct masses
    auto &p = parts[pp];
    auto &s = parts[ss];
    const Real qb = s.m / p.m;
    if (mass > 0) {
      p.m = mass / (1. + qb);
      s.m = qb * p.m;
    } else {
      mass = s.m + p.m;
    }

    // Initialize positions and velocities
    Real rb[3] = {Null<Real>()}, vb[3] = {Null<Real>()};
    init_orbit(mass, orb, rb, vb);
    const Real mu1 = p.m / mass;
    const Real mu2 = s.m / mass;
    p.x = Rb[0] + -mu2 * rb[0];
    p.y = Rb[1] + -mu2 * rb[1];
    p.z = Rb[2] + -mu2 * rb[2];
    p.vx = Vb[0] + -mu2 * vb[0];
    p.vy = Vb[1] + -mu2 * vb[1];
    p.vz = Vb[2] + -mu2 * vb[2];
    s.x = Rb[0] + mu1 * rb[0];
    s.y = Rb[1] + mu1 * rb[1];
    s.z = Rb[2] + mu1 * rb[2];
    s.vx = Vb[0] + mu1 * vb[0];
    s.vy = Vb[1] + mu1 * vb[1];
    s.vz = Vb[2] + mu1 * vb[2];
    p.init = 1;
    s.init = 1;
  }

  return new_parts;
}

//----------------------------------------------------------------------------------------
//! \fn  int NBody::ReadTripleBlock
//! \brief
int ReadTripleBlock(ParameterInput *pin, parthenon::InputBlock *pib,
                    std::map<int, ParticleParams> &parts) {
  int new_parts = 0;
  std::vector<std::string> subs = split(pib->block_name, '/');
  if (subs.size() <= 2) {
    // <nbody/triple1>
    Real mass = pin->GetOrAddReal(pib->block_name, "mass", -1.0);
    struct Orbit orb_o = {0};
    orb_o.a = pin->GetReal(pib->block_name, "ao");
    orb_o.e = pin->GetOrAddReal(pib->block_name, "eo", 0.0);
    orb_o.i = (pin->GetOrAddReal(pib->block_name, "io", 0.0) / 180.0) * M_PI;
    orb_o.o = (pin->GetOrAddReal(pib->block_name, "oo", 0.0) / 180.0) * M_PI;
    orb_o.O = (pin->GetOrAddReal(pib->block_name, "Oo", 0.0) / 180.0) * M_PI;
    orb_o.f = (pin->GetOrAddReal(pib->block_name, "fo", 180.0) / 180.0) * M_PI;
    struct Orbit orb = {0};
    orb.a = pin->GetReal(pib->block_name, "a");
    orb.e = pin->GetOrAddReal(pib->block_name, "e", 0.0);
    orb.i = (pin->GetOrAddReal(pib->block_name, "i", 0.0) / 180.) * M_PI;
    orb.o = (pin->GetOrAddReal(pib->block_name, "o", 0.0) / 180.) * M_PI;
    orb.O = (pin->GetOrAddReal(pib->block_name, "O", 0.0) / 180.) * M_PI;
    orb.f = (pin->GetOrAddReal(pib->block_name, "f", 180.) / 180.) * M_PI;
    Real Rc[3] = {Null<Real>()}, Vc[3] = {Null<Real>()};
    Rc[0] = pin->GetOrAddReal(pib->block_name, "x", 0.0);
    Rc[1] = pin->GetOrAddReal(pib->block_name, "y", 0.0);
    Rc[2] = pin->GetOrAddReal(pib->block_name, "z", 0.0);
    Vc[0] = pin->GetOrAddReal(pib->block_name, "vx", 0.0);
    Vc[1] = pin->GetOrAddReal(pib->block_name, "vy", 0.0);
    Vc[2] = pin->GetOrAddReal(pib->block_name, "vz", 0.0);

    // Particles designation
    int pp = pin->GetOrAddInteger(pib->block_name, "primary", -1);
    int ss = pin->GetOrAddInteger(pib->block_name, "secondary", -1);
    int tt = pin->GetOrAddInteger(pib->block_name, "tertiary", -1);
    auto itp = parts.find(pp);
    auto its = parts.find(ss);
    auto itt = parts.find(tt);

    if ((itp == parts.end()) || (its == parts.end()) || (itt == parts.end()) ||
        (pp == -1) || (ss == -1) || (tt == -1)) {
      // Particles don't exist, create them
      const Real qo = pin->GetReal(pib->block_name, "qo");
      const Real q = pin->GetReal(pib->block_name, "q");
      const Real radius = pin->GetOrAddReal(pib->block_name, "radius", 0.0);
      const Real rs = pin->GetOrAddReal(pib->block_name, "rsoft", 0.0);
      const Real racc = pin->GetOrAddReal(pib->block_name, "rsink", 0.0);
      const Real gamma = pin->GetOrAddReal(pib->block_name, "gamma", 0.0);
      const Real beta = pin->GetOrAddReal(pib->block_name, "beta", 0.0);
      const Real target_rad = pin->GetOrAddReal(pib->block_name, "refine_distance", 0.0);
      const int live = pin->GetOrAddInteger(pib->block_name, "live", 0);
      const Real live_after = pin->GetOrAddReal(pib->block_name, "live_after", 0);
      const std::string stype = pin->GetOrAddString(pib->block_name, "stype", "spline");
      const int couple = pin->GetOrAddInteger(pib->block_name, "couple", 0);
      if (mass < 0.0) {
        std::stringstream msg;
        msg << "mass < 0 for " << pib->block_name << ". Please set the mass.";
        PARTHENON_FAIL(msg);
      }

      // Create new particles
      const Real m1 = mass / (1.0 + qo);
      const Real mb = qo * m1;
      const Real m2 = mb / (1.0 + q);
      const Real m3 = q * m2;
      ParticleParams part1 = CreateNewParticle(m1, radius, rs, stype, racc, gamma, beta,
                                               target_rad, live, live_after);
      ParticleParams part2 = CreateNewParticle(m2, radius, rs, stype, racc, gamma, beta,
                                               target_rad, live, live_after);
      ParticleParams part3 = CreateNewParticle(m3, radius, rs, stype, racc, gamma, beta,
                                               target_rad, live, live_after);
      part1.couple = couple;
      part2.couple = couple;
      part3.couple = couple;

      int maxid = 0;
      for (auto const &[id, p] : parts) {
        maxid = std::max(maxid, id);
      }
      if (pp == -1) {
        pp = maxid + 1;
        part1.id = pp;
        parts[pp] = part1;
        maxid++;
        new_parts++;
      }
      if (ss == -1) {
        ss = maxid + 1;
        part2.id = ss;
        parts[ss] = part2;
        new_parts++;
      }
      if (tt == -1) {
        tt = maxid + 1;
        part3.id = tt;
        parts[tt] = part3;
        new_parts++;
      }
    }

    // Correct masses
    auto &p = parts[pp];
    auto &s = parts[ss];
    auto &t = parts[tt];
    const Real q1 = (s.m + t.m) / p.m;
    const Real q2 = t.m / s.m;
    if (mass > 0) {
      p.m = mass / (1. + q1);
      s.m = q1 * p.m / (1. + q2);
      t.m = q2 * s.m;
    } else {
      mass = s.m + p.m + t.m;
    }
    const Real mb = s.m + t.m;

    // Initialize positions & velocities for outer binary
    Real Rb[3] = {Null<Real>()}, Vb[3] = {Null<Real>()};
    init_orbit(mass, orb_o, Rb, Vb);
    Real r0[3] = {Null<Real>()}, v0[3] = {Null<Real>()};
    Real mu1 = p.m / mass;
    Real mu2 = mb / mass;
    for (int i = 0; i < 3; i++) {
      r0[i] = Rc[i] - mu2 * Rb[i];
      v0[i] = Vc[i] - mu2 * Vb[i];
      Rc[i] += mu1 * Rb[i];
      Vc[i] += mu1 * Vb[i];
    }

    // Initialize positions & velocities for inner binary
    Real rb[3] = {Null<Real>()}, vb[3] = {Null<Real>()};
    init_orbit(mb, orb, rb, vb);
    mu1 = s.m / mb;
    mu2 = t.m / mb;

    Real r1[3] = {Null<Real>()}, v1[3] = {Null<Real>()};
    Real r2[3] = {Null<Real>()}, v2[3] = {Null<Real>()};
    for (int i = 0; i < 3; i++) {
      r1[i] = Rc[i] - rb[i] * mu2;
      r2[i] = Rc[i] + rb[i] * mu1;
      v1[i] = Vc[i] - vb[i] * mu2;
      v2[i] = Vc[i] + vb[i] * mu1;
    }

    p.x = r0[0];
    p.y = r0[1];
    p.z = r0[2];
    p.vx = v0[0];
    p.vy = v0[1];
    p.vz = v0[2];
    s.x = r1[0];
    s.y = r1[1];
    s.z = r1[2];
    s.vx = v1[0];
    s.vy = v1[1];
    s.vz = v1[2];
    t.x = r2[0];
    t.y = r2[1];
    t.z = r2[2];
    t.vx = v2[0];
    t.vy = v2[1];
    t.vz = v2[2];
    p.init = 1;
    s.init = 1;
    t.init = 1;
  }

  return new_parts;
}

//----------------------------------------------------------------------------------------
//! \fn  int NBody::ReadNBodySystemBlock
//! \brief Initializes a generic N-body system from a file
//! The input file should read:
//! # mass  x  y  z  vx   vy   vz   sft  gamma  beta target_rad
int ReadNBodySystemBlock(ParameterInput *pin, parthenon::InputBlock *pib,
                         std::map<int, ParticleParams> &parts) {
  const int couple = pin->GetOrAddInteger(pib->block_name, "couple", 1);
  const int live = pin->GetOrAddInteger(pib->block_name, "live", 0);
  const Real live_after = pin->GetOrAddReal(pib->block_name, "live_after", 0.0);
  SoftType stype = ReturnSoft(pin->GetOrAddString(pib->block_name, "stype", "spline"));

  std::string fname = pin->GetString(pib->block_name, "input_file");
  std::vector<std::vector<Real>> data = ArtemisUtils::loadtxt(fname);
  const int npart = static_cast<int>(data.size());
  int count = 0;
  int maxid = 0;
  for (auto const &[id, p] : parts) {
    maxid = std::max(maxid, id);
  }
  int id = maxid + 1;

  // Get the max id of the current particles
  for (auto row : data) {
    const auto len = row.size();
    // Process row
    ParticleParams p = {0};
    p.couple = couple;
    p.live = live;
    p.live_after = live_after;
    p.stype = stype;
    int icol = 0;
    p.id = count;
    p.m = (row[icol]);
    p.x = (row[++icol]);
    p.y = (row[++icol]);
    p.z = (row[++icol]);
    p.vx = (row[++icol]);
    p.vy = (row[++icol]);
    p.vz = (row[++icol]);
    p.rs = (row[++icol]);
    p.racc = p.rs;
    p.gamma = 0.0;
    p.radius = 0.0;
    p.beta = 0.0;
    p.target_rad = 0.0;
    if (len > ++icol) p.gamma = (row[icol]);
    if (len > ++icol) p.beta = (row[icol]);
    if (len > ++icol) p.target_rad = (row[icol]);
    if (len > ++icol) p.radius = (row[icol]);
    p.init = 1;
    parts[id] = p;
    count++;
    id++;
  }

  return count;
}

//----------------------------------------------------------------------------------------
//! \fn  int NBody::ReadPlanetarySystemBlock
//! \brief Initializes a planetary system from a file
//! Initialize a planetary system from a file
//!
//! The input file should read:
//! # q  a   e   i  f omega   bigOm   sft gamma  beta  target_rad radius
//!
//! User must add the central object with a separate particle / binary / system block
int ReadPlanetarySystemBlock(ParameterInput *pin, parthenon::InputBlock *pib,
                             std::map<int, ParticleParams> &parts) {
  const int couple = pin->GetOrAddInteger(pib->block_name, "couple", 1);
  const int live = pin->GetOrAddInteger(pib->block_name, "live", 0);
  const Real live_after = pin->GetOrAddReal(pib->block_name, "live_after", 0.0);
  SoftType stype = ReturnSoft(pin->GetOrAddString(pib->block_name, "stype", "spline"));

  std::string fname = pin->GetString(pib->block_name, "input_file");
  std::vector<std::vector<Real>> data = ArtemisUtils::loadtxt(fname);
  const int npart = static_cast<int>(data.size());
  int count = 0;
  int maxid = 0;
  for (auto const &[id, p] : parts) {
    maxid = std::max(maxid, id);
  }
  int id = maxid + 1;

  // Geet the max id of the current particles
  for (auto row : data) {
    const auto len = row.size();
    ParticleParams p = {0};
    p.couple = couple;
    p.live = live;
    p.live_after = live_after;
    p.stype = stype;
    int icol = 0;
    p.id = id;
    Orbit orb = {0};
    Real q = row[icol];
    orb.a = row[++icol];
    orb.e = row[++icol];
    orb.i = (row[++icol] / 180.0) * M_PI;
    orb.f = (row[++icol] / 180.0) * M_PI;
    orb.o = (row[++icol] / 180.0) * M_PI;
    orb.O = (row[++icol] / 180.0) * M_PI;
    p.rs = row[++icol];
    p.racc = p.rs;
    p.gamma = 0.0;
    p.beta = 0.0;
    p.target_rad = 0.0;
    p.radius = 0.0;
    if (len > ++icol) p.gamma = row[icol];
    if (len > ++icol) p.beta = row[icol];
    if (len > ++icol) p.target_rad = row[icol];
    if (len > ++icol) p.radius = row[icol];
    Real rb[3] = {Null<Real>()}, vb[3] = {Null<Real>()};
    init_orbit(1.0, orb, rb, vb);
    p.m = q;
    p.x = rb[0];
    p.y = rb[1];
    p.z = rb[2];
    p.vx = vb[0];
    p.vy = vb[1];
    p.vz = vb[2];
    p.init = 1;
    parts[id] = p;
    count++;
    id++;
  }

  return count;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::NBodySetup
//! \brief Go through an input file and create particles
//!
//! Blocks are laid out as
//!
//!   <nbody>
//!   <nbody/particle1>
//!     mass = 1.0
//!   <nbody/particle1/soft>
//!     rs = 0.1
//!     type = spline
//!   <nbody/particle1/sink>
//!     rs = 0.1
//!     gamma = 1.0
//!   <nbody/particle0>
//!     mass = 1.0
//!   <nbody/binary1>
//!    mass = 1.0
//!    particles = 1 , 2
//!
//! Particle blocks are processed first, then binary blocks, then triple blocks
//!
//! You can setup a binary either with two particle blocks + a binary block
//! or you can specify just a binary block
std::map<int, ParticleParams> NBodySetup(ParameterInput *pin, const Real G, Real &mresc) {
  int npart = 0;
  parthenon::InputBlock *pib = pin->pfirst_block;
  std::map<int, ParticleParams> parts;
  while (pib != nullptr) {
    if (pib->block_name.compare(0, 14, "nbody/particle") == 0) {
      // Get the id first from the block_name, reading up to the second "/""
      std::vector<std::string> subs = split(pib->block_name, '/');
      int id = std::stoi(subs[1].substr(8));
      // Check if we read this particle yet
      if (parts.count(id) == 0) {
        ParticleParams part = {0};
        part.id = id;
        parts[id] = part;
        npart++;
      }
      ReadParticleBlock(pin, pib, parts[id]);
    }
    pib = pib->pnext;
  }

  // Initialize particles
  pib = pin->pfirst_block;
  while (pib != nullptr) {
    if (pib->block_name.compare(0, 12, "nbody/binary") == 0) {
      int new_p = ReadBinaryBlock(pin, pib, parts);
      npart += new_p;
    } else if (pib->block_name.compare(0, 12, "nbody/triple") == 0) {
      int new_p = ReadTripleBlock(pin, pib, parts);
      npart += new_p;
    } else if (pib->block_name.compare(0, 12, "nbody/system") == 0) {
      int new_p = ReadNBodySystemBlock(pin, pib, parts);
      npart += new_p;
    } else if (pib->block_name.compare(0, 12, "nbody/planet") == 0) {
      int new_p = ReadPlanetarySystemBlock(pin, pib, parts);
      npart += new_p;
    }
    pib = pib->pnext;
  }

  // Normalize so that the total mass is equal to gravity/GM and the COM is at zero
  Real mtot = 0.0;
  Real R[3] = {0.0};
  Real V[3] = {0.0};
  for (auto const &[id, p] : parts) {
    mtot += p.m;
    R[0] += p.m * p.x;
    R[1] += p.m * p.y;
    R[2] += p.m * p.z;
    V[0] += p.m * p.vx;
    V[1] += p.m * p.vy;
    V[2] += p.m * p.vz;
  }
  if (mresc == -Big<Real>()) {
    mresc = mtot;
  }
  for (auto &[id, p] : parts) {
    parts[id].m = p.m * mresc / mtot;
    parts[id].x = p.x - R[0];
    parts[id].y = p.y - R[1];
    parts[id].z = p.z - R[2];
    parts[id].vx = p.vx - V[0];
    parts[id].vy = p.vy - V[1];
    parts[id].vz = p.vz - V[2];
  }
  // if (parthenon::Globals::my_rank == 0) {
  //  std::cout << npart << " Initial Particles: " << std::endl;
  //  for (auto const &[id, p] : parts) {
  //    PrintParticle(id, p);
  //  }
  //}

  return parts;
}

} // namespace NBody
