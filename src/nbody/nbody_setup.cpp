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

#include <cctype>
#include <set>

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

bool IsDecimal(const std::string &value) {
  return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char c) {
    return std::isdigit(c) != 0;
  });
}

int ResolveParticleReference(ParameterInput *pin, const std::string &block_name,
                             const std::string &field,
                             const std::map<std::string, int> &particle_ids) {
  if (!pin->DoesParameterExist(block_name, field)) return -1;
  const std::string raw = SanitizeString(pin->GetAsUnresolvedString(block_name, field));
  if (IsDecimal(raw)) return std::stoi(raw);
  auto it = particle_ids.find(raw);
  if (it != particle_ids.end()) return it->second;
  std::stringstream msg;
  msg << "Unknown particle reference '" << raw << "' in " << block_name << "/" << field;
  PARTHENON_FAIL(msg);
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
void ReadParticleBlock(ParameterInput *pin, std::string block_name,
                       ParticleParams &part) {
  std::vector<std::string> subs = split(block_name, '/');
  if (subs.size() <= 2) {
    // <nbody/particle1>
    part.m = pin->GetReal(block_name, "mass");
    part.radius = pin->GetOrAddReal(block_name, "radius", 0.0);
    part.couple = pin->GetOrAddInteger(block_name, "couple", 1);
    part.live = pin->GetOrAddInteger(block_name, "live", 0);
    part.live_after = pin->GetOrAddReal(block_name, "live_after", 0.0);
    part.target_rad = pin->GetOrAddReal(block_name, "refine_distance", 0.0);
  } else {
    if (subs[2] == "soft") {
      // <nbody/particle1/soft>
      std::string ityp = pin->GetString(block_name, "type");
      if (ityp == "none") {
        part.rs = 0.0;
        part.stype = SoftType::plummer;
      } else if (ityp == "plummer") {
        part.rs = pin->GetReal(block_name, "radius");
        part.stype = SoftType::plummer;
      } else if (ityp == "spline") {
        part.rs = pin->GetReal(block_name, "radius");
        part.stype = SoftType::spline;
      } else {
        std::stringstream msg;
        msg << "Unknown particle softening type " << ityp;
        PARTHENON_FAIL(msg);
      }
    } else if (subs[2] == "sink") {
      // <nbody/particle1/sink>
      part.racc = pin->GetReal(block_name, "radius");
      part.gamma = pin->GetReal(block_name, "gamma");
      part.beta = pin->GetOrAddReal(block_name, "beta", 0.0);
    } else if (subs[2] == "initialize") {
      // <nbody/particle1/init>
      part.x = pin->GetOrAddReal(block_name, "x", 0.0);
      part.y = pin->GetOrAddReal(block_name, "y", 0.0);
      part.z = pin->GetOrAddReal(block_name, "z", 0.0);
      part.vx = pin->GetOrAddReal(block_name, "vx", 0.0);
      part.vy = pin->GetOrAddReal(block_name, "vy", 0.0);
      part.vz = pin->GetOrAddReal(block_name, "vz", 0.0);
      part.init = 1;
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  int NBody::ReadBinaryBlock
//! \brief
int ReadBinaryBlock(ParameterInput *pin, std::string block_name,
                    std::map<int, ParticleParams> &parts,
                    const std::map<std::string, int> &particle_ids) {
  int new_parts = 0;
  std::vector<std::string> subs = split(block_name, '/');
  if (subs.size() <= 2) {
    // <nbody/binary1>
    Real mass = pin->GetOrAddReal(block_name, "mass", -1.0);
    struct Orbit orb = {0};
    orb.a = pin->GetReal(block_name, "a");
    orb.e = pin->GetOrAddReal(block_name, "e", 0.0);
    orb.i = (pin->GetOrAddReal(block_name, "i", 0.0) / 180.0) * M_PI;
    orb.o = (pin->GetOrAddReal(block_name, "o", 0.0) / 180.0) * M_PI;
    orb.O = (pin->GetOrAddReal(block_name, "O", 0.0) / 180.0) * M_PI;
    orb.f = (pin->GetOrAddReal(block_name, "f", 180.0) / 180.0) * M_PI;
    Real Rb[3] = {Null<Real>()}, Vb[3] = {Null<Real>()};
    Rb[0] = pin->GetOrAddReal(block_name, "x", 0.0);
    Rb[1] = pin->GetOrAddReal(block_name, "y", 0.0);
    Rb[2] = pin->GetOrAddReal(block_name, "z", 0.0);
    Vb[0] = pin->GetOrAddReal(block_name, "vx", 0.0);
    Vb[1] = pin->GetOrAddReal(block_name, "vy", 0.0);
    Vb[2] = pin->GetOrAddReal(block_name, "vz", 0.0);

    // Particles designation
    int pp = ResolveParticleReference(pin, block_name, "primary", particle_ids);
    int ss = ResolveParticleReference(pin, block_name, "secondary", particle_ids);
    auto itp = parts.find(pp);
    auto its = parts.find(ss);

    if ((itp == parts.end()) || (its == parts.end()) || (pp == -1) || (ss == -1)) {
      const Real qb = pin->GetReal(block_name, "q");
      const Real radius = pin->GetOrAddReal(block_name, "radius", 0.0);
      const Real rs = pin->GetOrAddReal(block_name, "rsoft", 0.0);
      const Real racc = pin->GetOrAddReal(block_name, "rsink", 0.0);
      const Real gamma = pin->GetOrAddReal(block_name, "gamma", 0.0);
      const Real beta = pin->GetOrAddReal(block_name, "beta", 0.0);
      const Real target_rad = pin->GetOrAddReal(block_name, "refine_distance", 0.0);
      const int live = pin->GetOrAddInteger(block_name, "live", 0);
      const Real live_after = pin->GetOrAddReal(block_name, "live_after", 0);
      const std::string stype = pin->GetOrAddString(block_name, "stype", "spline");
      const int couple = pin->GetOrAddInteger(block_name, "couple", 0);
      if (mass < 0.0) {
        std::stringstream msg;
        msg << "mass < 0 for " << block_name << ". Please set the mass.";
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
int ReadTripleBlock(ParameterInput *pin, std::string block_name,
                    std::map<int, ParticleParams> &parts,
                    const std::map<std::string, int> &particle_ids) {
  int new_parts = 0;
  std::vector<std::string> subs = split(block_name, '/');
  if (subs.size() <= 2) {
    // <nbody/triple1>
    Real mass = pin->GetOrAddReal(block_name, "mass", -1.0);
    struct Orbit orb_o = {0};
    orb_o.a = pin->GetReal(block_name, "ao");
    orb_o.e = pin->GetOrAddReal(block_name, "eo", 0.0);
    orb_o.i = (pin->GetOrAddReal(block_name, "io", 0.0) / 180.0) * M_PI;
    orb_o.o = (pin->GetOrAddReal(block_name, "oo", 0.0) / 180.0) * M_PI;
    orb_o.O = (pin->GetOrAddReal(block_name, "Oo", 0.0) / 180.0) * M_PI;
    orb_o.f = (pin->GetOrAddReal(block_name, "fo", 180.0) / 180.0) * M_PI;
    struct Orbit orb = {0};
    orb.a = pin->GetReal(block_name, "a");
    orb.e = pin->GetOrAddReal(block_name, "e", 0.0);
    orb.i = (pin->GetOrAddReal(block_name, "i", 0.0) / 180.) * M_PI;
    orb.o = (pin->GetOrAddReal(block_name, "o", 0.0) / 180.) * M_PI;
    orb.O = (pin->GetOrAddReal(block_name, "O", 0.0) / 180.) * M_PI;
    orb.f = (pin->GetOrAddReal(block_name, "f", 180.) / 180.) * M_PI;
    Real Rc[3] = {Null<Real>()}, Vc[3] = {Null<Real>()};
    Rc[0] = pin->GetOrAddReal(block_name, "x", 0.0);
    Rc[1] = pin->GetOrAddReal(block_name, "y", 0.0);
    Rc[2] = pin->GetOrAddReal(block_name, "z", 0.0);
    Vc[0] = pin->GetOrAddReal(block_name, "vx", 0.0);
    Vc[1] = pin->GetOrAddReal(block_name, "vy", 0.0);
    Vc[2] = pin->GetOrAddReal(block_name, "vz", 0.0);

    // Particles designation
    int pp = ResolveParticleReference(pin, block_name, "primary", particle_ids);
    int ss = ResolveParticleReference(pin, block_name, "secondary", particle_ids);
    int tt = ResolveParticleReference(pin, block_name, "tertiary", particle_ids);
    auto itp = parts.find(pp);
    auto its = parts.find(ss);
    auto itt = parts.find(tt);

    if ((itp == parts.end()) || (its == parts.end()) || (itt == parts.end()) ||
        (pp == -1) || (ss == -1) || (tt == -1)) {
      // Particles don't exist, create them
      const Real qo = pin->GetReal(block_name, "qo");
      const Real q = pin->GetReal(block_name, "q");
      const Real radius = pin->GetOrAddReal(block_name, "radius", 0.0);
      const Real rs = pin->GetOrAddReal(block_name, "rsoft", 0.0);
      const Real racc = pin->GetOrAddReal(block_name, "rsink", 0.0);
      const Real gamma = pin->GetOrAddReal(block_name, "gamma", 0.0);
      const Real beta = pin->GetOrAddReal(block_name, "beta", 0.0);
      const Real target_rad = pin->GetOrAddReal(block_name, "refine_distance", 0.0);
      const int live = pin->GetOrAddInteger(block_name, "live", 0);
      const Real live_after = pin->GetOrAddReal(block_name, "live_after", 0);
      const std::string stype = pin->GetOrAddString(block_name, "stype", "spline");
      const int couple = pin->GetOrAddInteger(block_name, "couple", 0);
      if (mass < 0.0) {
        std::stringstream msg;
        msg << "mass < 0 for " << block_name << ". Please set the mass.";
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
int ReadNBodySystemBlock(ParameterInput *pin, std::string block_name,
                         std::map<int, ParticleParams> &parts) {
  const int couple = pin->GetOrAddInteger(block_name, "couple", 1);
  const int live = pin->GetOrAddInteger(block_name, "live", 0);
  const Real live_after = pin->GetOrAddReal(block_name, "live_after", 0.0);
  SoftType stype = ReturnSoft(pin->GetOrAddString(block_name, "stype", "spline"));

  std::string fname = pin->GetString(block_name, "input_file");
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
int ReadPlanetarySystemBlock(ParameterInput *pin, std::string block_name,
                             std::map<int, ParticleParams> &parts) {
  const int couple = pin->GetOrAddInteger(block_name, "couple", 1);
  const int live = pin->GetOrAddInteger(block_name, "live", 0);
  const Real live_after = pin->GetOrAddReal(block_name, "live_after", 0.0);
  SoftType stype = ReturnSoft(pin->GetOrAddString(block_name, "stype", "spline"));

  std::string fname = pin->GetString(block_name, "input_file");
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
  std::map<int, ParticleParams> parts;
  auto blocks = pin->GetBlockNamesWithPrefix("nbody/particle");
  std::set<int> used_ids;
  for (const auto &block_name : blocks) {
    const auto subs = split(block_name, '/');
    if (subs.size() != 2) continue;
    const std::string instance = subs.back();
    if (instance.rfind("particle", 0) == 0 && IsDecimal(instance.substr(8)))
      used_ids.insert(std::stoi(instance.substr(8)));
  }
  std::map<std::string, int> particle_ids;
  for (const auto &block_name : blocks) {
    const auto subs = split(block_name, '/');
    if (subs.size() != 2) continue;
    const std::string instance = subs.back();
    int id;
    if (instance.rfind("particle", 0) == 0 && IsDecimal(instance.substr(8))) {
      id = std::stoi(instance.substr(8));
    } else {
      id = 0;
      while (used_ids.count(id) != 0)
        ++id;
      used_ids.insert(id);
    }
    particle_ids[instance] = id;
    particle_ids[block_name] = id;
    // Check if we read this particle yet
    if (parts.count(id) == 0) {
      ParticleParams part = {0};
      part.id = id;
      parts[id] = part;
      npart++;
    }
    ReadParticleBlock(pin, block_name, parts[id]);
  }

  // Apply child configuration blocks using the effective parent path. Canonical
  // metadata makes `<nbody/particle(star)>` and `<./soft>` work without
  // requiring a `particleN` spelling.
  for (const auto &block : pin->GetBlocks()) {
    const auto segments = split(block.name, '/');
    const bool legacy_child =
        block.canonical_path.empty() && segments.size() == 3 &&
        segments[1].rfind("particle", 0) == 0 &&
        (segments[2] == "soft" || segments[2] == "sink" || segments[2] == "initialize");
    if (block.canonical_path != "nbody/particle/soft" &&
        block.canonical_path != "nbody/particle/sink" &&
        block.canonical_path != "nbody/particle/initialize" && !legacy_child)
      continue;
    const auto slash = block.name.find_last_of('/');
    if (slash == std::string::npos) continue;
    auto parent = particle_ids.find(block.name.substr(0, slash));
    if (parent != particle_ids.end())
      ReadParticleBlock(pin, block.name, parts[parent->second]);
  }

  // Initialize particles
  // Note that it is done this way to retain input deck ordering
  for (const auto &block : pin->GetBlocks()) {
    const auto &block_name = block.name;
    const bool binary =
        block.canonical_path == "nbody/binary" ||
        (block.canonical_path.empty() && block_name.rfind("nbody/binary", 0) == 0);
    const bool triple =
        block.canonical_path == "nbody/triple" ||
        (block.canonical_path.empty() && block_name.rfind("nbody/triple", 0) == 0);
    const bool system =
        block.canonical_path == "nbody/system" ||
        (block.canonical_path.empty() && block_name.rfind("nbody/system", 0) == 0);
    const bool planet =
        block.canonical_path == "nbody/planet" ||
        (block.canonical_path.empty() && block_name.rfind("nbody/planet", 0) == 0);
    if (binary) {
      int new_p = ReadBinaryBlock(pin, block_name, parts, particle_ids);
      npart += new_p;
    } else if (triple) {
      int new_p = ReadTripleBlock(pin, block_name, parts, particle_ids);
      npart += new_p;
    } else if (system) {
      int new_p = ReadNBodySystemBlock(pin, block_name, parts);
      npart += new_p;
    } else if (planet) {
      int new_p = ReadPlanetarySystemBlock(pin, block_name, parts);
      npart += new_p;
    }
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

  return parts;
}

} // namespace NBody
