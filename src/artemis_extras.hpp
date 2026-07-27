//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
#ifndef ARTEMIS_EXTRAS_HPP_
#define ARTEMIS_EXTRAS_HPP_

#include <functional>

namespace artemis {

//  sources
using UnsplitTaskFn =
    std::function<TaskStatus(MeshData<Real> *md, const Real time, const Real dt)>;

struct UnsplitTask {
  std::string name;
  UnsplitTaskFn function;
};

void RegisterUnsplitExplicitTask(const std::string &name, UnsplitTaskFn function);
void RegisterUnsplitImplicitTask(const std::string &name, UnsplitTaskFn function);

const std::vector<UnsplitTask> &GetUnsplitImplicitTasks();
const std::vector<UnsplitTask> &GetUnsplitExplicitTasks();

// TaskCollection function pointer for operator split tasks
using TaskCollectionFnPtr = TaskCollection (*)(Mesh *pm, const Real time, const Real dt);


}

#endif // ARTEMIS_EXTRAS_HPP_
