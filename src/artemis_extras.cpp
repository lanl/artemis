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

#include <utility> // std::move
#include <vector>  // std::vector

#include "artemis.hpp"
#include "artemis_extras.hpp"

namespace artemis {

std::vector<UnsplitTask> unsplit_expl_tasks;
std::vector<UnsplitTask> unsplit_impl_tasks;
std::vector<SplitTaskList> split_tasks;

//----------------------------------------------------------------------------------------
//! \fn void Artemis::RegisterUnsplitExplicitTask
//! \brief Registers a named unsplit and explicit task in execution order
void RegisterUnsplitExplicitTask(const std::string &name, UnsplitTaskFn function) {
  PARTHENON_REQUIRE(!name.empty(), "Unsplit task names cannot be empty");
  PARTHENON_REQUIRE(function != nullptr, "Cannot register a null unsplit task");
  for (const auto &task : unsplit_expl_tasks) {
    PARTHENON_REQUIRE(task.name != name, "Duplicate unsplit explicit task name: " + name);
  }
  unsplit_expl_tasks.push_back({name, std::move(function)});
}

//----------------------------------------------------------------------------------------
//! \fn void Artemis::RegisterUnsplitImplicitTask
//! \brief Registers a named unsplit and implcit task in execution order
void RegisterUnsplitImplicitTask(const std::string &name, UnsplitTaskFn function) {
  PARTHENON_REQUIRE(!name.empty(), "Unsplit task names cannot be empty");
  PARTHENON_REQUIRE(function != nullptr, "Cannot register a null unsplit task");
  for (const auto &task : unsplit_impl_tasks) {
    PARTHENON_REQUIRE(task.name != name, "Duplicate unsplit implicit task name: " + name);
  }
  unsplit_impl_tasks.push_back({name, std::move(function)});
}

//----------------------------------------------------------------------------------------
//! \fn void Artemis::RegisterSplitTaskList
//! \brief Registers a named operator split task collection in execution order
void RegisterSplitTaskList(const std::string &name, SplitTaskListFn function) {
  PARTHENON_REQUIRE(!name.empty(), "Split task collection name cannot be empty");
  PARTHENON_REQUIRE(function != nullptr, "Cannot register a null split task collection");
  for (const auto &task : split_tasks) {
    PARTHENON_REQUIRE(task.name != name, "Duplicate split task collection name: " + name);
  }
  split_tasks.push_back({name, std::move(function)});
}

//----------------------------------------------------------------------------------------
//! \fn const std::vector<UnsplitTask> &Artemis::GetUnsplitExplicitTasks
//! \brief Returns unsplit and explicit tasks in registration order
const std::vector<UnsplitTask> &GetUnsplitExplicitTasks() { return unsplit_expl_tasks; }

//----------------------------------------------------------------------------------------
//! \fn const std::vector<UnsplitTask> &Artemis::GetUnsplitImplicitTasks
//! \brief Returns unsplit and implicit tasks in registration order
const std::vector<UnsplitTask> &GetUnsplitImplicitTasks() { return unsplit_impl_tasks; }

//----------------------------------------------------------------------------------------
//! \fn const std::vector<UnsplitTask> &Artemis::GetSplitTaskLists
//! \brief Returns operator split task collections in registration order
const std::vector<SplitTaskList> &GetSplitTaskLists() { return split_tasks; }

} // namespace artemis
