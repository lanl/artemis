/*========================================================================================
** (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
**
** This program was produced under U.S. Government contract 89233218CNA000001 for Los
** Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
** for the U.S. Department of Energy/National Nuclear Security Administration. All rights
** in the program are reserved by Triad National Security, LLC, and the U.S. Department
** of Energy/National Nuclear Security Administration. The Government is granted for
** itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
** license in this material to reproduce, prepare derivative works, distribute copies to
** the public, perform publicly and display publicly, and to permit others to do so.
**========================================================================================
*/

function collapse_node(id) {
  /* Simple JS function to show/hide a div. */

  /* Return if this element doesn't exist */
  if (!document.getElementById) return;

  /* Else change toggle the display value between none and block */
  var myobj = document.getElementById(id).style;
  myobj.display = (myobj.display == 'block') ? 'none': 'block';

}
