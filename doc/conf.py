# ========================================================================================
# (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import glob


# These are routines that create the auto generated input parameter documentation
def open_close_div(count, title, css="art_node", disp="art_hide"):
    # Creates a collapsable div
    return """
  <div class=\"{}\" onclick=\"collapse_node('menu.{:d}')\">
    {}
   </div>
  <div id=\"menu.{:d}\" class=\"{}\">
""".format(
        css, count, title, count, disp
    )


def add_card(title, card, count, tabs=0, debug=False):
    # Adds a new node / option / variable to the documentation
    lines = ""
    if tabs == 0:
        tab_ = ""
        ntab_ = ""
    else:
        tab_ = "|" + "".join(["&nbsp;&nbsp;.&nbsp;&nbsp;"] * tabs)
        ntab_ = "".join(["&nbsp;&nbsp;&nbsp;&nbsp;"] * tabs)

    # common keys
    _type = card.pop("_type", None)
    _desc = card.pop("_description", "")
    _def = card.pop("_default", None)
    _units = card.pop("_units", "")

    btag = ["", ""]
    if _type is None or _type == "node":
        css_class = "art_node"
    elif _type == "opt":
        css_class = "art_opt"
        tab_ += "&nbsp;-&nbsp;"
    else:
        css_class = "art_var"
        if _def is None:
            btag[0] = "<b>"
            btag[1] = "</b>"

    lines += (
        "  "
        + open_close_div(count, tab_ + btag[0] + title + btag[1], css=css_class)
        + "\n"
    )
    count += 1
    desc_div = False
    type_div = False
    def_div = False
    units_div = False
    if len(_desc) > 0:
        desc_div = True
        extra_line = ""  # "| " if( _type is None or _type == "node" ) else ""
        lines += (
            "  "
            + open_close_div(
                count, extra_line + ntab_ + _desc, css="art_desc", disp="art_show"
            )
            + "\n"
        )
        count += 1
    if _type is not None and _type != "node" and _type != "opt":
        type_div = True
        lines += (
            "  "
            + open_close_div(
                count,
                ntab_ + "<em>" + "Type: " + str(_type) + "</em>",
                css="art_type",
                disp="art_show",
            )
            + "\n"
        )
        count += 1
    if len(_units) > 0:
        units_div = True
        lines += (
            "  "
            + open_close_div(
                count,
                ntab_ + "<em>" + "Units: " + str(_units) + "</em>",
                css="art_type",
                disp="art_show",
            )
            + "\n"
        )
        count += 1
    if _def is not None:
        def_div = True
        lines += (
            "  "
            + open_close_div(
                count,
                ntab_ + "<em>" + "Default: " + str(_def) + "</em>",
                css="art_type",
                disp="art_show",
            )
            + "\n"
        )
        count += 1

    for key, val in card.items():
        new_lines, count = add_card(key, val, count, tabs=tabs + 1, debug=debug)
        lines += new_lines

    lines += "  </div>\n"
    if desc_div:
        lines += "  </div>\n"
    if type_div:
        lines += "  </div>\n"
    if units_div:
        lines += "  </div>\n"
    if def_div:
        lines += "  </div>\n"
    return lines, count


def gen_rst(files, output, debug=False):
    # Create the parameters.rst file from the package yaml files
    import yaml
    from yaml import CLoader as Loader

    header = """
.. _parameters:

Input Parameters
================

The following is a complete listing of all possible input blocks and their parameters.
Click on a block to expand its contents. Required parameters are highlighted in black. 
Clicking on a parameter will bring up a description, available options, data type, and units when applicable.

.. raw:: html

"""

    lines = ""
    count = 1
    for file in files:
        with open(file, "r") as f:
            pars = yaml.load(f, Loader)
        for key, val in pars.items():
            new_lines, count = add_card(key, val, count, debug=debug)
            lines += new_lines
    with open(output, "w") as f:
        f.write(header)
        f.write(lines)


# Get all the yaml files in the source directory

files = glob.glob("../src/**/*.yaml", recursive=True)

# Create src/parameters.rst
gen_rst(files, "./src/parameters.rst", debug=False)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "artemis"

# replace with correct things
copyright = """
 (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
"""

author = "Adam M. Dempsey, et al."

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

# Add our special css and js files
html_static_path = ["_static"]
html_css_files = ["artemis_style.css"]
html_js_files = ["collapse.js"]

rst_prolog = """
.. |code| replace:: **artemis**
.. |artemis| replace:: **artemis**
.. |jaybenne| replace:: **jaybenne**
"""
