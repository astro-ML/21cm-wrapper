p21cmfast wrapper v0.3

Quickstart:
1) Install the requirements (pip install -r requirements.txt)
2) Set the cache directory for p21cmfast in the p21cmfastwrapper.py, line 17
3) Import the classes (from p21cmfastwrapper import *)
4) Open 21cmfast.ipynb for a brief live introduction

2 and 4 are optional steps

Philosphy:
- Optionality: All functions have set default arguments to simplify execution but also allow for
lower level calls
- Flexibility: The classes are written to work with them in a .ipynb notebook

Addendum:

There are two classes: Parameters (which handles everything related to the parameters) and Simulation (which handles p21cmfast and the visualization).
Simulation inherits Parameters, so there is no need to work with the Parameters class at all.

To-do (Priority):
- [x] implement (uniform / distribution function) sampling of parameters, run sims, and save them

To-do:
- [x] true multiprocessing of py21cmfast (GIL a problem??)
- [x] add saving method (!)
- [ ] make ps using custom z-bins
- [ ] show progress bars
- [ ] more cleanup, especially the plotting routines
- [ ] make a compare method (like ps) for global_props plot 

