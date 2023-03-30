# lattice_boltzman_fluid

Simulating Navier-Stokes equation using the 2DQ9 Lattice-Boltzman method.

### Dependencies
The neccesary dependencies for this project can be install via the poetry command:
```
poetry install
```

### Files Descriptions
Here is a short description of every files and folders to provide a better understanding of how to code is structured. For a better understanting of the simulations and the physics behind please consult the article provided.
  - **inputs.yaml:** Input parameters
  - **lattice.py:** Contain a description of the lattice class used for the simulations.
  - **lbm_flow.py:** Code used to simulate fluid dynamic arround various obstacles.
  - **cavity.py:** Code used to simulate fluid in the lid-driven cavity benchmark.
  - **anim.py:** Provide animation and various figures for each simulations.


### Article
You can find more informations on this project and the lbm method and this project in the file **modelisation_navier-stokes_lbm.pdf**
