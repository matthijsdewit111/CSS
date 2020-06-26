# Complex System Simulation
Project of the course Complex System Simulation part of the Msc Computational Science at the University of Amsterdam

#### Authors
* Parva Alavian
* Coen Lenting
* Sam Verhezen
* Matthijs de Wit

## Webpage
The webpage of the project is displayed at [https://matthijsdewit111.github.io/CSS](https://matthijsdewit111.github.io/CSS/index.html).

## Project
### General
The development of neuronal dendrites was simulated using diffusion-limited aggregation (DLA) including pruning in both 2 and 3 dimensional space. Pruning decreased the amount of terminal leafs and therefore provided the dendritic structure to the model. This project is based on a paper of A. Luczak (2006), who
simulated various types of dendritic neurons in 3D using DLA.

### Analysis
* The analysis consists of calculating and plotting the asymmetry index, branching order and terminal leafs with varying pruning spans (PS) of 10, 20, 30, 40 and 50.
* The average was taken over 5 simulations per PS (n=5)

## Files
### Data structure
* ```neuronal_tree.py``` contains the Tree and Node classes to represent the dendrite structure, is called by the DLA models

### DLA models
* ```diff2d.py``` and ```diff3d.py``` are the DLA model with diffusion in 2D and 3D, respectively
* ```randomwalk_2d.py``` and ```randomwalk_3d.py``` are the DLA model with the random walker in 2D and 3D, respectively

### Analysis
* ```fractal-dim.ipynb``` calculates the fractal dimension of the 2D model output, this was not used in the results
* ```analysisdiff.ipynb``` runs and analyses the diffusion condition over 20 simulations (n=20)
* ```analysis_randomwalker.ipynb``` runs and analyses the random walker condition over 5 simulations (n=5)

### Visualisation
* ```visualisation.ipynb``` is used to create the output shapes of the 2D and 3D conditions (picture and gif, respectively)

## How to run
* Run either one of the four DLA models to run one simulation and plot the outcome in 2D or 3D.
* To perform the analysis, run either one of the analysis files
