## Simulation Folder
this folder is the environment for running the simulations for data collection. It contains the Main/Thermal circuits. The switch subcircuit with input parameters, and the two Qorvo models that make up the JFET.
 - ".asc" files are the schematic drawings
 - ".asy" files are the symbol diagrams
 - ".txt" are the Qorvo SPICE model files.

## Python Scripts
 - DataCollection - data collection notebook used to run simulations 
 - DataCollate - used to scrape the data from the raw simulation files and build small dataset fragments used later to build dataset. This must be ran on every PC used to collect data
 - DatasetCombine - used to combine pieces of datasets from multiple computers and adding data taken after the main simulation. This helps to produce a single file containing the dataset to load once
 - Training Script - containing functions for training, graphing results, and saving/loading network snapshots
 - SimDataScraper - contains all the functions to scrape the data from the simulation files and construct the dataset object to be used

## Other files:
 - ConverterParameterSweepExperiments - large file containing a number of experimental parameter sweeps to help isolate the most significant converter measurements to find the desired parameters
 - NNExperiment - Neural Network training experiments containing results and hyperparameters used
 - Neural Network Based Digital Twin for a Half-bridge Master's Defense - Master's Thesis defense slide show
 - NeuralNetwork_Based_DigitalTwin_for_a_HalfBridge_LLC_Resonant_Converter - the thesis PDF
 

### Python Environment: VS-Code and Jupyter Notebook:
Any file ending with *.ipynb* is a jupyter notebook which allows you to run blocks of code without rerunning the whole program. This is useful for running code to setup things, and then repeatedly running other sections without having to reload libraries or datasets. 

You can download a Jupyter notebook extension with the vs-code IDE, which is the recommended method. You can download VSCode [here](https://code.visualstudio.com/download).

### LTSpice
To simulate using this system, LTSpice is required. When using the components in the "Simulation Folder" make sure they have the right path. Transferring LTSpice files between computers usually requires updating the paths or just re-adding components. If you delete and add back the CCJFET subcircuit, make sure to add back the parameter input statement as well. 


### Third Patry Libraries:
*install command is "pip install **library name**" though other package managers exist for python*

- [ltspice library](https://pypi.org/project/ltspice/) - useful for reading ltspice raw files
- [pytorch with cuda](https://pytorch.org/get-started/locally/ ) - Machine Learning library
- [numpy](https://numpy.org/install/) (should come with pytorch) - general math/vector library
- [matplotlib](https://matplotlib.org/stable/users/installing/index.html) - for graphing



