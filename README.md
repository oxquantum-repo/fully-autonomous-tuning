Code related to the publication [Fully autonomous tuning of a spin qubit](https://arxiv.org/abs/2402.03931)



# Requirements

## System requirements

This repo should work on all computers that can run a python environment. 

To run the full experiment, you will, of course, need access to a spin qubit experiment using transport measurements.

We tested this on a MacBook Air M2 with macOS Sonoma 14.5.


## Data

Please download data from [here](https://drive.google.com/drive/folders/1_81KzZ8lP8I3Z0P9k4NsKs4xBhn9KErA?usp=share_link).
The data contains the trained models and data from the experiments to recreate the plots from the paper.
Place the data in the folder ```data``` so that the demo modules can access it.


## To run the experiment live:

### Long path names on Windows

The full pipeline uses the file system to store information of the tuning process. For long pipelines, you need to enable long path names on Windows!
see: [this help article](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell)


### Key for MindFoundry server

If you use any of the routines that include a call to the MindFoundry server for optimisation, you need to obtain a key from them. Please get in touch with autoqt@mindfoundry.ai .



# Installation

To install this repo, we recommend using a virtual environmment manager, like anaconda, and create a dedicated environment.
The environment should have a python version <3.12. If you are using conda, run 

```conda create --name myenv310 python=3.10```

Activate the environment with 

```conda activate myenv310```


Once you have the environment, and the installer ```pip``` available, you can simply run the command 

```pip install .``` 

to install all necessary dependencies. The installation process should take around 10 mins.

The code here is making heavy use of subroutines that were developed in a separate package. We include this code, but it needs to be installed separately.
You need to navigate to the subfolder ```signal_processing/bias_triangle_processing```. Here, execute 

```pip install .```

again. This will make the module ```bias_triangle_detection``` available in the namespace.

Further, you need to navigate to the subfolder ```qcodes-addons```. Here, execute 

```pip install .```

again. This will make the module ```qcodes_addons``` available in the namespace.

These last two installations should only take a few minutes.


# Paper plots

First, download the data (see link above), then see the notebooks in ```data_visualisation``` to recreate the plots in the paper.


# Demonstrations
[todo: demonstration]
[report expected run times]
[write about the different demo notebooks]


We demonstrate how to use the data analysis methods in ```demonstrations```, see the Jupyter notebooks.


For a simple demonstration of how the 
framework can be used, see ```pipelines/simple_dummy_pipeline.py```. It needs to be executed from within 
the ```pipelines``` folder. It will create a new folder in ```data/experiments``` that corresponds to the
name of a dummy experiment. If successfully executed, there will be a markdown file in this folder in 
```documentation``` which showcases the execution of a mock experiment.
Execution should only take a few seconds.

# Code used for full tuning
[todo: instructions for use]
[more details on how to run the software]


The full pipeline can be executed by running 'pipelines/from_scratch_to_rabi.py'.


