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


We demonstrate how to use the data analysis methods in ```demonstrations```, see the Jupyter notebooks. They also should 
take at most about a minute to run.

### Double quantum dot detection

In stage 1, the algorithm needs to make a decision on whether a double quantum dot was found.
We demonstrate how to use our model to make that decision in the notebook ```Double quantum dot classifier.ipynb```.

### Recentering

In stage 3, the first step is to center the bias triangles, i.e., finding their extent so that
downstream stages can accurately take mesurements. This is demonstrated in ```Recentering.ipynb```.


### Wide shot PSB detection

The algorithm needs to make a decision based on measurements with multiple pairs of bias triangles. 
The method based on autocorrelation and neural networks is demonstrated in ```Wide shot PSB detection.ipynb```.

### High res PSB detection 

After the initial detection of PSB, the algorithm further retakes higher resolution measurements and
re-classifies the bias triangles based on the higher res images and a computer vision routine. This is demonstrated in
```High res PSB detection.ipynb```.

### Danon gap check

As a final check for PSB, use a classifier that looks for the Danon gap. This is demonstrated in 
```Danon gap check.ipynb```.

### Entropy score

In stage 4, the algorithm uses a score based on the entropy of data to find the correct parameters to coherently drive a qubit. 
This score is demonstrated in ```Entropy score.ipynb```.

### Rabi chevron analysis

The Rabi chevron measurements are analysed to find the mid-point of the chevrons, which is needed
to drive a qubit coherently on resonance. The method is demonstrated in ```Rabi chevron analysis.ipynb```.


### Framework

For a simple demonstration of how the 
framework can be used, see ```pipelines/simple_dummy_pipeline.py```. It needs to be executed from within 
the ```pipelines``` folder. It will create a new folder in ```data/experiments``` that corresponds to the
name of a dummy experiment. If successfully executed, there will be a markdown file in this folder in 
```documentation``` which showcases the execution of a mock experiment.
Execution should only take a few seconds.

# Code used for full tuning


The full pipeline can be executed by running 'pipelines/from_scratch_to_rabi.py'.
You will need access to a double quantum dot device that can host a qubit and is measured
via transport measurements. The code likely needs adjustments to work with your setup. Specifically, you need to 
connect to your control electronics. This is handled in ```experiment_control/init_basel.py```.
Please get in touch with the authors to discuss details.

