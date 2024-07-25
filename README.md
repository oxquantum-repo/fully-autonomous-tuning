Code related to the publication [Fully autonomous tuning of a spin qubit](https://arxiv.org/abs/2402.03931)


[todo: installation]
[make installation easy, write instructions, test, report normal install time]





# Requirements

[todo: system requirements]
[talk about software dependencies, operating system, report versions the software was tested on]



## Data

Please download data from [here](https://drive.google.com/drive/folders/1_81KzZ8lP8I3Z0P9k4NsKs4xBhn9KErA?usp=share_link) (REPLACE WITH PUBLIC DATA REPO, E.G., ZENODO)
The data contains the trained models and data from the experiments to recreate the plots from the paper.

## Long path names on Windows

The full pipeline uses the file system to store information of the tuning process. For long pipelines, you need to enable long path names on Windows!
see: [this help article](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell)


## Key for MindFoundry server

If you use any of the routines that include a call to the MindFoundry server for optimisation, you need to obtain a key from them. Please get in touch with autoqt@mindfoundry.ai .


# Paper plots

First, download the data (see link above), then see the notebooks in ```data_visualisation``` to recreate the plots in the paper.


# Demonstrations
[todo: demonstration]
[report expected run times]

We demonstrate how to use the data analysis methods in ```demonstrations```, see the Jupyter notebooks. For a simple demonstration of how the 
framework can be used, see ```demonstrations/simple_dummy_pipeline.py```. It needs to be executed from within 
the ```demonstrations``` folder. It will create a new folder in ```data/experiments``` that corresponds to the
name of a dummy experiment. If successfully executed, there will be a markdown file in this folder in 
```documentation``` which showcases the execution of a mock experiment.

# Code used for full tuning
[todo: instructions for use]
[more details on how to run the software]


The full pipeline can be executed by running 'pipelines/from_scratch_to_rabi.py'.


