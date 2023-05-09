# Vision Reconstruction Pipeline
A Python pipeline to recreate a patient's vision from an fMRI dataset

In order to receive the dataset used for this pipeline, please head over to [The Algonauts Project](http://algonauts.csail.mit.edu/challenge.html) website and follow the instructions to download the dataset.

Please also download the [Spike Image Decoder](https://github.com/jiankliu/Spike-Image-Decoder) so the overall file structure looks like so:


    ├── algonauts_2023_tutorial_data
    │   ├── subj01
    │   ├── ...
    │   └── subj08
    ├── Spike-Image-Decoder
    │   └── SID.py
    ├── main.py
    ├── requirements.txt
    └── README.md

# Intructions to Get Started

## 1. Cloning Repos and Setting the Right File Structure
Begin by cloning this repository as well as cloning the [Spike Image Decoder](https://github.com/jiankliu/Spike-Image-Decoder) and downloading the [The Algonauts Project](http://algonauts.csail.mit.edu/challenge.html) dataset (This may take a while depending on your network speed). Clone these projects and set up the file structure as seen above.

Please follow the instructions to install [TensorFlow](https://www.tensorflow.org/install/pip) until step 3, do not create a conda environment as this will be done in the next step.

## 2. Virtual Environment (optional but highly recommended)

A conda environment can be created using:

    conda env create -f environment.yml

This creates a Python 3.9 environment with the correct packages and package versions

Activate the conda environment by running:

    conda activate tf

Deactivate the environment by running:

    conda deactivate

Your terminal will likely have "(tf)" before your input when you activate the environment.

## 3. Run Pipeline

After the packages have been installed, the Pipeline can be run by running

    python main.py