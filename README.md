# Vision Reconstruction Pipeline
A Python pipeline to recreate a patient's vision from an fMRI dataset

In order to receive the dataset used for this pipeline, please head over to [The Algonauts Project](http://algonauts.csail.mit.edu/2021/challenge.html#DataRelease) website and follow the instructions to download the dataset.

Please also download the [Spike Image Decoder](https://github.com/jiankliu/Spike-Image-Decoder) so the overall file structure looks like so:


    ├── AlgonautsVideos268_All_30fpsmax
    │   ├── ...mp4
    │   └── ...mp4
    ├── participants_data_v2021
    │   ├── full_track
    │   │   ├── sub01
    │   │   ├── ...
    │   │   └── sub10   
    │   └── mini_track
    │       ├── sub01
    │       ├── ...
    │       └── sub10 
    ├── Spike-Image-Decoder
    │   └── SID.py
    ├── main.py
    ├── requirements.txt
    ├── README.md
    └── venv (will be created after following the commands below)

# Intructions to Get Started

## 1. Cloning Repos and Setting the Right File Structure
Begin by cloning this repository as well as cloning the [Spike Image Decoder](https://github.com/jiankliu/Spike-Image-Decoder) and downloading the [The Algonauts Project](http://algonauts.csail.mit.edu/2021/challenge.html#DataRelease) dataset (This may take a while depending on your network speed). Clone these projects and set up the file structure as seen above.

Please install [Python 3.7](https://www.python.org/downloads/release/python-370/) (newer versions do not support TensorFlow 1)

## 2. Virtual Environment (optional but highly recommended)

Setting up a virtual environment is the recommended way to run this pipeline to avoid cluttering your home environment although it is not technically required.

cd to the top level folder of cloned repo and run the command:

    python3.7 -m venv venv

to create a virtual environment using Python 3.7.

Activate the virtual environment by running:

    source venv/bin/activate

(Your terminal will likely have "(venv)" before your input)

## 3. Installing The Python Packages

The repository contains a requirements.txt with the exact versions of packages required for running this project. They can be installed by running:

    pip install -r requirements.txt

This may take a while depending on your network speed.

## 4. Run Pipeline

After the packages have been installed, the Pipeline can be run by running

    python main.py