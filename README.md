# Vision-Reconstruction-Pipeline
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
    └── README.md