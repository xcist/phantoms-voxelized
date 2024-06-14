## Plaque Phantom-voxelized
### Brief intro
This folder includes files for simulating a coronary plaque model using CatSim. Here are the volumetric rendering image and central cross-section images of the phantom:
![plaque overview picture](figures/Figure1.png "Volumetric rendering image and central cross-sections of the plaque model")
Below shows the key geometry information of the plaque:
![plaque geometry picture](figures/Figure2.png "Dimensions for the key features of the plaque components")
More details can be found in [our latest paper](https://arxiv.org/abs/2312.01566) (a new arXiv version to be updated to reflect the vessel wall thickness change).

### Usage
Please follow the steps below to use this plaque phantom for your simulation:
1. Download this folder *Plaque_phantom_voxelized*;
1. Copy all files in the materials folder into your CatSim material folder under the path: *YourCatSimPath/base/materials/*;
1. Add the path of this plaque model folder to your simulation environment searching path.
    - For matlab version CatSim, please add the following code into your simulation script:
    ```
    addpath('path_to_this_plaque_model_folder_on_your_computer');
    ```
    - For python version CatSim, please specify the path to this plaque phantom in your configuration file using the following format:
    ```
    cfg.phantom.filename = path_to_this_plaque_model_folder/LMZ_3D_Plaque0503_PCCT.json
    ```
1. Now you can start your simulation with CatSim.

### Citation
If you find our plaque model helpful, please cite [our paper](https://arxiv.org/abs/2312.01566) and we will be thankful. If you have questions regarding the plaque model, you can reach me at [lim34\@rpi.edu](mailto:lim34@rpi.edu?subject=QuestionsAboutThePlaqueModel). 
```
@article{li2023coronary,
  title={Coronary Atherosclerotic Plaque Characterization with Photon-counting CT: a Simulation-based Feasibility Study},
  author={Li, Mengzhou and Wu, Mingye and Pack, Jed and Wu, Pengwei and De Man, Bruno and Wang, Adam and Nieman, Koen and Wang, Ge},
  journal={arXiv preprint arXiv:2312.01566},
  year={2023}
}
``` 
