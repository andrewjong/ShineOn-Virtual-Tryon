# I.  Installation and Data


## 1) Installation

### Clone
We use [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch) as a submodule for the 
Flow layers, so please add --recurse-submodules in the clone command:
```bash
git clone --recurse-submodules https://github.com/andrewjong/ShineOn-Virtual-Tryon.git
```

### Conda Environment
This code is tested on PyTorch 1.6.0 and cudatoolkit 9.2. 

<!---
Our system CUDA _driver_ is 10.1, though any 10.x CUDA driver should work. This code 
will NOT work if your system's CUDA driver is 9.x.
-->

1) Install and activate the conda environment with:
    ```bash
    conda env create -f sams-pt1.6.yml
    conda activate sams-pt1.6
    ```
2) Next, install the custom FlowNet2 CUDA layers. We use our 
[custom fork](https://github.com/andrewjong/flownet2-pytorch-1.0.1-with-CUDA-10) that 
adds support for RTX GPU architectures.
    ```bash
   cd models/flownet2_pytorch
   bash install.sh
    ```
   
3) Last, you must install the [FlowNet2 pre-trained checkpoint](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view)
   provided by NVIDIA. You should place this checkpoint under the folder `models/flownet2_pytorch`


That's it!


<details>
    <summary><b>Docker Image</b> (if problems above)</summary>
<br>
TODO: double check this works    
Having trouble with the conda install? You can try our provided [Docker Image](https://hub.docker.com/r/andrewjong/2021-wacv).

1) If you don't have Docker installed, follow NVIDIA's [Docker install guide](https://github.com/NVIDIA/nvidia-docker#getting-started).

2) Pull and run the image via:
    ```bash
    docker run -it \
    --name 2021-wacv \
    -v /PATH/TO/PROJECT_DIR:/2021-wacv-video-vton  \
    -v /data_hdd/fw_gan_vvt/:/data_hdd/fw_gan_vvt/ \
    -v /PATH_TO_WARP-CLOTH/:/data_hdd/fw_gan_vvt/train/warp-cloth \
    --gpus all --shm-size 8G \
    andrewjong/2021-wacv:latest /bin/bash
    ```
    
    And once within the Docker container, run `conda activate sams-pt1.6`.
    
</details>


## 2) VVT Dataset Download
We add 
[densepose](https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose), [schp](https://github.com/PeikeLi/Self-Correction-Human-Parsing) 
and 
[flow](https://github.com/NVIDIA/flownet2-pytorch)
annotations to FW-GAN's original _VVT dataset_ 
(original dataset courtesy of [Haoye Dong](http://www.scholat.com/donghaoye)).

### Generate SCHP Annotations

First, follow the installation instructions in [schp](https://github.com/PeikeLi/Self-Correction-Human-Parsing).
We used the [SCHP pre-trained model](https://drive.google.com/drive/folders/1uOaQCpNtosIjEL2phQKEdiYd0Td18jNo) and 
evaluate.py script to generate frame-by-frame human parsing annotations.

A generic algorithm for this would be:
```bash
import os
import os.path as osp

home = "/path/to/fw_gan_vvt/test/test_frames"
schp = "/path/to/Self-Correction-Human-Parsing"
output = "/path/to/fw_gan_vvt/test/test_frames_parsing"
os.chdir(home)
paths = os.listdir('.')
paths.sort()
for vid in paths:
    os.chdir(osp.join(home, vid))
    input_dir = os.getcwd()
    output_dir = osp.join(output, vid)
    generate_seg = "python evaluate.py --dataset lip --restore-weight 
        checkpoints/exp-schp-201908261155-lip.pth --input " + input_dir + 
        " --output " + output_dir
    os.chdir(schp)
    os.system(generate_seg)
```
### Generate Flow Annotations
Follow the installation instructions on our [custom fork](https://github.com/andrewjong/flownet2-pytorch-1.0.1-with-CUDA-10).
Then, using the training command that is provided. We use ImagesFromFolder dataset instead of the MPISintel dataset
in the command. Similar to the SCHP annotation algorithm, we generate frame-by-frame flow annotations using methods in
`models/flownet2_pytorch`.


### Generate DensePose Annotations
Follow the installation instructions on [densepose](https://github.com/facebookresearch/DensePose). 
Then, use the following command to generate densepose annotations,
```bash
python2 tools/infer_simple.py \
    --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml \
    --output-dir DensePoseData/infer_out/ \
    --image-ext [jpg or png] \
    --wts https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl \
    [Input image]
    ```
