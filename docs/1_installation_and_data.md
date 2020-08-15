# I.  Installation and Data


## 1) Installation

### Clone
We use some submodules, so please add --recurse-submodules in the clone command:
```bash
git clone --recurse-submodules https://github.com/andrewjong/2021-wacv-video-vton.git
```


### Conda Environment
This code is tested on PyTorch 1.4.0 and cudatoolkit 10.0. 

Our system CUDA _driver_ is 10.1, though any 10.x CUDA driver should work. This code will NOT work if your system's CUDA driver is 9.x. If you have an 
incompatible CUDA driver, this repository can still be tested through our 
Colab notebook (COMING SOON).

1) Install and activate the conda environment with:
    ```bash
    conda env create -f sams_flownet_env.yml
    conda activate sams
    ```
2) Next, install the custom FlowNet2 CUDA layers. We use a custom fork that adds 
support for CUDA 10.0 and RTX GPU architectures.
    ```bash
   cd models/flownet2_pytorch
   bash install.sh
   cd ../..
    ```
That's it!

## 2) VVT Dataset Download
We add 
[densepose](https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose) 
and 
[flow](https://github.com/NVIDIA/flownet2-pytorch)
annotations to FW-GAN's original _VVT dataset_ 
(original dataset courtesy of [Haoye Dong](http://www.scholat.com/donghaoye)).

You may download our full annotated FW-GAN dataset here (COMING SOON).
