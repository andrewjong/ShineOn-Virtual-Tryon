# I.  Installation and Data


## 1) Installation

### Clone
We use [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch) as a submodule for the 
Flow layers, so please add --recurse-submodules in the clone command:
```bash
git clone --recurse-submodules https://github.com/andrewjong/2021-wacv-video-vton.git
```


### Conda Environment
This code is tested on PyTorch 1.6.0 and cudatoolkit 9.2. 

<!---
Our system CUDA _driver_ is 10.1, though any 10.x CUDA driver should work. This code 
will NOT work if your system's CUDA driver is 9.x. If you have an 
incompatible CUDA driver, this repository can still be tested through our 
Colab notebook (COMING SOON).
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
