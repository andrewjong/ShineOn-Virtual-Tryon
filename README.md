# SAMS-GAN: Self-Attentive MultiSPADE  <br/> for Video Virtual Try-on

This repository contains the code for our paper at WACV... PAPER AND WEBSITE LINK.

Key Contributions:
- SAMS-GAN (a generative adversarial model)
- SAMS ResBlock (sequence of layers)
- Self-Attentive MultiSPADE normalization (single layer that combines several image annotations)

## Results

Some cool media COMING SOON.

## How It Works
Diagrams of SAMS GAN and SAMS layers.

GANs

Progressive Video Training

Warping Module

## Documentation
- [**I. Installation and Data**](docs/1_installation_and_data.md)
- [**II. Inference**](docs/2_inference.md)
- [**III. Train**](docs/3_train.md)
- [**IV. Custom Datasets**](docs/4_custom_tryon_dataset.md)

### Acknowledgements and Related Code
- This code is based in part on Sergey Wong's stellar CP-VTON repository. Thank you very much,
Sergey, for your hard work.
- Thank you Haoye Dong and his team for hosting the VUHCS competition at CVPR 2020, 
providing the VVT Dataset, and giving access to the FW-GAN reference code.
- Thank you NVIDIA's team for their work on Vid2Vid and FlowNet2.
- Thank you Arun Mallya for answering implementation questions about MultiSPADE from WC-Vid2Vid.
- Credits to Self-Attention GAN for attention layers reference.
- Credits to Self-Corrective Human-Parsing for easy parsing of LIP clothing labels.
- Credits to the detectron2 repository for Densepose annotations.
