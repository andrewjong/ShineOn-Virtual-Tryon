# II. Inference

## Pre-trained Checkpoint
We provide our ShineOn model checkpoint on 
[Google Drive](https://drive.google.com/file/d/1mwSiJEzdzxXCuIm07QyRxGVo6qhAwBiC/view?usp=sharing).

## Command Line

### Reconstruction Task
Reconstruction tests how well model can synthesize the person re-wearing their ORIGINAL 
clothes.

1) Warp Module

    The warp module takes the product images
    ```bash
    python test.py \
    --name reconstruction_warp \
    --model warp \
    --workers 4 \
    --batch 4 \
    --dataset vvt \
    --datamode test \
    --checkpoint experiments/path/to/WARP/checkpoint.ckpt
    ```
   
2) Try-on Module
    ```bash
    python test.py \
    --name reconstruction_try_on \
    --model unet \
    --workers 4 \
    --batch 4 \
    --dataset vvt \
    --datamode test \
    --checkpoint experiments/path/to/UNET/checkpoint.ckpt \
    --warp_cloth_dir \
    test_results/reconstruction/checkpoint.ckpt/test/VVTDataset/warp-cloth
    ```



### Try-on Task
Try-on tests how well the model can synthesize the person wearing a NEW article of 
clothing.

Use the flag `--tryon_list` to choose your CSV file that specifies cloth-person try on 
pairs.

The CSV file should be formatted with two columns (no headers):
```
path/to/cloth/product_image.png, VIDEO_ID
```
where `path/to/cloth/product_image.png` is a path to the cloth image, and `VIDEO_ID`
is the name of a video folder containing the frames for one person.


1) Warp Module

    The `warp-cloth` folder must be generated for every `tryon_file.csv` you have.

    ```bash
    python test.py \
    --name warp_try_on \
    --model warp \
    --workers 4 \
    --batch 4 \
    --dataset vvt \
    --datamode test \
    --checkpoint experiments/path/to/WARP/checkpoint.ckpt \
    --tryon_list path/to/tryon_file.csv
    ```

2) Try-on Module
    ```bash
    python test.py \
    --name complete_try_on \
    --model unet \
    --workers 4 \
    --batch 4 \
    --dataset vvt \
    --datamode test \
    --checkpoint experiments/path/to/UNET/checkpoint.ckpt \
    --tryon_list path/to/tryon_file.csv \
    --warp_cloth_dir \
    test_results/tryon/checkpoint.ckpt/test/VVTDataset/warp-cloth
    ```