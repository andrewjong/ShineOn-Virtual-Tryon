# II. Inference

## Colab notebook (COMING SOON)


## Command Line

### Reconstruction Task
Reconstruction tests how well model can synthesize the person re-wearing their ORIGINAL 
clothes.

1) Warp Module

    The warp module takes the product images
    ```
    python test.py \
    --name reconstruction \
    --model warp \
    --workers 4 \
    --batch 4 \
    --dataset vvt \
    --datamode test \
    --checkpoint experiments/path/to/checkpoint.ckpt
    ```
   
2) Try-on Module
    ```
    python test.py \
    --name tryon \
    --model unet \
    --workers 4 \
    --batch 4 \
    --dataset vvt \
    --datamode test \
    --checkpoint path/to/unet/checkpoint.ckpt \
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
    ```
    python test.py \
    --name tryon \
    --model warp \
    --workers 4 \
    --batch 4 \
    --dataset vvt \
    --datamode test \
    --checkpoint experiments/path/to/checkpoint.ckpt \
    --tryon_list path/to/tryon_file.csv
    ```

2) Try-on Module
    ```
    python test.py \
    --name tryon \
    --model unet \
    --workers 4 \
    --batch 4 \
    --dataset vvt \
    --datamode test \
    --checkpoint path/to/unet/checkpoint.ckpt \
    --tryon_list path/to/tryon_file.csv \
    --warp_cloth_dir \
    test_results/tryon/checkpoint.ckpt/test/VVTDataset/warp-cloth
    ```
