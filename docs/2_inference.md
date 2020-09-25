# II. Inference

## Colab notebook (COMING SOON)


## Command Line


```bash
python test.py --model sams --checkpoint path/to/checkpoint.ckpt
```
## Reconstruction Task
1) Warp Module
    ```
    python test.py \
    --name reconstruction \
    --model warp \
    --workers 4 \
    --dataset vvt \
    --datamode test \
    --vvt_dataroot path/to/fw_gan_vvt \
    --checkpoint experiments/path/to/checkpoint.ckpt \
    ```

## Try-on Task
Specify --tryon_list to a CSV.
1) Warp Module

    The warp module takes the product images
    ```
    python test.py \

    --name tryon \
    --model warp \
    --workers 4 \
    --dataset vvt \
    --datamode test \
    --checkpoint experiments/path/to/checkpoint.ckpt \
    --tryon_list path/to/csv/file
    ```
   
2) Try-on Module
    ```
    python test.py \
    --name tryon \
    --model unet \
    --batch 4 \
    --datamode test \
    --checkpoint path/to/unet/checkpoint.ckpt \
    --tryon_list path/to/csv/file \
    --warp_cloth_dir results/tryon/checkpoint.ckpt/test/VVTDataset/warp-cloth
    ```

2) Try-on Module
    ```
    python test.py \
    --name experiment_2 \
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