# II. Inference

## Colab notebook (COMING SOON)


## Locally
TODO
Commandline on your own computer
something like

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

## Custom Try-on 