# II. Inference

## Colab notebook (COMING SOON)


## Locally
TODO
Commandline on your own computer
something like

```bash
python test.py --model sams --checkpoint path/to/checkpoint.ckpt
```

## Try-on (Reproduce Results in the Paper)
1) Warp Module
    ```
    python test.py \                                                                           
    --name tryon_warp \
    --model warp \
    --workers 8 \
    --dataset vvt \
    --vvt_dataroot /data_hdd/fw_gan_vvt \
    --datamode test \
    --checkpoint checkpoints/train_gmm_cp-vvt-mpv_vera/step_033150.pth \
    --result_dir results
    ```
2) Try-on Module
    ```
    python test.py \                
    --name tryon_final \
    --model unet \
    --batch 4 
    --datamode test \
    --person_inputs densepose agnostic \
    --cloth_inputs cloth \
    --self_attn --flow --n_frames_now 2 --n_frames_total 2 \
    --checkpoint path/to/unet/checkpoint.ckpt \
    --warp_cloth_dir results/tryon_warp \
    --result_dir results/try_on
    ```

## Custom Try-on 