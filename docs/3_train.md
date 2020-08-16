# III. Train

Our approach contains two models: 
the core **SAMS-GAN** and the auxiliary WarpModule.

We provide pretrained weights for the WarpModule, that you can find here (COMING SOON).
The WarpModule can be treated as a black box. It is used to pre-warp the garment image to
the shape of the user.

We also compare against a baseline UNet-Mask model (based on the TOM model from CP-VTON).


<details>
<summary><b>Tensorboard</b></summary>

All training progress can be viewed in Tensorboard.
```bash
tensorboard --logdir experiments/
```
We can port forward Tensorboard from a remote server like this:
```bash
ssh -N -L localhost:6006:localhost:6006 username@IP.ADDRESS
```


</details>


## SAMS-GAN
<details>
<summary>Instructions</summary>
<br />

COMING SOON

```bash
python train.py \
--name "SAMS-GAN_train" \
--model sams \
--ngf_power_start 6 \
--ngf_power_end 10 \
--n_frames_total 5 \
--n_frames_now 1 \
--batch_size 4 \
--num_workers 8
```

Progressive Training:
- `--n_frames_total` . Aim for max that fits on GPU, 5+ is ideal.
- `--n_frames_now` . Slowly manually increase from 1 up to `--n_frames_total`

Topics:
- Generator size
- Self attention
- N-Frames
    - trade batch size for more frames
- Progressive Training
- Tensorboard

</details>

## Baseline U-Net Mask (aka TOM)
<details>
<summary>Instructions</summary>
<br />


COMING SOON

</details>

## WarpModule (Optional)
<details>
<summary>Instructions</summary>
<br />


COMING SOON

</details>
