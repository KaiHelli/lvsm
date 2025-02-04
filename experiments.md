### Full res. run old model - [wandb](https://wandb.ai/tum-edu/lvsm/runs/pugrk3tl/overview)
Description: Uses old source code, thus running it is not that simple.

Filename: `full_run_epoch_96-step_100000.ckpt`

```bash
git checkout 6ef024df1dc5f1a6bccf9ead508b89b5f1243aab
sed -i 's|^\(\s*\)use_flex_attn\s*=.*|\1use_flex_attn = False|' src/model/model_wrapper.py
sed -i 's|^\(\s*\)if model_on_gpu(self) and parse_version(torch.__version__) >= parse_version("2.5.0"):\s*$|\1if False:|' src/model/transformer/multi_head_attention.py
python -m src.main +experiment=re10k mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/full_run_epoch_96-step_100000.ckpt model.lvsm.transformer_cfg.qk_exp_seq_len=0
```

```bash
git checkout 6963ad3cbbeee62890adde7a5adac636e230ce0a
sed -i 's|^\(\s*\)self.mha_scale\s*=.*|\1self.mha_scale=None|' src/model/transformer/multi_head_attention.py
sed -i 's|^\(\s*\)use_flex_attn\s*=.*|\1use_flex_attn = self.model_cfg.transformer_cfg.sdpa_kernel == "flex-attention"|' src/model/model_wrapper.py
python -m src.main +experiment=re10k mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/full_run_epoch_96-step_100000.ckpt model.lvsm.transformer_cfg.qk_exp_seq_len=0 model.lvsm.transformer_cfg.sdpa_kernel=torch-sdpa
```

```bash
sed -i 's|^\(\s*\)self.mha_scale\s*=.*|\1self.mha_scale=None|' src/model/transformer/multi_head_attention.py
python -m src.main +experiment=re10k mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/full_run_epoch_96-step_100000.ckpt lvsm_cfg.transformer_cfg.qk_norm=1 +lvsm_cfg.transformer_cfg.qk_exp_seq_len=0 lvsm_cfg.transformer_cfg.sdpa_kernel=torch-sdpa
```
**Results:**
```
psnr 24.47130544562089
ssim 0.8130653782894737
lpips 0.12137643914473684
```

### Full res. run - [wandb](https://wandb.ai/tum-edu/lvsm/runs/44x9llvm)
Description: Finetuned from the old 11day training model `full_run_epoch_96-step_100000.ckpt`

Filename: `full_run_ft_epoch_96-step_100000.ckpt`

```bash
python -m src.main +experiment=re10k  mode=test dataset/view_sampler=evaluation test.compute_scores=true lvsm_cfg.transformer_cfg.sdpa_kernel=torch-sdpa checkpointing.load=checkpoints/full_run_ft_epoch_96-step_100000.ckpt
```


```bash
# Export videos and images
python -m src.main +experiment=re10k  mode=test dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json test.compute_scores=false test.save_video=true test.save_image=true lvsm_cfg.transformer_cfg.sdpa_kernel=flex-attention checkpointing.load=checkpoints/full_run_ft_epoch_96-step_100000.ckpt test.ffmpeg_path=/home/team15/ffmpeg-7.0.2-amd64-static
```

**Results:**
```
psnr 26.871857191386976
ssim 0.8729440789473685
lpips 0.10847553453947369
```

**Multi-View Experiment:**
```bash
python -m src.main +experiment=re10k  mode=test dataset/view_sampler=evaluation test.compute_scores=true lvsm_cfg.transformer_cfg.sdpa_kernel=torch-sdpa checkpointing.load=checkpoints/full_run_ft_epoch_96-step_100000.ckpt dataset.view_sampler.index_path=assets/multi_view_experiment/evaluation_index_num_src_2.json
python -m src.main +experiment=re10k  mode=test dataset/view_sampler=evaluation test.compute_scores=true lvsm_cfg.transformer_cfg.sdpa_kernel=torch-sdpa checkpointing.load=checkpoints/full_run_ft_epoch_96-step_100000.ckpt dataset.view_sampler.index_path=assets/multi_view_experiment/evaluation_index_num_src_3.json
python -m src.main +experiment=re10k  mode=test dataset/view_sampler=evaluation test.compute_scores=true lvsm_cfg.transformer_cfg.sdpa_kernel=torch-sdpa checkpointing.load=checkpoints/full_run_ft_epoch_96-step_100000.ckpt dataset.view_sampler.index_path=assets/multi_view_experiment/evaluation_index_num_src_4.json
python -m src.main +experiment=re10k  mode=test dataset/view_sampler=evaluation test.compute_scores=true lvsm_cfg.transformer_cfg.sdpa_kernel=torch-sdpa checkpointing.load=checkpoints/full_run_ft_epoch_96-step_100000.ckpt dataset.view_sampler.index_path=assets/multi_view_experiment/evaluation_index_num_src_5.json
python -m src.main +experiment=re10k  mode=test dataset/view_sampler=evaluation test.compute_scores=true lvsm_cfg.transformer_cfg.sdpa_kernel=torch-sdpa checkpointing.load=checkpoints/full_run_ft_epoch_96-step_100000.ckpt dataset.view_sampler.index_path=assets/multi_view_experiment/evaluation_index_num_src_6.json
python -m src.main +experiment=re10k  mode=test dataset/view_sampler=evaluation test.compute_scores=true lvsm_cfg.transformer_cfg.sdpa_kernel=torch-sdpa checkpointing.load=checkpoints/full_run_ft_epoch_96-step_100000.ckpt dataset.view_sampler.index_path=assets/multi_view_experiment/evaluation_index_num_src_7.json
python -m src.main +experiment=re10k  mode=test dataset/view_sampler=evaluation test.compute_scores=true lvsm_cfg.transformer_cfg.sdpa_kernel=torch-sdpa checkpointing.load=checkpoints/full_run_ft_epoch_96-step_100000.ckpt dataset.view_sampler.index_path=assets/multi_view_experiment/evaluation_index_num_src_8.json
```

```
num_src = [2, 3, 4, 5, 6, 7, 8]
model 1
psnr_a = [23.480610356186375, 23.866545128099848, 24.37404320456765, 24.667772466486152, 24.650622338959664, 24.632012396147758, 24.645158131917317]
ssim_a = [0.8074100378787878, 0.8121448863636364, 0.8237452651515151, 0.8265861742424242, 0.8243371212121212, 0.8251657196969697, 0.8239820075757576]
lpips_a = [0.15422289299242425, 0.14604048295454544, 0.1376657196969697, 0.13624526515151514, 0.1376657196969697, 0.1386126893939394, 0.14000355113636365]
```


### 8x8 run 1 - [wandb](https://wandb.ai/tum-edu/lvsm/runs/g1pj4x1v)
Filename: `8x8-r1-epoch_96-step_100000.ckpt`

```bash
python -m src.main +experiment=re10k_vae  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/8x8-r1-epoch_96-step_100000.ckpt 
```

**Results:**
```
psnr 21.585058061700117
ssim 0.6742907072368421
lpips 0.3643863075657895
```

### 8x8 run 2 - [wandb](https://wandb.ai/tum-edu/lvsm/runs/751eglvk)
Description: Finetuning based on 8x8 run 1 for another 100_000 steps.

Filename: `8x8-r2-epoch_96-step_100000.ckpt`

```bash
python -m src.main +experiment=re10k_vae  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/8x8-r2-epoch_96-step_100000.ckpt 
```

**Results:**
```
psnr 21.585058061700117
ssim 0.6742907072368421
lpips 0.3643863075657895
```

### 24x24 run 1 - [wandb](https://wandb.ai/tum-edu/lvsm/runs/da6vp79n)
Filename: `24x24-r1-epoch_96-step_100000.ckpt`

```bash
python -m src.main +experiment=re10k_vae  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/24x24-r1-epoch_96-step_100000.ckpt lvsm_cfg.patch_size=24 
```

**Results:**
```
psnr 18.51806264174612
ssim 0.50146484375
lpips 0.5971422697368421
```

### 6x6 run 1 - [wandb](https://wandb.ai/tum-edu/lvsm/runs/4rgpow2c)
Filename: `6x6-r1-epoch_96-step_100000.ckpt`

```bash
python -m src.main +experiment=re10k_vae  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/6x6-r1-epoch_96-step_100000.ckpt lvsm_cfg.patch_size=6 
```

**Results:**
```
psnr 22.57803211714092
ssim 0.7219880756578947
lpips 0.29425370065789475
```

### 4x4 run 1 - [wandb](https://wandb.ai/tum-edu/lvsm/runs/agnj9goq?nw=nwuserkaihelli)
Filename: `4x4-r1-epoch_96-step_100000.ckpt`

```bash
python -m src.main +experiment=re10k_vae  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/4x4-r1-epoch_96-step_100000.ckpt lvsm_cfg.patch_size=4 
```

**Results:**
```
psnr 23.58388604615864
ssim 0.7661903782894737
lpips 0.2327174136513158
```

### 4x4 run 200k steps - [wandb](https://wandb.ai/tum-edu/lvsm/runs/gv2in2xr?nw=nwuserkaihelli)
Filename: `4x4-200k-epoch_193-step_200000.ckpt`

```bash
python -m src.main +experiment=re10k_vae  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/4x4-200k-epoch_193-step_200000.ckpt lvsm_cfg.patch_size=4 
```

**Results:**
```
psnr 24.020906799717952
ssim 0.7836143092105263
lpips 0.21688682154605263
```

### 6x6 run 200k steps - multiple wandb runs...
Filename: `6x6-200k-epoch_193-step_200000.ckpt`

```bash
python -m src.main +experiment=re10k_vae  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/6x6-200k-epoch_193-step_200000.ckpt lvsm_cfg.patch_size=6 
```

**Results:**
```
psnr 23.221296812358656
ssim 0.7491776315789473
lpips 0.2590075041118421
```

### 4x4 run 200k steps decoder finetuning - [wandb](https://wandb.ai/tum-edu/lvsm/runs/wdgz38ez)
Description: VAE Decoder is LoRA finetuned starting off from `4x4-200k-epoch_193-step_200000.ckpt`

Filename: `4x4-200k-vae-ft_epoch_18-step_18700.ckpt`

```bash
python -m src.main +experiment=re10k_vae_ft  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/4x4-200k-vae-ft_epoch_18-step_18700.ckpt lvsm_cfg.patch_size=4 
```

```bash
# Export videos and images
python -m src.main +experiment=re10k_vae_ft  mode=test dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json test.compute_scores=false test.save_video=true test.save_image=true checkpointing.load=checkpoints/4x4-200k-vae-ft_epoch_18-step_18700.ckpt  lvsm_cfg.patch_size=4 test.ffmpeg_path=/home/team15/ffmpeg-7.0.2-amd64-static
```

**Results:**
```
psnr 24.352111088602168
ssim 0.7983141447368421
lpips 0.16682514391447367
```

**Multi-View Experiment:**
```bash
python -m src.main +experiment=re10k_vae_ft  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/4x4-200k-vae-ft_epoch_18-step_18700.ckpt lvsm_cfg.patch_size=4 dataset.view_sampler.index_path=assets/multi_view_experiment/evaluation_index_num_src_2.json
python -m src.main +experiment=re10k_vae_ft  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/4x4-200k-vae-ft_epoch_18-step_18700.ckpt lvsm_cfg.patch_size=4 dataset.view_sampler.index_path=assets/multi_view_experiment/evaluation_index_num_src_3.json
python -m src.main +experiment=re10k_vae_ft  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/4x4-200k-vae-ft_epoch_18-step_18700.ckpt lvsm_cfg.patch_size=4 dataset.view_sampler.index_path=assets/multi_view_experiment/evaluation_index_num_src_4.json
python -m src.main +experiment=re10k_vae_ft  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/4x4-200k-vae-ft_epoch_18-step_18700.ckpt lvsm_cfg.patch_size=4 dataset.view_sampler.index_path=assets/multi_view_experiment/evaluation_index_num_src_5.json
python -m src.main +experiment=re10k_vae_ft  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/4x4-200k-vae-ft_epoch_18-step_18700.ckpt lvsm_cfg.patch_size=4 dataset.view_sampler.index_path=assets/multi_view_experiment/evaluation_index_num_src_6.json
python -m src.main +experiment=re10k_vae_ft  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/4x4-200k-vae-ft_epoch_18-step_18700.ckpt lvsm_cfg.patch_size=4 dataset.view_sampler.index_path=assets/multi_view_experiment/evaluation_index_num_src_7.json
python -m src.main +experiment=re10k_vae_ft  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/4x4-200k-vae-ft_epoch_18-step_18700.ckpt lvsm_cfg.patch_size=4 dataset.view_sampler.index_path=assets/multi_view_experiment/evaluation_index_num_src_8.json
```

```
num_src = [2, 3, 4, 5, 6, 7, 8]
psnr_b = [21.478627233794242, 22.830755898446746, 23.76575082721132, 24.39931360880534, 24.738095023415305, 24.842648072676226, 25.040739464037348]
ssim_b = [0.7101089015151515, 0.7540246212121212, 0.787405303030303, 0.8061079545454546, 0.8125, 0.8172348484848485, 0.8200757575757576]
lpips_b = [0.24431818181818182, 0.20756392045454544, 0.18087121212121213, 0.1670217803030303, 0.16199100378787878, 0.15901692708333334, 0.15394176136363635]
```

### 4x4 run only 2 target views - [wandb](https://wandb.ai/tum-edu/lvsm/runs/gupat7c3)
Description: Finetuned from the 4x4 200k steps model `4x4-200k-epoch_193-step_200000.ckpt`

Filename: `4x4-2views-epoch_96_step_100000.ckpt`

```bash
python -m src.main +experiment=re10k_vae  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/4x4-2views-epoch_96_step_100000.ckpt lvsm_cfg.patch_size=4 
```

**Results:**
```
psnr 24.10576521722894
ssim 0.7950246710526315
lpips 0.2021484375
```

### 6x6 run only 2 target views - [wandb](https://wandb.ai/tum-edu/lvsm/runs/h44mlxlp)
Description: Finetuned from the 6x6 200k steps model `6x6-200k-epoch_193-step_200000.ckpt`

Filename: `6x6-2views-epoch_96_step_100000.ckpt`

```bash
python -m src.main +experiment=re10k_vae  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/6x6-2views-epoch_96_step_100000.ckpt lvsm_cfg.patch_size=6 
```

**Results:**
```
psnr 23.343489998265316
ssim 0.7612561677631579
lpips 0.24691611842105263
```

### 6x6 run finetune stochastic_encoder=False - [wandb](https://wandb.ai/tum-edu/lvsm/runs/h44mlxlp)
Description: Finetuned from the 6x6 200k steps model `6x6-200k-epoch_193-step_200000.ckpt`

Filename: `6x6-non_stoch-epoch_96_step_100000.ckpt`

```bash
python -m src.main +experiment=re10k_vae  mode=test dataset/view_sampler=evaluation test.compute_scores=true checkpointing.load=checkpoints/6x6-non_stoch-epoch_96_step_100000.ckpt lvsm_cfg.patch_size=6 lvsm_cfg.vae_cfg.stochastic_encoder=false
```

**Results:**
```
psnr 23.32089336294877
ssim 0.7550884046052632
lpips 0.2530453330592105
```
