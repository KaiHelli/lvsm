<h1 align="center">LVSM: A Large View Synthesis Model with<br>Minimal 3D Inductive Bias</h1>

An unofficial implementation of the multi-view image synthesis architecture proposed in the [LVSM](https://haian-jin.github.io/projects/LVSM/) paper. This implementation may not fully adhere to all original details or guarantee complete accuracy.

## Installation

To get started, clone this project, create a conda virtual environment using Python 3.10+, and install the requirements:

```bash
git clone https://github.com/kaihelli/lvsm.git
cd lvsm
conda create -n lvsm python=3.10
conda activate lvsm
# In case a different CUDA version than the default (12.4) is required, execute the following line with the correct version specified.
pip install 'torch>=2.5.1' 'torchvision>=0.20' --index-url https://download.pytorch.org/whl/cu124
# In case 3D scene visualisations are needed, install PyTorch3D (fvcore is not needed as a requirement since PyTorch3D v0.7.8)
# Make sure torch and torchvision is installed first using the line above.
pip install ninja iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# Install the rest of the requirements
pip install -r requirements.txt
```

If you plan to use `flex-attention` version 2.5.1 of the `torch` package lacks important [fixes](https://github.com/pytorch/pytorch/issues/135161) that currently are only included in the nightly releases. To install them, do the following:

```bash
git clone https://github.com/kaihelli/lvsm.git
cd lvsm
conda create -n lvsm_nightly python=3.10
conda activate lvsm_nightly
# For flex-attemtion, install the latest nightly release.
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
# In case 3D scene visualisations are needed, install PyTorch3D (fvcore is not needed as a requirement since PyTorch3D v0.7.8)
# Make sure torch and torchvision is installed first using the line above.
pip install ninja iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# Install the rest of the requirements
pip install -r requirements_nightly.txt
```

## Acquiring Datasets

LVSM uses RealEstate10K for scene-level experiments as well as Objaverse for object-level datasets. This repository only builds upon the scene-level dataset RealEstate10K. Adding support for Objaverse might be added at a later point.

### RealEstate10K

As this repository builds upon MVSplat, it supports the same datasets as pixelSplat. Below we quote pixelSplat's [detailed instructions](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) on getting datasets.

> pixelSplat was trained using versions of the RealEstate10k and ACID datasets that were split into ~100 MB chunks for use on server cluster file systems. Small subsets of the Real Estate 10k and ACID datasets in this format can be found [here](https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe?usp=sharing). To use them, simply unzip them into a newly created `datasets` folder in the project root directory.

> If you would like to convert downloaded versions of the Real Estate 10k and ACID datasets to our format, you can use the [scripts here](https://github.com/dcharatan/real_estate_10k_tools). Reach out to us (pixelSplat) if you want the full versions of our processed datasets, which are about 500 GB and 160 GB for Real Estate 10k and ACID respectively.

In order to allow for preprocessing such as VAE pre-encoding or image scaling, we encode some meta information for a saved dataset. After extracting RealEstate10K, create `meta.json` in the root dataset directory with the following contents:

```json
{
    "expected_shape": [3, 360, 640]
}
```

For rescaling, pre-encoding and compressing the dataset, see the scripts `src/scripts/preprocess_dataset.py`as well as `src/scripts/compress_dataset.py`.

## Running the Code

### Evaluation

To render novel views and compute evaluation metrics from a pretrained model,

* get the [pretrained models](https://drive.google.com/drive/folders/1-CiI4o2CyfFF8VX3biroksxE-Wr5lJcI?usp=sharing), and save them to `/checkpoints`

* run the following:

```bash
# re10k
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true

# re10k with vae
python -m src.main +experiment=re10k_vae \
checkpointing.load=checkpoints/re10k_vae.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true
```

* the rendered novel views will be stored under `outputs/test`

To render videos from a pretrained model, run the following

```bash
# re10k
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
test.save_video=true \
test.save_image=false \
test.compute_scores=false

# re10k with vae
python -m src.main +experiment=re10k_vae \
checkpointing.load=checkpoints/re10k_vae.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
test.save_video=true \
test.save_image=false \
test.compute_scores=false
```

To reproduce the results of our work, see `experiments.md` as well as `evaluation.ipynb`.

### Training

Run the following:

```bash
# train lvsm on re10k without vae
python -m src.main +experiment=re10k dataset.roots=["path to dataset"]

# train lvsm on re10k with vae
python -m src.main +experiment=re10k_vae dataset.roots=["path to dataset"]
```

Our models are trained with a single RTX3080 TI (11GB) GPU.

<details>
  <summary><b>Training on multiple nodes (https://github.com/donydchen/mvsplat/issues/32)</b></summary>
Since this project is built on top of pytorch_lightning, it can be trained on multiple nodes hosted on the SLURM cluster. For example, to train on 2 nodes (with 2 GPUs on each node), add the following lines to the SLURM job script. Note however, this is untested in the current version of the code.

```bash
#SBATCH --nodes=2           # should match with trainer.num_nodes
#SBATCH --gres=gpu:2        # gpu per node
#SBATCH --ntasks-per-node=2

# optional, for debugging
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
# optional, set network interface, obtained from ifconfig
export NCCL_SOCKET_IFNAME=[YOUR NETWORK INTERFACE]
# optional, set IB GID index
export NCCL_IB_GID_INDEX=3

# run the command with 'srun'
srun python -m src.main +experiment=re10k dataset.roots=["path to dataset"] \
data_loader.train.batch_size=4 \
trainer.num_nodes=2
```

References:
* [Pytorch Lightning: RUN ON AN ON-PREM CLUSTER (ADVANCED)](https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html)
* [NCCL: How to set NCCL_SOCKET_IFNAME](https://github.com/NVIDIA/nccl/issues/286)
* [NCCL: NCCL WARN NET/IB](https://github.com/NVIDIA/nccl/issues/426)

</details>

<details>
  <summary><b>Fine-tune from the released weights (https://github.com/donydchen/mvsplat/issues/45)</b></summary>
To fine-tune from the released weights <i>without</i> loading the optimizer states, run the following:

```bash
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/re10k.ckpt \
checkpointing.resume=false
```

</details>

## BibTeX

```bibtex
@misc{jin2024lvsmlargeviewsynthesis,
      title={LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias}, 
      author={Haian Jin and Hanwen Jiang and Hao Tan and Kai Zhang and Sai Bi and Tianyuan Zhang and Fujun Luan and Noah Snavely and Zexiang Xu},
      year={2024},
      eprint={2410.17242},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.17242}, 
}
```

## Acknowledgements

This project builds upon the paper [LVSM](https://haian-jin.github.io/projects/LVSM/). The code is largely based on [MVSplat](https://github.com/donydchen/mvsplat) with the multi-view view sampler provided by [DepthSplat](https://github.com/cvg/depthsplat) and [latentSplat](https://github.com/Chrixtar/latentsplat). Many thanks to these projects for their excellent contributions!
