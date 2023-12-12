# Get attribution scores using our pre-trained models pre-computed TRAK features
## DDPMs trained on CIFAR-10
Run our [`demo_CIFAR10.ipynb`](https://github.com/MadryLab/journey-TRAK/blob/main/examples/demo_CIFAR10.ipynb) notebook. In just a few minutes, you can generate images with our pre-trained models and compute attribution scores like these:

![example scores CIFAR-10](https://github.com/MadryLab/journey-TRAK/blob/main/assets/example_scores.png)

## LDMs trained on MS COCO
Run our [`demo_MSCOCO.ipynb`](https://github.com/MadryLab/journey-TRAK/blob/main/examples/demo_MSCOCO.ipynb) notebook. In just a few minutes, you can generate images with our pre-trained models and compute attribution scores.

# Compute attribution features

## DDPMs trained on CIFAR-10
Simply run `python examples/featurize_cifar10.py --ckpt_dir <CKPT DIR>` and TRAK
features will be saved in `./trak_results`.
Once you are done, you can get scores in our demo notebook `demo_CIFAR10.ipynb`.


## Compute attribution features in parallel with `slurm` and `ffcv`
Here we featurize and score in parallel, and use `ffcv` for dataloading. This
speeds up the process significantly, and is the recommended way to featurize and
score for larger experiments.

### Setup
1. Install `ffcv`
2. Make sure you are on a cluster with `slurm`


### DDPMs trained on CIFAR-10
1. Edit the `sbatch` script `featurize_cifar10_parallel.sbatch` to your liking. The current version
will download 20 checkpoints of a DDPM model trained on CIFAR-10 and featurize
them in parallel, with each process requesting one `A100` GPU and 8 CPUs.
Results will be saved in `./trak_results`
2. Run `sbatch examples/featurize_cifar10_parallel.sbatch` from the root of the repo

   
### LDMs trained on MS COCO
1. Edit the `sbatch` script `featurize_mscoco_parallel.sbatch` to your liking. The current version
will download 20 checkpoints of an LDM model trained on MS COCO and featurize
them in parallel, with each process requesting one `A100` GPU and 8 CPUs.
Results will be saved in `./trak_results`
2. Run `sbatch examples/featurize_mscoco_parallel.sbatch` from the root of the repo

Once you are done, you can get scores in our demo notebook `demo_MSCOCO.ipynb`.
