# Attribute CIFAR10 in parallel with `slurm` and `ffcv`
Here we featurize and score in parallel, and use `ffcv` for dataloading. This
speeds up the process significantly, and is the recommended way to featurize and
score for larger experiments.

## Setup
1. Install `ffcv`
2. Make sure you are on a cluster with `slurm`
3. Edit the `sbatch` script `run.sbatch` to your liking. The current version
will download 20 checkpoints of a DDPM model trained on CIFAR10 and featurize
them in parallel, with each process requesting one `A100` GPU and 20 CPUs.
Results will be saved in `./trak_results`
4. Run `sbatch run.sbatch`

Once the job is done, you can play around with the scores by loading them into
our demo notebook `examples/demo.py`, or analyze them on your own.