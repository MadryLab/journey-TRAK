from dataclasses import dataclass, field
from trak import TRAKer
from diffusion_trak import DiffusionModelOutput
from diffusion_trak import DiffusionGradientComputer
from tqdm import tqdm
from pathlib import Path
import torch
import transformers
import numpy as np

from utils import get_cifar_loader, get_cifar_model


@dataclass
class TrakConfig:
    proj_dim: int = field(default=2048)
    conditional_diffusion: bool = field(default=False)
    latent_diffusion: bool = field(default=False)
    num_timesteps: int = field(default=20)
    start_tstep: int = field(default=0)
    end_tstep: int = field(default=1000)
    save_dir: str = field(default="./trak_results")


@dataclass
class OtherConfig:
    dataset_name: str = field(default="cifar10")
    sample_size: int = field(default=32)
    n_channels: int = field(default=3)
    ckpt_dir: str = field(default="./checkpoints")
    batch_size: int = field(default=64)
    model_id: int = field(default=-1)


def load_checkpoint_from_dir(ckpt_dir, model_id) -> list:
    ckpt_dir = sorted(list(Path(ckpt_dir).iterdir()))
    print(f'Loading checkpoint from {ckpt_dir[model_id]}')
    return torch.load(ckpt_dir[model_id])


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((TrakConfig, OtherConfig))
    trak_config, config = parser.parse_args_into_dataclasses()

    model = get_cifar_model(config.sample_size, config.n_channels)
    model.eval()

    loader_train = get_cifar_loader(config.batch_size, split="train")
    ckpt = load_checkpoint_from_dir(config.ckpt_dir, config.model_id)

    task = DiffusionModelOutput(conditional=trak_config.conditional_diffusion,
                                latent=trak_config.latent_diffusion)

    print(f'Loader length {len(loader_train.dataset)}')
    traker = TRAKer(model=model,
                    task=task,
                    gradient_computer=DiffusionGradientComputer,
                    proj_dim=trak_config.proj_dim,
                    save_dir=trak_config.save_dir,
                    train_set_size=len(loader_train.dataset),
                    load_from_save_dir=(config.model_id == 0),
                    device='cuda')

    traker.gradient_computer._are_we_featurizing = True
    traker.task._are_we_featurizing = True

    traker.load_checkpoint(ckpt, model_id=config.model_id)

    for batch in tqdm(loader_train, desc='Computing TRAK embeddings...'):
        # batch will consist of [image, label, timestep]
        current_bs = batch['images'].shape[0]

        # CIFAR-10 is unconditional, so below we are adding a dummy label
        batch = [batch['images'], torch.tensor([0] * current_bs, dtype=torch.int32)]
        batch = [x.cuda() for x in batch]
        # we can fix timesteps to be evenly spaced instead, if we want
        timesteps = np.random.choice(np.arange(trak_config.start_tstep,
                                               trak_config.end_tstep),
                                     size=[current_bs, trak_config.num_timesteps],
                                     replace=True)
        batch.append(torch.tensor(timesteps).cuda())

        traker.featurize(batch=batch, num_samples=current_bs)

traker.finalize_features(model_ids=[config.model_id])
