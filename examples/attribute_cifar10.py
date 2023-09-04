from dataclasses import dataclass, field
from trak import TRAKer
from diffusion_trak import DiffusionModelOutput
from diffusion_trak import DiffusionGradientComputer
from diffusers import UNet2DModel
from pathlib import Path
import torch
import transformers


@dataclass
class TrakConfig:
    proj_dim: int = field(default=4096)
    conditional_diffusion: bool = field(default=False)
    latent_diffusion: bool = field(default=False)
    vqvae_diffusion: bool = field(default=False)
    batch_size: int = field(default=16)


@dataclass
class OtherConfig:
    dataset_name: str = field(default="CIFAR10")
    sample_size: int = field(default=32)
    n_channels: int = field(default=3)
    ckpt_dir: str = field(default="./checkpoints")


def get_model(sample_size, n_channels):
    # Model matching DDPM paper
    model = UNet2DModel(
        sample_size=sample_size,  # the target image resolution
        in_channels=n_channels,  # the number of input channels, 3 for RGB images
        out_channels=n_channels,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 256, 256, 256),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "UpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D"
            ),
        attention_head_dim=None,
    )

    return model.cuda()


def get_loaders(config):
    # TODO
    pass


def load_checkpoints_from_dir(ckpt_dir):
    ckpts = []
    ckpt_dir = Path(ckpt_dir)
    for ckpt_path in sorted(list(ckpt_dir.iterdir())):
        ckpts.append(torch.load(ckpt_path))
    return ckpts


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((TrakConfig, OtherConfig))
    trak_config, config = parser.parse_args_into_dataclasses()

    model = get_model(config.sample_size, config.n_channels)
    model.eval()

    loader_train, loader_test = get_loaders(config)

    ckpts = load_checkpoints_from_dir(config.ckpt_dir)

    task = DiffusionModelOutput(conditional=trak_config.conditional_diffusion,
                                latent=trak_config.latent_diffusion,
                                vqvae=trak_config.vqvae_diffusion)

    traker = TRAKer(model=model,
                    task=task,
                    gradient_computer=DiffusionGradientComputer,
                    proj_dim=config.proj_dim,
                    train_set_size=len(loader_train.indices),
                    device='cuda')
