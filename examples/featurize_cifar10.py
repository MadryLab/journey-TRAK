from dataclasses import dataclass, field
from trak import TRAKer
from diffusion_trak import DiffusionModelOutput
from diffusion_trak import DiffusionGradientComputer
from diffusers import UNet2DModel
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import torch
import transformers
import numpy as np


@dataclass
class TrakConfig:
    proj_dim: int = field(default=2048)
    conditional_diffusion: bool = field(default=False)
    latent_diffusion: bool = field(default=False)
    vqvae_diffusion: bool = field(default=False)
    num_timesteps: int = field(default=5)
    start_tstep: int = field(default=0)
    end_tstep: int = field(default=1000)


@dataclass
class OtherConfig:
    dataset_name: str = field(default="cifar10")
    sample_size: int = field(default=32)
    n_channels: int = field(default=3)
    ckpt_dir: str = field(default="./checkpoints")
    batch_size: int = field(default=56)


def get_model(sample_size, n_channels) -> UNet2DModel:
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


def get_loader(config, split="train"):
    dataset = load_dataset(config.dataset_name, split=split)

    preprocess = transforms.Compose(
        [
            transforms.Resize((config.sample_size, config.sample_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # apply image transofmrations on the fly during training
    def transform(examples) -> dict[str, list]:
        if config.dataset_name == 'cifar10':
            images = examples["img"]
        else:
            images = examples["image"]

        images = [preprocess(image.convert("RGB")) for image in images]

        # return images
        return {"images": images}

    dataset.set_transform(transform)

    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=config.batch_size,
                                                   shuffle=False)

    return train_dataloader


def load_checkpoints_from_dir(ckpt_dir) -> list:
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

    loader_train = get_loader(config, split="train")

    ckpts = load_checkpoints_from_dir(config.ckpt_dir)

    task = DiffusionModelOutput(conditional=trak_config.conditional_diffusion,
                                latent=trak_config.latent_diffusion,
                                vqvae=trak_config.vqvae_diffusion)

    print(f'Loader length {len(loader_train.dataset)}')
    traker = TRAKer(model=model,
                    task=task,
                    gradient_computer=DiffusionGradientComputer,
                    proj_dim=trak_config.proj_dim,
                    train_set_size=len(loader_train.dataset),
                    device='cuda')

    traker.gradient_computer._are_we_featurizing = True
    traker.task._are_we_featurizing = True

    for model_id, ckpt in enumerate(ckpts):
        traker.load_checkpoint(ckpt, model_id=model_id)

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

    traker.finalize_features()

    # In case we want to score:
    traker.gradient_computer._are_we_featurizing = False
    traker.task._are_we_featurizing = False

    loader_val = get_loader(config, split="test")

    ...

    # def assemble_batch(feature_extractor, x_0s, x_ts):
    # current_bs = x_0s.shape[0]
    # # we can fix timesteps to be evenly spaced instead, if we want
    # timesteps = np.random.choice(np.arange(trak_config.start_tstep, trak_config.end_tstep),
    #                              size=[current_bs, trak_config.num_timesteps_score, 1],
    #                              replace=True)
    # timesteps = torch.randint(low=trak_config.start_tstep,
    #                           high=trak_config.end_tstep,
    #                           size=(current_bs, trak_config.num_timesteps_score, 1))
    # timesteps = torch.cat([timesteps, torch.max(torch.tensor(0.), timesteps - 1)], dim=-1).to(torch.long)

    # images = torch.stack([x_0s, x_ts], dim=2)

    # return [images,
    #         torch.zeros(current_bs),
    #         timesteps], current_bs
