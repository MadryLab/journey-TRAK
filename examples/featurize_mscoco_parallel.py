from dataclasses import dataclass, field
from trak import TRAKer
from diffusion_trak import DiffusionModelOutput
from diffusion_trak import DiffusionGradientComputer
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from pathlib import Path
import torch
import transformers
import numpy as np

from utils import get_mscoco_model, get_mscoco_loader

# to prevent:
# "RuntimeError: Batching rule not implemented for
# aten::_chunk_grad_outputs_efficient_attention. We could not generate a
# fallback."
# when running with functorch, we need to disable memory-efficient SDP

torch.backends.cuda.enable_mem_efficient_sdp(False)


@dataclass
class TrakConfig:
    proj_dim: int = field(default=2048)
    conditional_diffusion: bool = field(default=True)
    latent_diffusion: bool = field(default=True)
    num_timesteps: int = field(default=20)
    start_tstep: int = field(default=0)
    end_tstep: int = field(default=1000)
    save_dir: str = field(default='./trak_results_mscoco')


@dataclass
class OtherConfig:
    dataset_name: str = field(default="mscoco")
    sample_size: int = field(default=128)
    n_channels: int = field(default=4)
    ckpt_dir: str = field(default="./mscoco_checkpoints")
    batch_size: int = field(default=32)
    model_id: int = field(default=-1)


def load_checkpoints_from_dir(ckpt_dir, model_id) -> list:
    ckpt_dir = sorted(list(Path(ckpt_dir).iterdir()))
    try:
        ckpt_path = ckpt_dir[model_id].joinpath('checkpoints/checkpoint-200/pytorch_model.bin')
        print(f'Loading checkpoint from {ckpt_path}')
        return torch.load(ckpt_path)
    except:  # noqa
        raise FileNotFoundError(f'Failed to load checkpoint from {ckpt_path}')


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((TrakConfig, OtherConfig))
    trak_config, config = parser.parse_args_into_dataclasses()

    model = get_mscoco_model(config.sample_size, config.n_channels)
    model.eval()

    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="text_encoder").cuda()

    loader_train = get_mscoco_loader(config.batch_size, split="train")

    ckpt = load_checkpoints_from_dir(config.ckpt_dir, config.model_id)

    task = DiffusionModelOutput(conditional=trak_config.conditional_diffusion,
                                latent=trak_config.latent_diffusion)

    print(f'Loader length {len(loader_train.dataset)}')
    traker = TRAKer(model=model,
                    task=task,
                    gradient_computer=DiffusionGradientComputer,
                    proj_dim=trak_config.proj_dim,
                    train_set_size=len(loader_train.dataset),
                    save_dir=trak_config.save_dir,
                    load_from_save_dir=(config.model_id == 0),
                    device='cuda')

    traker.gradient_computer._are_we_featurizing = True
    traker.task._are_we_featurizing = True

    traker.load_checkpoint(ckpt, model_id=config.model_id)

    for batch in tqdm(loader_train, desc='Computing TRAK embeddings...'):
        # batch will consist of [image, label, timestep]
        imgs, captions = batch

        current_bs = len(imgs)

        tokens = [tokenizer(caption,
                            max_length=tokenizer.model_max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt")['input_ids'].cuda()
                  for caption in captions]

        # 5 captions per image, WLOG choosing the first one
        caption_inds = np.random.randint(low=0, high=5, size=current_bs)
        tokens = [tkn[caption_inds[i]] for i, tkn in enumerate(tokens)]

        tokens = torch.stack(tokens)
        states = text_encoder(tokens)[0].unsqueeze(1)
        # import ipdb; ipdb.set_trace()

        batch = [torch.stack(imgs).half().cuda(), states]

        # we can fix timesteps to be evenly spaced instead, if we want
        timesteps = np.random.choice(np.arange(trak_config.start_tstep,
                                               trak_config.end_tstep),
                                     size=[current_bs, trak_config.num_timesteps],
                                     replace=True)

        batch.append(torch.tensor(timesteps).cuda())

        traker.featurize(batch=batch, num_samples=current_bs)

    traker.finalize_features(model_ids=[config.model_id])
