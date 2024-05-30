from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
import random
import torch
import numpy as np


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # model = Unet(
    #     channels= 4,
    #     dim = 64,
    #     out_dim = 3,
    #     dim_mults = (1, 2, 4, 8),
    #     flash_attn = True
    # )

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model = Unet(
        dim=32,
        cond_dim=512,
        dim_mults=(1, 2, 4, 8),
        num_resnet_blocks=3,
        cond_images_channels=1,
        layer_attns=(False, True, True, True),
    )

    # model = Unet(
    #     dim = 32,
    #     cond_dim = 64,
    #     dim_mults = (1, 2, 4, 8),
    #     cond_images_channels = 1,
    #     layer_attns = (False, False, False, True),
    #     layer_cross_attns= (False, False, False, True),
    # )

    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=1000,  # number of steps
        sampling_timesteps=50,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        cond_drop_prob=0.1
    )

    trainer = Trainer(
        diffusion,
        "data/new/image",
        "data/new/mask",
        "data/new/text",
        train_batch_size=32,
        train_lr=8e-5,
        train_num_steps=100000,  # total training steps
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        calculate_fid=False,  # whether to calculate fid during training
        save_and_sample_every=1000,
        augment_flip=False,
        results_folder="./results/text8",
        cond_scale=1,
    )

    trainer.train()
