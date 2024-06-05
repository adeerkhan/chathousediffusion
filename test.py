from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, seed_torch
import os
import pickle


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    results_folder = "./results/text19"
    train_num_workers = 0
    with open(os.path.join(results_folder, "params.pkl"), "rb") as f:
        params = pickle.load(f)

    model = Unet(**params["unet_dict"])

    diffusion = GaussianDiffusion(model, **params["diffusion_dict"])

    trainer = Trainer(
        diffusion,
        "../chathousediffusion/data/0531/image",
        "../chathousediffusion/data/0531/mask",
        "../chathousediffusion/data/0531/text",
        **params["trainer_dict"],
        results_folder=results_folder,
        train_num_workers=train_num_workers
    )

    trainer.val(load_model=5)
