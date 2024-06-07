from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
import pickle

def predict(mask,text,repredict=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    results_folder = "./results/text21"
    train_num_workers = 0
    with open(os.path.join(results_folder, "params.pkl"), "rb") as f:
        params = pickle.load(f)

    model = Unet(**params["unet_dict"])

    diffusion = GaussianDiffusion(model, **params["diffusion_dict"])

    trainer = Trainer(
        diffusion,
        "",
        "",
        "",
        **params["trainer_dict"],
        results_folder=results_folder,
        train_num_workers=train_num_workers,
        mode="predict",
        inject_step=48
    )

    image = trainer.predict(35, mask, text, repredict)
    return image