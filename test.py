from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, seed_torch
import os
import pickle
from PIL import Image
import pandas as pd


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    results_folder = "./results/text16"
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
        train_num_workers=train_num_workers,
        mode="predict"
    )

    # trainer.val(load_model=5)
    index = 37
    mask_path = f"../chathousediffusion/data/0531/mask_test/{index}.png"
    text_path = "../chathousediffusion/data/0531/text_test/rooms2.csv"
    mask = Image.open(mask_path)
    texts = pd.read_csv(text_path)
    texts = [p for p in zip(texts["0"], texts["1"])]
    for x in texts:
        if int(x[0].replace(".png", "").replace(".json", "").split("/")[-1]) == index:
            text = x[1]
    image = trainer.predict(52, mask, text)
    image.show()
