from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, seed_torch
import os
import pickle
from PIL import Image
import pandas as pd


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    results_folder = "./results/text21"
    train_num_workers = 0
    with open(os.path.join(results_folder, "params.pkl"), "rb") as f:
        params = pickle.load(f)

    model = Unet(**params["unet_dict"])

    params["diffusion_dict"]["sampling_timesteps"] = 10

    diffusion = GaussianDiffusion(model, **params["diffusion_dict"])

    trainer = Trainer(
        diffusion,
        "../chat_test_data/0605/image",
        "../chat_test_data/0605/mask",
        "../chat_test_data/0605/text",
        **params["trainer_dict"],
        results_folder=results_folder,
        train_num_workers=train_num_workers,
        mode="val",
    )


    seed_torch()
    trainer.val(load_model=59)

    # index = 37
    # mask_path = f"../chathousediffusion/data/0531/mask_test/{index}.png"
    # text_path = "../chathousediffusion/data/0531/text_test/rooms2.csv"
    # mask = Image.open(mask_path)
    # texts = pd.read_csv(text_path)
    # texts = [p for p in zip(texts["0"], texts["1"])]
    # for x in texts:
    #     if int(x[0].replace(".png", "").replace(".json", "").split("/")[-1]) == index:
    #         text = x[1]
    # seed_torch()
    # image = trainer.predict(30, mask, text)
    # image.save(f"image_{index}.png")
    # new_text=text.replace("\"south\"", "\"southwest\"")
    # seed_torch()
    # new_image = trainer.predict(30, mask, new_text)
    # new_image.save(f"image_{index}_new.png")
    