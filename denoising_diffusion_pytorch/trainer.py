from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import utils

from tqdm.auto import tqdm
from ema_pytorch import EMA


from .fid_evaluation import FIDEvaluation

from .version import __version__

import os

from .utils import exists, has_int_squareroot, divisible_by, num_to_groups
from .dataset import Dataset, collate_fn
from .eval import cal_iou
from itertools import cycle
from .image_process import convert_gray_to_rgb, convert_mult_to_rgb


# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder_image,
        folder_mask,
        folder_text,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder="./results",
        amp=False,
        mixed_precision_type="fp16",
        split_batches=True,
        convert_image_to=None,
        calculate_fid=True,
        inception_block_idx=2048,
        max_grad_norm=1.0,
        num_fid_samples=50000,
        save_best_and_latest_only=False,
        cond_scale=1,
        mask=0.1,
        use_graphormer=True,
        onehot=True
    ):
        super().__init__()

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling
        self.onehot=onehot

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: "L", 3: "RGB", 4: "RGBA"}.get(3)

        # sampling and training hyperparameters

        assert has_int_squareroot(
            num_samples
        ), "number of samples must have an integer square root"
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (
            train_batch_size * gradient_accumulate_every
        ) >= 16, f"your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above"

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.train_ds = Dataset(
            folder_image,
            folder_mask,
            folder_text,
            self.image_size,
            augment_flip=augment_flip,
            convert_image_to=convert_image_to,
            mask=mask,
            onehot=onehot
        )

        self.val_ds = Dataset(
            folder_image + "_test",
            folder_mask + "_test",
            folder_text + "_test",
            self.image_size,
            augment_flip=augment_flip,
            convert_image_to=convert_image_to,
            mask=0,
            onehot=onehot
        )

        assert (
            len(self.train_ds) >= 100
        ), "you should have at least 100 images in your folder. at least 10k images recommended"

        # self.train_ds, self.val_ds = torch.utils.data.random_split(self.ds, [len(self.ds) - train_batch_size, train_batch_size])

        # dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        train_dl = DataLoader(
            self.train_ds,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=8,
            # num_workers=0,
            collate_fn=collate_fn,
        )

        val_dl = DataLoader(
            self.val_ds,
            batch_size=train_batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        self.train_dl = cycle(train_dl)
        # self.val_dl = cycle(val_dl)
        self.val_dl = val_dl

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        self.ema = EMA(
            diffusion_model, beta=ema_decay, update_every=ema_update_every
        )
        self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0



        # FID-score computation

        self.calculate_fid = calculate_fid

        if self.calculate_fid:
            if not is_ddim_sampling:
                pass
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.train_dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx,
            )

        if save_best_and_latest_only:
            assert (
                calculate_fid
            ), "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10  # infinite

        self.save_best_and_latest_only = save_best_and_latest_only
        self.cond_scale = cond_scale
        self.use_graphormer = use_graphormer

    @property
    def device(self):
        return "cuda"

    def save(self, milestone):

        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": (
                None
            ),
            "version": __version__,
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone):
        device = self.device

        data = torch.load(
            str(self.results_folder / f"model-{milestone}.pt"), map_location=device
        )

        model = self.model
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])

        if "version" in data:
            print(f"loading from version {data['version']}")


    def train(self):

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            position=0,
        ) as pbar:

            while self.step < self.train_num_steps:
                # profiler = Profiler()
                # profiler.start()
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    img, feature, text, graphormer_dict, _ = next(self.train_dl)
                    img=img.to(self.device)
                    feature=feature.to(self.device)
                    graphormer_dict={
                        k:v.to(self.device) for k,v in graphormer_dict.items()
                    }

                    if self.use_graphormer:
                        text = None
                    else:
                        graphormer_dict = None
                    loss = self.model(img, feature, text, graphormer_dict)
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()

                    loss.backward()

                pbar.set_description(f"loss: {total_loss:.4f}")

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()


                self.step += 1

                # profiler.stop()
                # profiler.print()
                self.ema.update()

                if self.step != 0 and divisible_by(
                    self.step, self.save_and_sample_every
                ):
                    self.ema.ema_model.eval()

                    with torch.inference_mode():
                        milestone = self.step // self.save_and_sample_every
                        all_images_list = []
                        val_imgs = []
                        val_texts = []
                        val_features = []
                        idxs = []
                        for (
                            val_img,
                            val_feature,
                            val_text,
                            val_graphormer_dict,
                            idx,
                        ) in self.val_dl:
                            if val_img.shape[0] != self.batch_size:
                                batch_size = val_img.shape[0]
                            else:
                                batch_size = self.batch_size
                            val_img=val_img.to(self.device)
                            val_feature=val_feature.to(self.device)
                            val_graphormer_dict={
                                k:v.to(self.device) for k,v in val_graphormer_dict.items()
                            }
                            val_imgs.append(val_img)
                            val_texts.append(val_text)
                            val_features.append(val_feature)
                            idxs.append(idx)
                            if self.use_graphormer:
                                val_text = None
                            else:
                                val_graphormer_dict = None
                            images = self.ema.ema_model.sample(
                                batch_size=batch_size,
                                feature=val_feature,
                                text=val_text,
                                graphormer_dict=val_graphormer_dict,
                                cond_scale=self.cond_scale,
                            )
                            all_images_list.append(images)

                    if not os.path.exists(
                        self.results_folder / f"step-{milestone}"
                    ):
                        os.makedirs(self.results_folder / f"step-{milestone}")
                    micro_iou_list = []
                    macro_iou_list = []
                    for i in range(len(val_imgs)):
                        for j in range(
                            self.batch_size
                            if i != len(val_imgs) - 1
                            else val_imgs[i].shape[0]
                        ):
                            if self.onehot:
                                # new_image = torch.where(
                                #     val_features[i][j] > 0.5,
                                #     0,
                                #     all_images_list[i][j],
                                # )
                                img = convert_mult_to_rgb(all_images_list[i][j], val_features[i][j])
                                val_img= convert_mult_to_rgb(val_imgs[i][j], val_features[i][j])

                            else:
                                new_image = torch.where(
                                    val_features[i][j] > 0.5,
                                    13/17,
                                    all_images_list[i][j],
                                )
                                img = convert_gray_to_rgb(new_image)
                                val_img = convert_gray_to_rgb(val_imgs[i][j])
                                utils.save_image(
                                    new_image,
                                    str(
                                        self.results_folder
                                        / f"step-{milestone}"
                                        / f"sample-{idxs[i][j]}.png"
                                    ),
                                )
                                utils.save_image(
                                    val_imgs[i][j],
                                    str(
                                        self.results_folder
                                        / f"step-{milestone}"
                                        / f"real-{idxs[i][j]}.png"
                                    ),
                                )

                            micro_iou, macro_iou = cal_iou(img, val_img)
                            # print(f"image{idxs[i][j]}-micro_iou: {micro_iou}, macro_iou: {macro_iou}")
                            micro_iou_list.append(micro_iou)
                            macro_iou_list.append(macro_iou)
                            
                            utils.save_image(
                                img,
                                str(
                                    self.results_folder
                                    / f"step-{milestone}"
                                    / f"rgb_sample-{idxs[i][j]}.png"
                                ),
                            )
                            utils.save_image(
                                val_img,
                                str(
                                    self.results_folder
                                    / f"step-{milestone}"
                                    / f"rgb_real-{idxs[i][j]}.png"
                                ),
                            )
                            with open(
                                self.results_folder
                                / f"step-{milestone}"
                                / f"val_text-{idxs[i][j]}.txt",
                                "w",
                            ) as f:
                                f.write(val_texts[i][j])
                            utils.save_image(
                                val_features[i][j],
                                str(
                                    self.results_folder
                                    / f"step-{milestone}"
                                    / f"feature-{idxs[i][j]}.png"
                                ),
                            )
                    micro_iou = sum(micro_iou_list) / len(micro_iou_list)
                    macro_iou = sum(macro_iou_list) / len(macro_iou_list)
                    print(f"micro_iou: {micro_iou}, macro_iou: {macro_iou}")
                    with open(
                        self.results_folder / f"step-{milestone}" / f"iou.txt", "w"
                    ) as f:
                        for i in range(len(micro_iou_list)):
                            f.write(
                                f"image{idxs[i//self.batch_size][i%self.batch_size]}-micro_iou: {micro_iou_list[i]}, macro_iou: {macro_iou_list[i]}\n"
                            )
                        f.write(f"micro_iou: {micro_iou}, macro_iou: {macro_iou}")

                    # whether to calculate fid

                    if self.calculate_fid:
                        fid_score = self.fid_scorer.fid_score()
                    if self.save_best_and_latest_only:
                        if self.best_fid > fid_score:
                            self.best_fid = fid_score
                            self.save("best")
                        self.save("latest")
                    else:
                        self.save(milestone)
                    del all_images_list
                    torch.cuda.empty_cache()
                pbar.update(1)


    def val(self):
        """还没改，先别用"""
        self.load("20")
        self.ema.copy_params_from_model_to_ema()
        self.ema.ema_model.eval()
        with torch.inference_mode():
            all_images_list = []
            val_imgs = []
            val_texts = []
            val_features = []
            idxs = []
            for val_img, val_feature, val_text, val_graphormer_dict, idx in self.val_dl:
                if val_img.shape[0] != self.batch_size:
                    batch_size = val_img.shape[0]
                else:
                    batch_size = self.batch_size
                val_imgs.append(val_img)
                val_texts.append(val_text)
                val_features.append(val_feature)
                idxs.append(idx)
                if self.use_graphormer:
                    val_text = None
                else:
                    val_graphormer_dict = None
                images = self.ema.ema_model.sample(
                    batch_size=batch_size,
                    feature=val_feature,
                    text=val_text,
                    graphormer_dict=val_graphormer_dict,
                    cond_scale=self.cond_scale,
                )
                all_images_list.append(images)

        if not os.path.exists(self.results_folder / f"cond_scale-{self.cond_scale}"):
            os.makedirs(self.results_folder / f"cond_scale-{self.cond_scale}")
        micro_iou_list = []
        macro_iou_list = []
        for i in range(len(val_imgs)):
            for j in range(
                self.batch_size if i != len(val_imgs) - 1 else val_imgs[i].shape[0]
            ):
                micro_iou, macro_iou = cal_iou(all_images_list[i][j], val_imgs[i][j])
                print(
                    f"image{idxs[i][j]}-micro_iou: {micro_iou}, macro_iou: {macro_iou}"
                )
                micro_iou_list.append(micro_iou)
                macro_iou_list.append(macro_iou)
                utils.save_image(
                    all_images_list[i][j],
                    str(
                        self.results_folder
                        / f"cond_scale-{self.cond_scale}"
                        / f"sample-{idxs[i][j]}.png"
                    ),
                )
                utils.save_image(
                    val_imgs[i][j],
                    str(
                        self.results_folder
                        / f"cond_scale-{self.cond_scale}"
                        / f"real-{idxs[i][j]}.png"
                    ),
                )
                with open(
                    self.results_folder
                    / f"cond_scale-{self.cond_scale}"
                    / f"val_text-{idxs[i][j]}.txt",
                    "w",
                ) as f:
                    f.write(val_texts[i][j])
                utils.save_image(
                    val_features[i][j],
                    str(
                        self.results_folder
                        / f"cond_scale-{self.cond_scale}"
                        / f"feature-{idxs[i][j]}.png"
                    ),
                )
        micro_iou = sum(micro_iou_list) / len(micro_iou_list)
        macro_iou = sum(macro_iou_list) / len(macro_iou_list)
        print(f"micro_iou: {micro_iou}, macro_iou: {macro_iou}")
        with open(
            self.results_folder / f"cond_scale-{self.cond_scale}" / f"iou.txt", "w"
        ) as f:
            for i in range(len(micro_iou_list)):
                f.write(
                    f"image{idxs[i//self.batch_size][i%self.batch_size]}-micro_iou: {micro_iou_list[i]}, macro_iou: {macro_iou_list[i]}\n"
                )
            f.write(f"micro_iou: {micro_iou}, macro_iou: {macro_iou}")
