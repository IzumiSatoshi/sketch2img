import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import wandb
from pipeline_ddpm_sketch2img import DDPMSketch2ImgPipeline
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        pretrained_model_name_or_path,
        dataset,
        save_path,
        num_epochs,
        batch_size,
        lr,
        grad_accumulation_steps,
        train_dataset_rate,
        project_name,
        run_name,
        image_log_steps,
        device="cuda",
    ):
        self.save_path = save_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.grad_accumulation_steps = grad_accumulation_steps
        self.train_dataset_rate = train_dataset_rate
        self.image_log_steps = image_log_steps
        self.device = device
        self.global_step = 0

        # split dataset
        dataset_len = int(len(dataset) * train_dataset_rate)
        dataset, _ = torch.utils.data.random_split(
            dataset, [dataset_len, len(dataset) - dataset_len]
        )

        self.dataloader = DataLoader(dataset, batch_size, shuffle=True)

        self.pipe = DDPMSketch2ImgPipeline.from_pretrained(
            pretrained_model_name_or_path
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.pipe.unet.parameters(),
            lr=lr,
        )

        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config={
                "pretrained_model": pretrained_model_name_or_path,
                "save_path": save_path,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning rate": lr,
                "grad_accumulation_steps": grad_accumulation_steps,
                "train_dataset_rate": train_dataset_rate,
                "device": device,
            },
        )

        # initial prints
        print("base model : ", pretrained_model_name_or_path)
        print("trained model will save at : ", save_path)
        print("dataset length :", len(dataset))

        print("saving initial model...")
        self.pipe.save_pretrained(save_path)  # make sure save is successful

    def log_sample(self, sketch):
        assert type(sketch) == torch.Tensor
        assert len(sketch.shape) == 4, "sketch's shape == (bs, c, h, w)"

        sketch = sketch[0].unsqueeze(0)  # only use first sketch

        image = self.pipe.sample(sketch, self.scheduler.num_train_timesteps)

        # to pil
        image = image.squeeze(0)
        image = self.denormalize(image).cpu().to(torch.uint8)
        image = transforms.functional.to_pil_image(image)
        sketch = sketch.squeeze(0)
        sketch = self.denormalize(sketch).cpu().to(torch.uint8)
        sketch = transforms.functional.to_pil_image(sketch)

        self.run.log(
            {"sketch": wandb.Image(sketch), "generated image": wandb.Image(image)},
            step=self.global_step,
        )

        return image

    def calc_weight_abs_avg(self):
        sketch_channels_num = (
            self.pipe.unet.config["in_channels"] - self.pipe.unet.config["out_channels"]
        )

        # Get weight corresponding to sketch channels
        weight = self.pipe.unet.conv_in.weight[:, -sketch_channels_num:]

        weight_abs = torch.abs(weight)
        weight_abs_avg = weight_abs.mean().item()

        return weight_abs_avg

    def train(self):
        self.optimizer.zero_grad()

        for epoch in tqdm(range(self.num_epochs), desc="epoch"):
            for step, (image, sketch) in enumerate(tqdm(self.dataloader), desc="step"):
                self.global_step += 1

                bs = image.shape[0]

                image = image.to(self.device)
                sketch = sketch.to(self.device)
                noise = torch.randn_like(image).to(self.device)

                timesteps = torch.randint(
                    0,
                    self.pipe.scheduler.num_train_timesteps,
                    (bs,),
                    device=self.device,
                ).long()

                # set up model input
                noisy_image = self.pipe.scheduler.add_noise(image, noise, timesteps)
                model_input = torch.cat([noisy_image, sketch], dim=1).to(self.device)

                noise_pred = self.pipe.unet(model_input, timesteps).sample

                loss = F.mse_loss(noise_pred, noise)
                loss.backward(loss)

                # optimizer's step
                if (step + 1) % self.grad_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # logging
                weight_abs_avg = self.calc_weight_abs_avg()

                if self.global_step % self.image_log_steps == 0:
                    self.log_sample(sketch)

                self.run.log(
                    {
                        "weight_abs_avg": weight_abs_avg,
                        "loss": loss.item(),
                        "epoch": epoch,
                    },
                    step=self.global_step,
                )

            # save every epoch
            print("saving model...")
            self.pipe.save_pretrained(self.save_path)
