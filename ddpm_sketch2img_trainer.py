import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
        device="cuda",
    ):
        self.save_path = save_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.grad_accumulation_steps = grad_accumulation_steps
        self.train_dataset_rate = train_dataset_rate
        self.device = device

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

    def log_sample(self, sketch):
        image = self.pipe.sample(sketch, self.scheduler.num_train_timesteps)
        self.run.log()

    def train(self):
        self.optimizer.zero_grad()

        for epoch in range(self.num_epochs):
            print(epoch)
            for step, (image, sketch) in enumerate(tqdm(self.dataloader)):
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
                self.run.log({"loss": loss.item()})
