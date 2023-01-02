from diffusers import DDPMPipeline
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


def train(dataset):
    # define configs
    save_path = ""
    pretrained_model_name_or_path = ""
    num_epochs = 2
    batch_size = 64
    lr = 1e-5
    grad_accumulation_steps = 2
    train_dataset_rate = 0.1
    device = "cuda"
    wandb_project_name = "ddpm_train_test"
    wandb_run_name = None

    # wandb initizalize
    config = dict(
        batch_size=batch_size,
        lr=lr,
        grad_accumulation_steps=grad_accumulation_steps,
        num_epochs=num_epochs,
        device=device,
        train_dataset_rate=train_dataset_rate,
    )
    wandb.init(project="ddpm_train_test", name=wandb_run_name, config=config)

    # load pipe
    pipe = DDPMPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path
    ).to(device)

    # split dataset
    dataset_len = int(len(dataset) * train_dataset_rate)
    dataset, _ = torch.utils.data.random_split(
        dataset, [dataset_len, len(dataset) - dataset_len]
    )

    # train
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(epoch)
        for step, (image, sketch) in enumerate(tqdm(dataloader)):
            bs = image.shape[0]

            # prepare valiables
            image = image.to(device)
            sketch = sketch.to(device)
            noise = torch.randn_like(image).to(device)
            timesteps = torch.randint(
                0, pipe.scheduler.num_train_timesteps, (bs,), device=device
            ).long()

            # set up model input
            noisy_image = pipe.scheduler.add_noise(image, noise, timesteps)
            model_input = torch.cat([noisy_image, sketch], dim=1).to(device)

            # prediction
            noise_pred = pipe.unet(model_input, timesteps).sample

            # backward
            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)

            # optimizer's step
            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # logging
            wandb.log({"loss": loss.item()})
    # save
    pipe.save_pretrained(save_path)

    # finish wandb
    wandb.finish()
