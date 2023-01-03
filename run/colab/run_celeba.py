import sys
import random

sys.path.append("./src")  # running on /sketch2img
from datasets_sketch2img import CelebaDataset
from trainer_ddpm_sketch2img import Trainer

project_name = "sketch2img_celeba_ddpm"
run_name = "1epoch" + "_" + str(random.randint(0, 1000))
data_dir = "/content/download/celeba_hq_256"
pretrained_model_path = (
    "/content/drive/MyDrive/Project/sketch2img/model/init_s2i_celeba"
)
save_path = "/content/drive/MyDrive/Project/sketch2img/model/celeba_" + run_name

train_dataset_rate = 1
image_log_steps = 500
num_epochs = 1
batch_size = 5
grad_accumulation_steps = 1
lr = 1e-4
save_pipe_steps = image_log_steps

dataset = CelebaDataset(data_dir)
trainer = Trainer(
    pretrained_model_name_or_path=pretrained_model_path,
    dataset=dataset,
    save_path=save_path,
    num_epochs=num_epochs,
    batch_size=batch_size,
    lr=lr,
    grad_accumulation_steps=grad_accumulation_steps,
    train_dataset_rate=train_dataset_rate,
    project_name=project_name,
    run_name=run_name,
    image_log_steps=image_log_steps,
    save_pipe_steps=save_pipe_steps,
)

trainer.train()
