import sys
import random

from src.datasets_sketch2img import FashionMNISTDataset
from src.trainer_ddpm_sketch2img import Trainer

project_name = "ddpm_train_test"
run_name = "s2i_fmnist_30epoch_lr_-1" + "_" + str(random.randint(0, 1000))
data_dir = "data/FashionMNIST"
pretrained_model_path = "model/init_s2i_fmnist_5epochs"
save_path = "model/" + run_name

train_dataset_rate = 1
image_log_steps = 400
num_epochs = 30
batch_size = 128
grad_accumulation_steps = 1
lr = 1e-3
save_pipe_steps = image_log_steps

dataset = FashionMNISTDataset(data_dir)
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
    save_pipe_steps=image_log_steps,
)

trainer.train()
