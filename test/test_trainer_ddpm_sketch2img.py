import numpy as np
import torch
import sys

sys.path.append("src")

from datasets_sketch2img import FashionMNISTDataset
from trainer_ddpm_sketch2img import Trainer

import random
import shutil


project_name = "test"
run_name = "trainer_test" + "_" + str(random.randint(0, 100000))
data_dir = "data/FashionMNIST"
pretrained_model_path = "model/init_s2i_fmnist_5epochs"
save_path = "model/" + run_name

train_dataset_rate = 0.0001
image_log_steps = 5
num_epochs = 2
batch_size = 2
grad_accumulation_steps = 1
lr = 1e-3

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

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
)

trainer.train()

shutil.rmtree(save_path)
