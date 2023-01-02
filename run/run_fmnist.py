import sys
import random

sys.path.append(".")  # couldn't import in local without this
from sketch2img_datasets import FashionMNISTDataset
from ddpm_sketch2img_trainer import Trainer

project_name = "ddpm_train_test"
run_name = "s2i_fmnist_10epoch" + "_" + str(random.randint(0, 1000))
data_dir = "data/FashionMNIST"
pretrained_model_path = "model/init_s2i_fmnist_5epochs"
save_path = "model/" + run_name

train_dataset_rate = 1
image_log_steps = 100
num_epochs = 10
batch_size = 128
lr = 1e-5
grad_accumulation_steps = 5

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
