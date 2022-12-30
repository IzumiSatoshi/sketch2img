import torchvision
import torch
from tqdm import tqdm
from diffusers.pipeline_utils import DiffusionPipeline


class Sketch2ImgPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
        )

    @torch.no_grad()
    def __call__(
        self,
        sketches,
        num_inference_step=10,
    ):
        # input sketch : numpy array(batch_size, 1, 28, 28), 0~255
        # return : numpy array(batch_size, 1, 28, 28), -1 ~ 1
        # TODO: map output to 0 ~ 255

        sketches = torch.from_numpy(sketches).float()
        sketches = self.normalize(sketches)

        self.scheduler.set_timesteps(num_inference_step, device=self.device)

        sketches = sketches.to(self.device)
        samples = torch.randn_like(sketches).to(self.device)

        for t in tqdm(self.scheduler.timesteps):
            x = torch.concat([samples, sketches], dim=1).to(self.device)
            residuals = self.unet(x, t).sample
            samples = self.scheduler.step(residuals, t, samples).prev_sample

        # samples = self.denormalize(samples).cpu().int().numpy()
        samples = samples.cpu().numpy()
        return samples

    def normalize(self, x):
        # map x to -1 < x < 1
        # I'm doing normalization with zero understanding :o
        x /= 255
        x = torchvision.transforms.Normalize(0.5, 0.5)(x)
        return x

    def denormalize(self, x):
        x = x * 0.5 + 0.5  # map from (-1, 1) back to (0, 1)
        x = x * 255
        return x
