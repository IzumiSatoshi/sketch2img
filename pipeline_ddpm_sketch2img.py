from diffusers import DiffusionPipeline
import torch
from torchvision import transforms
from tqdm import tqdm


class DDPMSketch2ImgPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, sketch, num_inference_step=10):
        # sketch : (bs, 1, x, x)

        self.scheduler.set_timesteps(num_inference_step, device=self.device)

        sketch = torch.from_numpy(sketch).float()
        sketch = self.normalize(sketch).to(self.device)
        s = sketch.shape
        # Assume image's channels == out_channels
        image = torch.randn((s[0], self.unet.config["out_channels"], s[2], s[3])).to(
            self.device
        )

        for t in tqdm(self.scheduler.timesteps):
            model_input = torch.concat([image, sketch], dim=1).to(self.device)
            model_output = self.unet(model_input, t).sample
            image = self.scheduler.step(model_output, t, image).prev_sample

        image = self.denormalize(image).cpu().int().numpy()
        return image

    def normalize(self, x):
        # map x to -1 < x < 1
        # I'm doing normalization with zero understanding :o
        x /= 255
        x = transforms.Normalize([0.5], [0.5])(x)
        return x

    def denormalize(self, x):
        x = x * 0.5 + 0.5  # map from (-1, 1) back to (0, 1)
        x = x * 255
        return x
