from diffusers import DiffusionPipeline
import torch
from torchvision import transforms
from tqdm import tqdm


class DDPMSketch2ImgPipeline(DiffusionPipeline):
    # TODO: Move transforms to another class

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, sketch, num_inference_step=1000):
        # sketch : PIL
        # returl : PIL

        sketch = transforms.functional.pil_to_tensor(sketch).float()
        sketch = self.normalize(sketch).to(self.device)
        sketch = sketch.unsqueeze(0)

        image = self.sample(sketch, num_inference_step)

        image = image.squeeze(0)
        image = self.denormalize(image).cpu().to(torch.uint8)

        image = transforms.functional.to_pil_image(image)
        return image

    def sample(self, transformed_sketch, num_inference_step):
        # Is this the right place to set timesteps?
        self.scheduler.set_timesteps(num_inference_step, device=self.device)

        s = transformed_sketch.shape
        # Assume image's channels == out_channels
        image = torch.randn((s[0], self.unet.config["out_channels"], s[2], s[3])).to(
            self.device
        )

        for t in tqdm(self.scheduler.timesteps):
            model_input = torch.concat([image, transformed_sketch], dim=1).to(
                self.device
            )
            model_output = self.unet(model_input, t).sample
            image = self.scheduler.step(model_output, t, image).prev_sample

        return image

    def normalize(self, x):
        assert x.dtype == torch.float
        # map x to -1 < x < 1
        # I'm doing normalization with zero understanding :o
        x = x / 255.0
        x = transforms.Normalize([0.5], [0.5])(x)
        return x

    def denormalize(self, x):
        assert x.dtype == torch.float
        x = x * 0.5 + 0.5  # map from (-1, 1) back to (0, 1)
        x = x * 255.0
        return x
