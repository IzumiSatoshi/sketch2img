from diffusers import DiffusionPipeline
import torch
from torchvision import transforms
from tqdm import tqdm


class DDPMSketch2ImgPipeline(DiffusionPipeline):
    # TODO: Move transforms to another class

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    def __call__(self, sketch, num_inference_step=1000, tqdm_leave=True):
        # sketch : PIL
        # returl : PIL

        sketch = transforms.functional.pil_to_tensor(sketch).float()
        sketch = self.normalize(sketch).to(self.device)
        sketch = sketch.unsqueeze(0)

        image = self.sample(sketch, num_inference_step, tqdm_leave)

        image = image.squeeze(0)
        image = self.denormalize(image)
        image = self.denormalized_tensor_to_pil(image)

        return image

    def sample(self, transformed_sketch, num_inference_step, tqdm_leave=True):
        assert (
            len(transformed_sketch.shape) == 4
        ), f"(bs, c, h, w) but {transformed_sketch.shape}"

        # Is this the right place to set timesteps?
        self.scheduler.set_timesteps(num_inference_step, device=self.device)

        s = transformed_sketch.shape
        # Assume image's channels == out_channels
        image = torch.randn((s[0], self.unet.config["out_channels"], s[2], s[3])).to(
            self.device
        )

        for t in tqdm(self.scheduler.timesteps, leave=tqdm_leave):
            model_input = torch.concat([image, transformed_sketch], dim=1).to(
                self.device
            )
            with torch.no_grad():
                model_output = self.unet(model_input, t).sample
            image = self.scheduler.step(model_output, t, image).prev_sample

        return image

    def denormalized_tensor_to_pil(self, tensor):
        assert len(tensor.shape) == 3, f"(c, h, w) but {tensor.shape}"

        tensor = tensor.cpu().clip(0, 255).to(torch.uint8)
        pil = transforms.functional.to_pil_image(tensor)
        return pil

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
