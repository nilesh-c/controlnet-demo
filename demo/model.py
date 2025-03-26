import cv2
import einops
import numpy as np
import random
import torch

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import config

# Helper functions for color space conversions and brightness adjustment
def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

def lab2rgb(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def rgb2yuv(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)

def yuv2rgb(yuv: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

def srgb2lin(s):
    s = s.astype(float) / 255.0
    return np.where(s <= 0.0404482362771082, s / 12.92, np.power(((s + 0.055) / 1.055), 2.4))

def lin2srgb(lin):
    return 255 * np.where(lin > 0.0031308, 1.055 * (np.power(lin, (1.0 / 2.4))) - 0.055, 12.92 * lin)

def get_luminance(linear_image: np.ndarray, luminance_conversion=[0.2126, 0.7152, 0.0722]):
    return np.sum([[luminance_conversion]] * linear_image, axis=2)

def take_luminance_from_first_chroma_from_second(luminance, chroma, mode="lab", s=1):
    assert luminance.shape == chroma.shape, f"{luminance.shape=} != {chroma.shape=}"
    if mode == "lab":
        lab = rgb2lab(chroma)
        lab[:, :, 0] = rgb2lab(luminance)[:, :, 0]
        return lab2rgb(lab)
    if mode == "yuv":
        yuv = rgb2yuv(chroma)
        yuv[:, :, 0] = rgb2yuv(luminance)[:, :, 0]
        return yuv2rgb(yuv)
    if mode == "luminance":
        lluminance = srgb2lin(luminance)
        lchroma = srgb2lin(chroma)
        return lin2srgb(
            np.clip(
                lchroma * ((get_luminance(lluminance) / (get_luminance(lchroma))) ** s)[:, :, np.newaxis],
                0,
                1,
            )
        )

"""
A class to generate synthetic images using the ControlNet model, based on an input image and text prompts. 
The generated image is be conditioned on an edge map created using the Canny detector and various hyperparameters.
"""
class ControlNetImageGenerator:
    """
    Initializes the ControlNet image generator.

    Parameters:
        config_path (str): Path to the configuration file for the ControlNet model.
        weights_path (str): Path to the pre-trained model weights.
        device (str): The device to run the model on, either "cuda" for GPU or "cpu" for CPU.
    """
    def __init__(self, config_path="./models/cldm_v15.yaml", weights_path="./models/control_sd15_canny.pth", device="cuda"):
        self.apply_canny = CannyDetector()
        self.model = create_model(config_path).cpu()
        self.model.load_state_dict(load_state_dict(weights_path, location=device))
        self.model = self.model.to(device)
        self.ddim_sampler = DDIMSampler(self.model)
        self.device = device

    """
    Generates synthetic images from an input image using the ControlNet model and provided text prompts.

    The process involves creating an edge map from the input image, conditioning the model on this edge map 
    and the provided prompts, and using DDIM sampling to generate the image.

    Parameters:
        input_image (np.ndarray): The input image to be used for image generation.
        prompt (str): The main prompt that guides the image generation.
        a_prompt (str): Additional positive prompt that further refines the generation.
        n_prompt (str): Negative prompt that discourages undesirable features in the generated image.
        num_samples (int, optional): The number of images to generate. Default is 1.
        image_resolution (int, optional): The resolution of the input image after resizing. Default is 512.
        ddim_steps (int, optional): Number of steps to use in the DDIM sampler. Default is 10.
        guess_mode (bool, optional): If True, the model uses guess mode for more aggressive control. Default is False.
        strength (float, optional): Strength of control over the generated image. Default is 1.0.
        scale (float, optional): The scale of unconditional guidance. Default is 9.0.
        seed (int, optional): The random seed for reproducibility. Default is -1 (random seed).
        eta (float, optional): A parameter controlling the randomness of the sampling. Default is 0.0.
        low_threshold (int, optional): Low threshold for Canny edge detection. Default is 50.
        high_threshold (int, optional): High threshold for Canny edge detection. Default is 100.
        apply_luminance (bool, optional): If True, applies luminance adjustment based on the original image. Default is True.

    Returns:
        List[np.ndarray]: A list of generated images. The first element is the inverted edge map, followed by the generated images.
    """
    def generate_image(self, input_image: np.ndarray, prompt: str, a_prompt: str, n_prompt: str, 
                       num_samples: int = 1, image_resolution: int = 512, ddim_steps: int = 10, 
                       guess_mode: bool = False, strength: float = 1.0, scale: float = 9.0, 
                       seed: int = -1, eta: float = 0.0, low_threshold: int = 50, high_threshold: int = 100,
                       apply_luminance: bool = True):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            # Create edge map using the Canny detector
            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().to(self.device) / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if getattr(config, "save_memory", False):
                self.model.low_vram_shift(is_diffusing=False)

            cond_text = [prompt + ", " + a_prompt] * num_samples
            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning(cond_text)]}
            un_cond_text = [n_prompt] * num_samples
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning(un_cond_text)]}
            shape = (4, H // 8, W // 8)

            if getattr(config, "save_memory", False):
                self.model.low_vram_shift(is_diffusing=True)

            if guess_mode:
                self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)]
            else:
                self.model.control_scales = [strength] * 13

            samples, intermediates = self.ddim_sampler.sample(
                ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
                unconditional_guidance_scale=scale, unconditional_conditioning=un_cond
            )

            if getattr(config, "save_memory", False):
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            results = [x_samples[i] for i in range(num_samples)]

            # apply luminance adjustment to the generated image using the original image's luminance
            if apply_luminance:
                results[-1] = take_luminance_from_first_chroma_from_second(
                    resize_image(HWC3(input_image), image_resolution),
                    results[-1],
                    mode="lab"
                )

            # Return a list: first element is the inverted edge map, followed by the generated images.
            return [255 - detected_map] + results
