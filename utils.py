import torch
from torchvision import utils
import torchvision
from model import Generator
from tqdm import tqdm
import streamlit as st
import os
from skimage import img_as_ubyte
from bokeh.models.widgets import Div

def open_link(url, new_tab=True):
    """Dirty hack to open a new web page with a streamlit button."""
    # From: https://discuss.streamlit.io/t/how-to-link-a-button-to-a-webpage/1661/3
    if new_tab:
        js = f"window.open('{url}')"  # New tab or window
    else:
        js = f"window.location.href = '{url}'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)

def tensor2image(tensor):
    tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
    return tensor*0.5 + 0.5

def generate(truncation, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        samples = []
        samples_z = []
        for i in range(4):
          sample_z = torch.randn(5, 512, device=device)
          samples_z.append(sample_z)
          sample, _ = g_ema(
              [sample_z], truncation=truncation, truncation_latent=mean_latent
          )
          samples.append(sample)

        utils.save_image(
            torch.cat(samples, 0),
            f"sample/samples.png",
            nrow=5,
            normalize=True,
            range=(-1, 1),
        )
    return samples_z

def generate_single(row, col, samples_z, truncation, g_ema, device, mean_latent):

    sample_z = torch.unsqueeze(samples_z[row-1][col-1], dim=0)
    sample, _ = g_ema(
              [sample_z], truncation=truncation, truncation_latent=mean_latent
          )
    utils.save_image(
            sample,
            f"sample/sample.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )
    return sample


device = 'cuda'

torch.set_grad_enabled(False)

def sefa(idx, degree, latent, eigvec, trunc,g_ema):
    
  direction = degree * eigvec[:, idx].unsqueeze(0)

  img, _ = g_ema(
      [latent + direction],
      truncation=1,
      truncation_latent=trunc,
      input_is_latent=True,
  )

  utils.save_image(
      img,
      f"output.png",
      normalize=True,
      range=(-1, 1),
      nrow=1,
  )
  return latent+direction


def load_model(generator, model_file_path):
    #ensure_checkpoint_exists(model_file_path)
    ckpt = torch.load(model_file_path, map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"], strict=False)
    return generator.mean_latent(50000)
