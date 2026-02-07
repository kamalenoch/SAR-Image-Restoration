import numpy as np
import torch
from metrics.quality import psnr, ssim
from metrics.enl import enl
from metrics.runtime import runtime
from metrics.params import param_count
from cfar.ca_cfar import ca_cfar

def evaluate(model, original, denoised):
    p = psnr(original, denoised)
    s = ssim(original, denoised)
    roi = denoised[50:100, 50:100]
    e = enl(roi)

    det_map = ca_cfar(denoised)
    det_count = det_map.sum()

    rt = runtime(model, torch.tensor(denoised).unsqueeze(0).unsqueeze(0))

    return p, s, e, det_count, rt
