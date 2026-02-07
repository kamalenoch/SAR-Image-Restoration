from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def psnr(ref, out):
    return peak_signal_noise_ratio(ref, out, data_range=1.0)

def ssim(ref, out):
    return structural_similarity(ref, out, data_range=1.0)
