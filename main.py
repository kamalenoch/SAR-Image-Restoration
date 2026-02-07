from data.load_sar import load_sar_images
from dsp.wavelet import wavelet_denoise
from train import train_model
from evaluate import evaluate
import numpy as np

images = load_sar_images("data/raw")
wavelet_imgs = [wavelet_denoise(i) for i in images]

model = train_model(wavelet_imgs)

for img in images[:3]:
    inp = wavelet_denoise(img)
    out = model(
        torch.tensor(inp).unsqueeze(0).unsqueeze(0)
    ).detach().numpy()[0,0]

    p,s,e,d,rt = evaluate(model, img, out)
    print(p,s,e,d,rt)
