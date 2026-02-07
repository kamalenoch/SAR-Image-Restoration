import time, torch
def runtime(model, x):
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        model(x)
    return (time.time() - t0) * 1000
