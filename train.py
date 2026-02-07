import torch
from torch.utils.data import DataLoader
from model.cnn import LightweightCNN
from data.self_supervised_dataset import SARSelfSupervised

def train_model(images, epochs=50):
    dataset = SARSelfSupervised(images)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LightweightCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.L1Loss()

    model.train()
    for e in range(epochs):
        total = 0
        for x, y in loader:
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {e+1}, Loss {total/len(loader):.5f}")

    return model
