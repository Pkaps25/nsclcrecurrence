import torch.optim as optim
import torch.nn as nn

import os 
import torch.distributed as dist

import torch.nn as nn
import torch

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

torch.cuda.set_device(0)

class VolumeClassifier(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=256, num_classes=2):
        super().__init__()
        self.embedder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, D, H, W)
        B, _, D, H, W = x.shape
        x = x.squeeze(1)  # (B, D, H, W)

        # Reshape to (B * D, 3, H, W)
        x = x.view(B * D, H, W)
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)  # → (B*D, 3, H, W)
        x = self.embedder(x)  # → (B*D, embed_dim)
        x = x.view(B, D, -1)  # → (B, D, embed_dim)

        lstm_out, _ = self.lstm(x)  # → (B, D, hidden_dim)
        final_out = lstm_out[:, -1, :]  # use last LSTM output
        return self.classifier(final_out)
    


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print("HERE")
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print("THERE")

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, val_dl):
    
    setup(rank, world_size)
    model = VolumeClassifier()
    model = FSDP(model, device_id=rank)
    # model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()  # set model to training mode
        running_loss = 0.0

        for volumes, labels in val_dl:
            volumes = volumes.to(rank)
            labels = labels.to(rank)

            optimizer.zero_grad()

            outputs = model(volumes)  # forward pass
            loss = criterion(outputs, labels)  # compute loss

            loss.backward()  # backprop
            optimizer.step()  # update weights

            running_loss += loss.item() * volumes.size(0)

        epoch_loss = running_loss / len(val_dl.dataset)
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    cleanup()


if __name__ == "__main__": 
    from datasets import getNoduleInfoList
    nodules = getNoduleInfoList(['sclc'])
    from datasets import NoduleDataset
    dataset = NoduleDataset(
        nodules, 
        isValSet_bool=True,
        dilate=20,
        resample=[224, 224, 224],
        box_size=[50, 50, 50],
        fixed_size=True
    )

    from torch.utils.data import DataLoader
    val_dl = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
    )
    import torch.multiprocessing as mp 
    mp.spawn(train,
                args=(4, val_dl),
                nprocs=4,
                join=True)
    