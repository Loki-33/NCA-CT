import torch
import numpy as np 
import torch.nn as nn 
from datasets import load_dataset 
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import imageio 
import os 
from tqdm import trange 
from tqdm import tqdm

class NCA(nn.Module):
    def __init__(self, grid_size, hidden_channels, fire_rate):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_channels = hidden_channels
        self.fire_rate= fire_rate
        self.total_channels = hidden_channels+1
        # (identity, sobel_x, sobel_y) * total_channels
        self.perception_channels = self.total_channels * 3
        
        self.update_net = nn.Sequential(
            nn.Conv2d(self.perception_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, self.total_channels, kernel_size=1)
        )

        self.reg_perception_filters()

    def reg_perception_filters(self):
        identity = torch.tensor([[0., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 0.]])
        sobel_x = torch.tensor([[1., 0., 1.],
                                [2., 0., -2.],
                                [1., 0., -1.]]) / 8.0

        sobel_y = torch.tensor([[1., 2., 1.],
                                [0., 0., 0.],
                               [-1., -2., -1.]]) / 8.0
        filters = torch.stack([identity, sobel_x, sobel_y])[:, None, :, :]
        self.register_buffer('perception_filters', filters)

    def initialize_grid(self, batch_size):
        grid = torch.zeros((batch_size, self.total_channels, self.grid_size, self.grid_size))
        center = self.grid_size//2 
        grid[:,:,  center-1:center+2, center-1:center+2] = torch.randn(batch_size, self.total_channels, 3, 3)*0.1 
        return grid 

    def percieve(self, grid):
        batch_size = grid.size(0)
        perception = []
        for i in range(self.total_channels):
            channel = grid[:, i:i+1, :, :]
            filtered = F.conv2d(channel, self.perception_filters, padding=1)
            perception.append(filtered)

        perception = torch.cat(perception, dim=1)
        return perception

    def apply_stochastic_mask(self, update):
        mask = (torch.randn_like(update[:, :1, :, :])<self.fire_rate).float()
        return update*mask 

    def forward(self, grid):
        perception = self.percieve(grid)
        update = self.update_net(perception)
        update = self.apply_stochastic_mask(update)
        new_grid = grid + update

        new_grid = torch.clamp(new_grid, -2.0, 2.0)
        return new_grid

    def get_visible_channel(self, grid):
        return grid[:, 0:1, :, :]


@torch.no_grad()
def rollout(nca, steps=128, batch_size=1):
    grid = nca.initialize_grid(batch_size)
    frames = []
    for t in range(steps):
        grid = nca(grid)
        if t%4 == 0:
            frames.append(nca.get_visible_channel(grid))
    return frames 

@torch.no_grad()
def damage(nca, steps=128):
    grid = nca.initialize_grid(1)
    for _ in range(64):
        grid= nca(grid)

    grid[:, :, 10:18, 10:18] = 0.0 

    frames = []
    for _ in range(steps):
        grid = nca(grid)
        frames.append(nca.get_visible_channel(grid))

    return frames 

def save_frames(frames, folder='nca_frames'):
    os.makedirs(folder, exist_ok=True)
    for i, frame in enumerate(frames):
        plt.imsave(f'{folder}/frame_{i:03d}.png', frame.squeeze(0).squeeze(0), cmap='gray')

def save_gifs(frames, filename='nca_growth.gif'):
    imgs = [(frame[0,0].numpy()*255).astype('uint8') for frame in frames]
    imageio.mimsave(filename, imgs, duration=0.1)

def transform(example):
    img = example['image']
    img = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
    return {'image': img, 'label': example['label']}

if __name__ == '__main__':
    train_data = load_dataset('mnist')['train']
    train_data = train_data.filter(lambda x: x['label']==3)
    train_data = train_data.with_transform(transform)
    batch_size=8
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    nca = NCA(grid_size=28, hidden_channels=16, fire_rate=0.5)
    optimizer = torch.optim.Adam(nca.parameters(), lr=1e-3)
    
    steps_min = 32
    steps_max = 64 

    for epoch in range(10):
        loader = tqdm(train_loader, desc=f'Epoch{epoch+1}')
        for i, batch in enumerate(loader):
            target = batch['image'].unsqueeze(1)
            grid = nca.initialize_grid(batch_size=target.size(0))
            steps = torch.randint(steps_min, steps_max, (1,)).item()
            for _ in range(steps):
                grid = nca(grid)

            pred = nca.get_visible_channel(grid)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nca.parameters(), 1.0)
            optimizer.step()
            loader.set_postfix(loss=loss.item())
            loader.update(1)
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")
        if (epoch+1)%5==0:
            torch.save(nca, 'nca_model.pth')
    frames = rollout(nca, steps=32)
    save_gifs(frames)
