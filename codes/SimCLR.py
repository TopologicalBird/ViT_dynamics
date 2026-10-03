import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import timm
import numpy as np
from torchvision import transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.models.vision_transformer import VisionTransformer

seed = 42

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

###############################################################
# Data Augmentation
###############################################################

class SimCLRTransform:

    def __init__(self, image_size=224):

        color_jitter = transforms.ColorJitter(
            0.4,0.4,0.4,0.1
        )

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

###############################################################
# Dataset
###############################################################

train_dataset = STL10(
    root="./data",
    split="unlabeled",
    download=True,
    transform=SimCLRTransform(224)
)

loader = DataLoader(
    train_dataset,
    batch_size=200,
    shuffle=True,
    num_workers=10,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True
)

###############################################################
# SimCLR Model
###############################################################

class SimCLR(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=0,      
            embed_dim=256,      
            depth=4,         
            num_heads=4,     
            mlp_ratio=4.0,
            qkv_bias=True,
        )

        feature_dim = self.encoder.num_features

        self.projector = nn.Sequential(
            nn.Linear(feature_dim,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,128)
        )

    def forward(self,x):

        h = self.encoder(x)
        z = self.projector(h)

        z = F.normalize(z,dim=1)

        return h,z

###############################################################
# NT-Xent Loss
###############################################################

class NTXentLoss(nn.Module):

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self,z1,z2):

        batch_size = z1.size(0)

        z = torch.cat([z1,z2],dim=0)

        similarity = torch.matmul(z,z.T)

        mask = torch.eye(
            2*batch_size,
            dtype=torch.bool,
            device=z.device
        )

        similarity = similarity / self.temperature

        similarity.masked_fill_(mask,-1e9)

        positives = torch.cat([
            torch.diag(similarity,batch_size),
            torch.diag(similarity,-batch_size)
        ])

        denominator = torch.logsumexp(similarity,dim=1)

        loss = -(positives-denominator)

        return loss.mean()

from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimCLR().to(device)

criterion = NTXentLoss(temperature=0.2)

optimizer = AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=50
)

epochs = 50
loss_list=[]
for epoch in range(epochs):

    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{epochs}]")

    for i, ((x1, x2), _) in enumerate(pbar):

        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        optimizer.zero_grad()

        _, z1 = model(x1)
        _, z2 = model(x2)

        loss = criterion(z1, z2)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{total_loss/(i+1):.4f}"
        )
    scheduler.step()

    avg_loss = total_loss / len(loader)
    loss_list.append(avg_loss)
    print(f"Epoch [{epoch+1:3d}/{epochs}] Loss: {avg_loss:.4f}")

loss_list=np.array(loss_list)