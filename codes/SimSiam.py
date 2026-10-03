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

seed = 127

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
# SimSiam Model
###############################################################

class SimSiam(nn.Module):

    def __init__(self):

        super().__init__()


        ########################################################
        # Encoder
        ########################################################

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


        ########################################################
        # Projector
        ########################################################

        self.projector = nn.Sequential(

            nn.Linear(
                feature_dim,
                2048
            ),

            nn.BatchNorm1d(2048),

            nn.ReLU(inplace=True),

            nn.Linear(
                2048,
                2048
            ),

            nn.BatchNorm1d(
                2048,
                affine=False
            )
        )


        ########################################################
        # Predictor
        ########################################################

        self.predictor = nn.Sequential(

            nn.Linear(
                2048,
                512
            ),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Linear(
                512,
                2048
            )
        )


    ############################################################
    # Forward
    ############################################################

    def forward(self, x1, x2):

        ########################################################
        # Encoder
        ########################################################

        h1 = self.encoder(x1)
        h2 = self.encoder(x2)


        ########################################################
        # Projector
        ########################################################

        z1 = self.projector(h1)
        z2 = self.projector(h2)


        ########################################################
        # Predictor
        ########################################################

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)


        return h1, h2, z1, z2, p1, p2

###############################################################
# SimSiam Loss
###############################################################

def negative_cosine_similarity(p, z):

    z = z.detach()

    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)

    return -(p * z).sum(dim=1).mean()


def simsiam_loss(p1, z2, p2, z1):

    loss1 = negative_cosine_similarity(
        p1,
        z2
    )

    loss2 = negative_cosine_similarity(
        p2,
        z1
    )

    loss = 0.5 * (loss1 + loss2)

    return loss


###############################################################
# Training
###############################################################
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimSiam().to(device)

criterion = simsiam_loss

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

    pbar = tqdm(
        loader,
        desc=f"Epoch [{epoch+1}/{epochs}]"
    )

    for i, ((x1, x2), _) in enumerate(pbar):

        x1 = x1.to(
            device,
            non_blocking=True
        )

        x2 = x2.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        #######################################################
        # SimSiam Forward
        #######################################################

        _, _, z1, z2, p1, p2 = model(
            x1,
            x2
        )

        #######################################################
        # SimSiam Loss
        #######################################################

        loss = criterion(
            p1,
            z2,
            p2,
            z1
        )

        #######################################################
        # Backward
        #######################################################

        loss.backward()

        optimizer.step()

        #######################################################
        # Logging
        #######################################################

        total_loss += loss.item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{total_loss/(i+1):.4f}"
        )
    scheduler.step()

    avg_loss = total_loss / len(loader)
    loss_list.append(avg_loss)
    print(
        f"Epoch [{epoch+1:3d}/{epochs}] "
        f"Loss: {avg_loss:.4f}"
    )
loss_list=np.array(loss_list)



