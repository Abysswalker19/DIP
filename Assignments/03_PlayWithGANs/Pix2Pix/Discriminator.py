import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_channels=6):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            
            # 第二层卷积
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            # 第三层卷积
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            # 第四层卷积
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(256 * 16 * 16, 256),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 使用 Sigmoid 将输出值映射到 [0, 1]
        )

    def forward(self, image_rgb, image_semantic):
        x = torch.cat([image_rgb,image_semantic],dim=1)
        x = self.model(x) 
        out = self.fc(x)
        return out

