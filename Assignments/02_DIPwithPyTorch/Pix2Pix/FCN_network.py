import torch
import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        ### FILL: add more CONV Layers
        self.conv1 = self._conv_block(3, 16)
        self.conv2 = self._conv_block(16, 32)
        self.conv3 = self._conv_block(32, 64)
        self.conv4 = self._conv_block(64, 128)  # 增加更多卷积层

        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        self.deconv1 = self._deconv_block(128, 64)
        self.deconv2 = self._deconv_block(64, 32)
        self.deconv3 = self._deconv_block(32, 16)
        self.deconv4 = nn.Sequential(  # 增加反卷积层
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 输出RGB值
        )

    def _conv_block(self, in_channels, out_channels, dropout_prob=0.1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob)  # 添加Dropout层
        )

    def _deconv_block(self, in_channels, out_channels, dropout_prob=0.1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob)  # 添加Dropout层
        )

    def forward(self, x):
        # Encoder forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Decoder forward pass
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        output = self.deconv4(x)
        ### FILL: encoder-decoder forward pass
        
        return output
    
