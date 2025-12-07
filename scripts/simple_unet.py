import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    Upsampling block:
      - upsample encoder feature (in_ch -> in_ch//2)
      - concatenate with skip (skip_ch)
      - DoubleConv(in_ch//2 + skip_ch -> out_ch)
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class SimpleUNet(nn.Module):
    """
    4-level U-Net Autoencoder with 1 channel inp and 1 chann out final layer:
      enc: 1 -> 32 -> 64 -> 128 -> 256
      dec: 256 -> 128 -> 64 -> 32 -> 1
    """
    def __init__(self, in_channels=1, out_channels=1, base_ch=32):
        super().__init__()

        c1 = base_ch           # 32
        c2 = base_ch * 2       # 64
        c3 = base_ch * 4       # 128
        c4 = base_ch * 8       # 256

        # encoder
        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)

        # decoder
        self.up1 = Up(c4, c3, c3)   # 256 -> 128, concat with 128, out 128
        self.up2 = Up(c3, c2, c2)   # 128 -> 64 , concat with 64 , out 64
        self.up3 = Up(c2, c1, c1)   # 64  -> 32 , concat with 32 , out 32

        self.outc = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)       # [B,32,H,W]
        x2 = self.down1(x1)    # [B,64,H/2,W/2]
        x3 = self.down2(x2)    # [B,128,H/4,W/4]
        x4 = self.down3(x3)    # [B,256,H/8,W/8]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)  # [B,1,h,w]
        return logits
