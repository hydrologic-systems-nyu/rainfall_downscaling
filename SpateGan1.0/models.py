import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        device = x.device
        batch, channels, time, height, width = x.shape

        mask_shape = (batch, channels, 1, height, width)
        mask = torch.bernoulli(torch.ones(mask_shape, device=device) * (1 - self.p))
        mask = mask.repeat(1, 1, time, 1, 1) / (1 - self.p)
        
        return x * mask


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_layer_norm: bool = True,
                 stride: int = 1, padding_type: Optional[bool] = None):
        super().__init__()

        padding = 0 if padding_type else 1
        self.use_layer_norm = use_layer_norm
        self.padding_type = padding_type

        self.padding_layer = nn.ReflectionPad3d(1) if padding_type else None

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=padding, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=False)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=padding, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=False)

        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.adjust_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1,
                                         stride=stride, bias=False)
            self.adjust_norm = nn.InstanceNorm3d(out_channels)
        else:
            self.adjust_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if self.padding_layer:
            x = self.padding_layer(x)

        out = self.conv1(x)
        if self.use_layer_norm:
            out = self.norm1(out)
        out = self.relu(out)

        if self.padding_layer:
            out = self.padding_layer(out)

        out = self.conv2(out)
        if self.use_layer_norm:
            out = self.norm2(out)

        if self.adjust_conv:
            residual = self.adjust_conv(residual)
            residual = self.adjust_norm(residual)

        out += residual
        return self.relu(out)


class Interpolate(nn.Module):
    def __init__(self, scale_factor: tuple, mode: str = 'trilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)




class Generator(nn.Module):
    def __init__(self, filter_size: int = 96):
        super().__init__()

        self.filter_size = filter_size
        self._initialize_layers()
        

    def _initialize_layers(self):
        self.apply(self._init_weights)
        f = self.filter_size

        self.input_pad = nn.ReflectionPad3d((1, 1, 1, 1, 0, 0))

        self.res1 = ResidualBlock3D(1, f, use_layer_norm=False, padding_type=True)
        self.res2 = ResidualBlock3D(f, f, use_layer_norm=False, padding_type=True)
        self.res3 = ResidualBlock3D(f, f, use_layer_norm=True, padding_type=True)

        self.down0 = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(f, f, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )

        self.up0 = Interpolate((2, 2, 2))
        self.res4 = ResidualBlock3D(f, f, padding_type=True)

        self.up1 = Interpolate((1, 2, 2))
        self.res5 = ResidualBlock3D(f, f, padding_type=True)

        self.up2 = Interpolate((3, 1, 1))
        self.res6 = ResidualBlock3D(f, f, padding_type=True)

        self.up3 = Interpolate((1, 3, 3))
        self.res7 = ResidualBlock3D(f, f, padding_type=True)

        self.res8 = ResidualBlock3D(f, f, padding_type=True)
        self.res9 = ResidualBlock3D(f, f, use_layer_norm=False, padding_type=True)

        self.output_conv = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(f, 1, kernel_size=3, padding=0),
            nn.Softplus()
        )
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.res1(x)
        x1 = CustomDropout(p=0.2,)(x1)
        x2_stay = self.res2(x1)
        
        # optional UNET structure
        # x2 = self.down0(x2_stay)
        # x2_stay = x2_stay[:, :, 4:-4, 7:-7, 7:-7] + x2
        # ....
        # ...
        # ..
        
        x2 = self.res3(x2_stay)
        x2 = CustomDropout(p=0.2, )(x2)

        x2 = self.up0(x2)
        x2 = self.res4(x2)

        x2 = self.up1(x2)
        x2 = self.res5(x2)
        x2 = CustomDropout(p=0.2, )(x2)

        x2 = self.up2(x2)
        x2 = self.res6(x2)

        x2 = self.up3(x2)
        x2 = self.res7(x2)

        x2 = self.res8(x2)
        x2 = self.res9(x2)

        output = self.output_conv(x2)

        ### optional final constraint layer, like https://www.nature.com/articles/s42256-022-00540-1 or https://doi.org/10.48550/arXiv.2411.16098
        
        return output
    
    

class Discriminator(nn.Module):
    def __init__(self, filter_size: int = 128):
        super(Discriminator, self).__init__()
        self.apply(self._init_weights)
        f = filter_size
        
        self.int_reflection = nn.ReflectionPad3d((1, 1, 1, 1, 0, 0))

        # HIGH RESOLUTION layers:
        self.conv1 = ResidualBlock3D(1, f, use_layer_norm=False, stride=(1,1,1), )
        self.conv2 = ResidualBlock3D(f, f, use_layer_norm=True, stride=(2,2,2),)
        self.conv3 = ResidualBlock3D(f, f, use_layer_norm=True, stride=(3,2,2), )
        self.conv4 = ResidualBlock3D(f, f, use_layer_norm=True, stride=(1,3,3), )
        self.conv5 = ResidualBlock3D(f, f//2, use_layer_norm=True, stride=(2,2,2), )
        
        
        # LOW RESOLUTION layers
        self.conv1_1 = ResidualBlock3D(1, f//2, use_layer_norm=False, stride=(1),)
        self.conv1_2 = ResidualBlock3D(f//2, f//4, use_layer_norm=False, stride=(2,2,2),)
        
        self.conv_combined = ResidualBlock3D(f//2+f//4, f//2, use_layer_norm=True, stride=(2),)

        # Output convolution
        self.output_conv = nn.Sequential(nn.Conv3d(f//2, 1, kernel_size=3, padding=1))
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    

    def forward(self, x, y):
        
        noise_x = torch.randn(x.size()).cuda() * 0.05
        noise_y = torch.randn(y.size()).cuda() * 0.05
        x = x + noise_x
        y = y + noise_y
        
        
        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)
        
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x1)
        
        xy = torch.cat((y5, x2), dim=1)
        xy = self.conv_combined(xy)
        
        out = self.output_conv(xy)
        
        return out

