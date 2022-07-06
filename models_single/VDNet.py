import torch
import torch.nn as nn
import math


# -------------------  Basic Model Components  -----------------------  

class FCReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class FcBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features,math.ceil(out_features/4))
        self.fc2 = nn.Linear(math.ceil(out_features/4), math.ceil(out_features/2))
        self.fc3 = nn.Linear(math.ceil(out_features/2),math.ceil(out_features / 4))
        self.fc4 = nn.Linear(math.ceil(out_features / 4), math.ceil(out_features / 2))
        self.fc5 = nn.Linear(math.ceil(out_features / 2), out_features)   
        
    def forward(self,x):
        return self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(x)))))


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class TwoConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvReLU(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvReLU(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1=self.conv1(x)
        x2=self.conv2(x1)
        return x1, x2


class TconvReLU(nn.Module):
    ''' For upsampling '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        stride=stride, output_padding=output_padding, padding=padding,
                                        dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.tconv(x))

    
# -----------------------  Dense Blocks  -----------------------

class TwoConvReLU_Dense1(nn.Module):
    def __init__(self, in_channels1, in_channels2,out_channels):
        super().__init__()
        self.conv1 = ConvReLU(in_channels1, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvReLU(in_channels2, out_channels, kernel_size=3, padding=1)

    def forward(self, x, c1,c2):
        x1 = self.conv1(torch.cat((c1,c2,x),dim=1))
        x2 = self.conv2(torch.cat((c1,c2,x1,x),dim=1))
        return x2
    
    
class TwoConvReLU_Dense2(nn.Module):
    def __init__(self, in_channels1,in_channels2, out_channels):
        super().__init__()
        self.conv1 = ConvReLU(in_channels1, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvReLU(in_channels2, out_channels, kernel_size=3, padding=1)

    def forward(self, x,c1,c2,d1):
        x1 = self.conv1(torch.cat((d1,c1,c2,x),dim=1))
        x2 = self.conv2(torch.cat((d1,c1,c2,x,x1),dim=1))
        return x2


# -----------------------  Complete Model  -------------------------

class VDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = FcBlock(104, 64*64)

        # Left
        self.L_64 = TwoConvReLU(1, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.L_32 = TwoConvReLU(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.L_16 = TwoConvReLU(128, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.L_8 = TwoConvReLU(256, 512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottom
        self.B_4 = TwoConvReLU(512,1024)

        # Right
        self.tconv1 = TconvReLU(1024, 512, kernel_size=2, stride=2)
        self.R_8 = TwoConvReLU_Dense2(1792, 2304 , 512)

        self.tconv2 = TconvReLU(512, 256, kernel_size=2, stride=2)
        self.R_16 = TwoConvReLU_Dense2(896, 1152 , 256)

        self.tconv3 = TconvReLU(256, 128, kernel_size=2, stride=2)
        self.R_32 = TwoConvReLU_Dense2(448, 576, 128)

        self.tconv4 = TconvReLU(128, 64, kernel_size=2, stride=2)
        self.R_64 = TwoConvReLU_Dense1(192, 256, 64)

        # Output
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        xfc = self.fc(x)
        xfc = xfc.view(-1, 1, 64, 64)  # reshape to  1 x 64 x 64
        xc1,xc2=self.L_64(xfc)
        xd1=self.maxpool1(xc2)

        xc3,xc4=self.L_32(xd1)

        xd2=self.maxpool2(xc4)

        xc5,xc6=self.L_16(xd2)

        xd3=self.maxpool3(xc6)

        xc7,xc8=self.L_8(xd3)

        xd4=self.maxpool4(xc8)

        xc9,xc10=self.B_4(xd4)

        xu1=self.tconv1(xc10)

        xc12=self.R_8(xu1,xc7,xc8,xd3)

        xu2 = self.tconv2(xc12)

        xc14 = self.R_16(xu2, xc5, xc6, xd2)

        xu3 = self.tconv3(xc14)

        xc16 = self.R_32(xu3, xc3, xc4, xd1)

        xu4 = self.tconv4(xc16)

        xc18 = self.R_64(xu4, xc1, xc2)

        out = xfc+self.outconv(xc18)

        return out


if __name__ == '__main__':

    x = torch.empty(2, 104)

    model = VDNet()

    y = model(x)

    # results
    print('----------------------------')
    print(x.shape)
    print(y.shape)
    print('----------------------------')
