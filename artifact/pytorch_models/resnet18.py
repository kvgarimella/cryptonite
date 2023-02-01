import torch
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.convL0_      = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL0_        = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act0         = nn.ReLU()

        self.convL1_      = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL1_        = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act1         = nn.ReLU()

        self.convL2_      = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL2_        = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act2         = nn.ReLU()

        self.convL3_      = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL3_        = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act3         = nn.ReLU()

        self.convL4_      = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL4_        = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act4         = nn.ReLU()

        self.convL5_      = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bnL5_        = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act5         = nn.ReLU()

        self.convL6_      = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL6_        = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.shortcutL6_  = nn. Sequential(
                            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False),
                            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.act6         = nn.ReLU()

        self.convL7_      = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL7_        = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act7         = nn.ReLU()

        self.convL8_      = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL8_        = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act8         = nn.ReLU()

        self.convL9_      = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bnL9_        = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act9         = nn.ReLU()

        self.convL10_     = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL10_       = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.shortcutL10_ = nn. Sequential(
                            nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False),
                            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.act10        = nn.ReLU()

        self.convL11_     = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL11_       = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act11        = nn.ReLU()

        self.convL12_     = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL12_       = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act12        = nn.ReLU()

        self.convL13_     = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bnL13_       = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act13        = nn.ReLU()

        self.convL14_     =  nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL14_       =  nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.shortcutL14_ =  nn.Sequential(
                             nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False),
                             nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.act14        = nn.ReLU()

        self.convL15_     = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL15_       = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act15        = nn.ReLU()

        self.convL16_     = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL16_       = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act16        = nn.ReLU()
        #self.avgpoolL16_  = nn.AvgPool2d(kernel_size=kernel_size, stride=4, padding=0)
        self.avgpoolL16_  = nn.AdaptiveAvgPool2d(output_size=(1,1))


        self.linearL17_   = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        self.flatten      = nn.Flatten()

    def forward(self, x):
        out = []

        x = self.convL0_(x)
        x = self.bnL0_(x)
        x = self.act0(x)
        skip = x.clone()
        out.append(x.clone())
        
        x = self.convL1_(x)
        x = self.bnL1_(x)
        x = self.act1(x)
        out.append(x.clone())

        x = self.convL2_(x)
        x = self.bnL2_(x)
        x = x + skip
        x = self.act2(x)
        skip = x.clone()
        out.append(x.clone())
        
        x = self.convL3_(x)
        x = self.bnL3_(x)
        x = self.act3(x)
        out.append(x.clone())

        x = self.convL4_(x)
        x = self.bnL4_(x)
        x = x + skip
        x = self.act4(x)
        skip = x.clone()
        out.append(x.clone())
        
        x = self.convL5_(x)
        x = self.bnL5_(x)
        x = self.act5(x)
        out.append(x.clone())

        x = self.convL6_(x)
        x = self.bnL6_(x)
        x = x + self.shortcutL6_(skip)
        x = self.act6(x)
        skip = x.clone()
        out.append(x.clone())

        x = self.convL7_(x)
        x = self.bnL7_(x)
        x = self.act7(x)
        out.append(x.clone())

        x = self.convL8_(x)
        x = self.bnL8_(x)
        x = x + skip
        x = self.act8(x)
        skip = x.clone()
        out.append(x.clone())
        
        x = self.convL9_(x)
        x = self.bnL9_(x)
        x = self.act9(x)
        out.append(x.clone())

        x = self.convL10_(x)
        x = self.bnL10_(x)
        x = x + self.shortcutL10_(skip)
        x = self.act10(x)
        skip = x.clone()
        out.append(x.clone())

        x = self.convL11_(x)
        x = self.bnL11_(x)
        x = self.act11(x)
        out.append(x.clone())

        x = self.convL12_(x)
        x = self.bnL12_(x)
        x = x + skip
        x = self.act12(x)
        skip = x.clone()
        out.append(x.clone())

        x = self.convL13_(x)
        x = self.bnL13_(x)
        x = self.act13(x)
        out.append(x.clone())

        x = self.convL14_(x)
        x = self.bnL14_(x)
        x = x + self.shortcutL14_(skip)
        x = self.act14(x)
        skip = x.clone()
        out.append(x.clone())

        x = self.convL15_(x)
        x = self.bnL15_(x)
        x = self.act15(x)
        out.append(x.clone())
        
        x = self.convL16_(x)
        x = self.bnL16_(x)
        x = x + skip
        x = self.act16(x)
        out.append(x.clone())
        x = self.avgpoolL16_(x)

        x = self.flatten(x)

        x = self.linearL17_(x)
        out.append(x)
        return out

if __name__ == "__main__":
    net = ResNet18(num_classes=200)
    net.eval();
    torch.manual_seed(1)
    x   = torch.randn(2,3,64,64)
    out = net(x)
    print("len(out):", len(out))
    print("out[-1].shape:", out[-1].shape)
