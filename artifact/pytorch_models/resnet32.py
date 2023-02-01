import torch
import torch.nn as nn
import torch.nn.functional as F
class ResNet32(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet32, self).__init__()
        self.convL0_    = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL0_      = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act0       = nn.ReLU()

        self.convL1_    = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL1_      = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act1       = nn.ReLU()
        
        self.convL2_    = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL2_      = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act2       = nn.ReLU()
 
        self.convL3_    = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL3_      = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act3       = nn.ReLU()
        
        self.convL4_    = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL4_      = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act4       = nn.ReLU()

        self.convL5_    = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL5_      = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act5       = nn.ReLU()
        
        self.convL6_    = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL6_      = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act6       = nn.ReLU()

        self.convL7_    = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL7_      = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act7       = nn.ReLU()
        
        self.convL8_    = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL8_      = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act8       = nn.ReLU()

        self.convL9_    = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL9_      = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act9       = nn.ReLU()
        
        self.convL10_   = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL10_     = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act10       = nn.ReLU()

        self.convL11_   = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bnL11_     = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act11       = nn.ReLU()
        
        self.convL12_   = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL12_     = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act12       = nn.ReLU()
        #LambdaLayer()

        self.convL13_   = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL13_     = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act13       = nn.ReLU()
        
        self.convL14_   = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL14_     = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act14       = nn.ReLU()

        self.convL15_   = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL15_     = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act15       = nn.ReLU()
        
        self.convL16_   = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL16_     = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act16       = nn.ReLU()

        self.convL17_   = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL17_     = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act17       = nn.ReLU()
        
        self.convL18_   = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL18_     = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act18       = nn.ReLU()

        self.convL19_   = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL19_     = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act19       = nn.ReLU()
        
        self.convL20_   = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL20_     = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act20       = nn.ReLU()

        self.convL21_   = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bnL21_     = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act21       = nn.ReLU()
        
        self.convL22_   = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL22_     = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act22      = nn.ReLU()
        #LambdaLayer()

        self.convL23_   = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL23_     = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act23       = nn.ReLU()
        
        self.convL24_   = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL24_     = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act24       = nn.ReLU()

        self.convL25_   = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL25_     = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act25       = nn.ReLU()
        
        self.convL26_   = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL26_     = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act26       = nn.ReLU()

        self.convL27_   = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL27_     = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act27       = nn.ReLU()
        
        self.convL28_   = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL28_     = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act28       = nn.ReLU()

        self.convL29_   = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL29_     = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act29       = nn.ReLU()
        
        self.convL30_   = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL30_     = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act30       = nn.ReLU()
        self.avgpoolL31_   = nn.AdaptiveAvgPool2d((1,1))

        self.linearL31_ = nn.Linear(in_features=64, out_features=num_classes, bias=True)
        self.flatten    = nn.Flatten()

    def forward(self, x):
        out = []
        
        x = self.convL0_(x)
        x = self.bnL0_(x)
        x = self.act0(x)
        out.append(x.clone())
        
        #1.1
        skip = x.clone()
        x = self.convL1_(x)
        x = self.bnL1_(x)
        x = self.act1(x)
        out.append(x.clone())
        x = self.convL2_(x)
        x = self.bnL2_(x)
        x = x + skip
        x = self.act2(x)
        out.append(x.clone())

        #1.2
        skip = x.clone()
        x = self.convL3_(x)
        x = self.bnL3_(x)
        x = self.act3(x)
        out.append(x.clone())
        x = self.convL4_(x)
        x = self.bnL4_(x)
        x = x + skip
        x = self.act4(x)
        out.append(x.clone())

        #1.3
        skip = x.clone()
        x = self.convL5_(x)
        x = self.bnL5_(x)
        x = self.act5(x)
        out.append(x.clone())
        x = self.convL6_(x)
        x = self.bnL6_(x)
        x = x + skip
        x = self.act6(x)
        out.append(x.clone())

        #1.4
        skip = x.clone()
        x = self.convL7_(x)
        x = self.bnL7_(x)
        x = self.act7(x)
        out.append(x.clone())
        x = self.convL8_(x)
        x = self.bnL8_(x)
        x = x + skip
        x = self.act8(x)
        out.append(x.clone())

        #1.5
        skip = x.clone()
        x = self.convL9_(x)
        x = self.bnL9_(x)
        x = self.act9(x)
        out.append(x.clone())
        x = self.convL10_(x)
        x = self.bnL10_(x)
        x = x + skip
        x = self.act10(x)
        out.append(x.clone())

        #2.1
        skip = x.clone()
        x = self.convL11_(x)
        x = self.bnL11_(x)
        x = self.act11(x)
        out.append(x.clone())
        x = self.convL12_(x)
        x = self.bnL12_(x)
        x = x + F.pad(skip[:, :, ::2, ::2], (0, 0, 0, 0, 32//4, 32//4), "constant", 0)
        x = self.act12(x)
        out.append(x.clone())

        #2.2
        skip = x.clone()
        x = self.convL13_(x)
        x = self.bnL13_(x)
        x = self.act13(x)
        out.append(x.clone())
        x = self.convL14_(x)
        x = self.bnL14_(x)
        x = x + skip
        x = self.act14(x)
        out.append(x.clone())

        #2.3
        skip = x.clone()
        x = self.convL15_(x)
        x = self.bnL15_(x)
        x = self.act15(x)
        out.append(x.clone())
        x = self.convL16_(x)
        x = self.bnL16_(x)
        x = x + skip
        x = self.act16(x)
        out.append(x.clone())

        #2.4
        skip = x.clone()
        x = self.convL17_(x)
        x = self.bnL17_(x)
        x = self.act17(x)
        out.append(x.clone())
        x = self.convL18_(x)
        x = self.bnL18_(x)
        x = x + skip
        x = self.act18(x)
        out.append(x.clone())

        #2.5
        skip = x.clone()
        x = self.convL19_(x)
        x = self.bnL19_(x)
        x = self.act19(x)
        out.append(x.clone())
        x = self.convL20_(x)
        x = self.bnL20_(x)
        x = x + skip
        x = self.act20(x)
        out.append(x.clone())

        #3.1
        skip = x.clone()
        x = self.convL21_(x)
        x = self.bnL21_(x)
        x = self.act21(x)
        out.append(x.clone())
        x = self.convL22_(x)
        x = self.bnL22_(x)
        x = x + F.pad(skip[:, :, ::2, ::2], (0, 0, 0, 0, 64//4, 64//4), "constant", 0)
        x = self.act22(x)
        out.append(x.clone())

        #3.2
        skip = x.clone()
        x = self.convL23_(x)
        x = self.bnL23_(x)
        x = self.act23(x)
        out.append(x.clone())
        x = self.convL24_(x)
        x = self.bnL24_(x)
        x = x + skip
        x = self.act24(x)
        out.append(x.clone())

        #3.3
        skip = x.clone()
        x = self.convL25_(x)
        x = self.bnL25_(x)
        x = self.act25(x)
        out.append(x.clone())
        x = self.convL26_(x)
        x = self.bnL26_(x)
        x = x + skip
        x = self.act26(x)
        out.append(x.clone())

        #3.4
        skip = x.clone()
        x = self.convL27_(x)
        x = self.bnL27_(x)
        x = self.act27(x)
        out.append(x.clone())
        x = self.convL28_(x)
        x = self.bnL28_(x)
        x = x + skip
        x = self.act28(x)
        out.append(x.clone())

        #3.5
        skip = x.clone()
        x = self.convL29_(x)
        x = self.bnL29_(x)
        x = self.act29(x)
        out.append(x.clone())
        x = self.convL30_(x)
        x = self.bnL30_(x)
        x = x + skip
        x = self.act30(x)
        out.append(x.clone())
        x = self.avgpoolL31_(x)
        
        x = self.flatten(x)
        
        x = self.linearL31_(x)
        out.append(x)
        return out

if __name__ == "__main__":
    net = ResNet32(num_classes=200)
    net.eval();
    torch.manual_seed(1)
    x   = torch.randn(2,3,64,64)
    out = net(x)
    print("len(out):", len(out))
    print("out[-1].shape:", out[-1].shape)