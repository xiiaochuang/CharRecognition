from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
        						nn.Conv2d(1,16,kernel_size=3) ,
                                nn.BatchNorm2d(16) ,
                                nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
        						nn.Conv2d(16,32,kernel_size=3) ,
                                nn.BatchNorm2d(32) ,
                                nn.ReLU(inplace=True) ,
                                nn.MaxPool2d(kernel_size=2 , stride=2))

        self.layer3 = nn.Sequential(
        						nn.Conv2d(32,64,kernel_size=3) ,
                                nn.BatchNorm2d(64) ,
                                nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
        						nn.Conv2d(64,128,kernel_size=3) ,
                                nn.BatchNorm2d(128) ,
                                nn.ReLU(inplace=True) ,
                                nn.MaxPool2d(kernel_size=2 , stride=2))

        self.fc = nn.Sequential(nn.Linear(128*4*4,1024) ,
                                nn.ReLU(inplace=True) ,
                                nn.Linear(1024,128) ,
                                nn.ReLU(inplace=True) ,
                                nn.Linear(128,10) )
    def forward( self , x):
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = x.view(x.size(0) , -1)
        x = x.reshape(x.size(0) , -1)
        fc_out = self.fc(x)
        return fc_out

