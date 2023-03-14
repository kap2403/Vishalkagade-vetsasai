import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16_NET(nn.Module):
    def __init__(self):
        super(VGG16_NET, self).__init__()
        self.conv1  = nn.Conv2d(in_channels=3,   out_channels=64,  kernel_size=3, padding=1)
        self.conv2  = nn.Conv2d(in_channels=64,  out_channels=64,  kernel_size=3, padding=1)
        self.conv3  = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=3, padding=1)
        self.conv4  = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5  = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6  = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7  = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv8  = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9  = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.maxpool= nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool1= nn.MaxPool2d(kernel_size=3, stride=1,padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 
        self.conv14 = nn.Conv2d(512, 1024,kernel_size=3,padding=6,dilation=6)
        self.conv15 = nn.Conv2d(1024, 1024,kernel_size=1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        head1=x
        x = self.maxpool(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        head2=x
        return head1,head2

class Auxilarylayers(nn.Module):
    def __init__(self):
        super(Auxilarylayers,self).__init__()
        self.conv16 =nn.Conv2d(1024,256,kernel_size=1,padding=0)
        self.conv17 =nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1)
        
        self.conv18 =nn.Conv2d(512,128,kernel_size=1,padding=0)
        self.conv19 =nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)
        
        self.conv20 =nn.Conv2d(256,128,kernel_size=1,padding=0)
        self.conv21 =nn.Conv2d(128,256,kernel_size=3,padding=0)
        
        self.conv22 =nn.Conv2d(256,128,kernel_size=1,padding=0)
        self.conv23 =nn.Conv2d(128,256,kernel_size=3,padding=0)

    def forward(self,head2):
        x =F.relu(self.conv16(head2))
        x =F.relu(self.conv17(x))
        head3=x
        x =F.relu(self.conv18(x))
        x =F.relu(self.conv19(x))
        head4=x
        x =F.relu(self.conv20(x))
        x =F.relu(self.conv21(x))
        head5=x
        x =F.relu(self.conv22(x))
        x =F.relu(self.conv23(x))
        head6=x
        
        return head3,head4,head5,head6
  

class Predection(nn.Module):
    def __init__(self,classes):
        super(Predection,self).__init__()
        self.classes=classes
        self.Head1 = nn.Conv2d(512,4*4,kernel_size=3, padding=1)
        self.Head2 = nn.Conv2d(1024,6*4,kernel_size=3, padding=1)
        self.Head3 = nn.Conv2d(512,6*4,kernel_size=3, padding=1)
        self.Head4 = nn.Conv2d(256,6*4,kernel_size=3, padding=1)
        self.Head5 = nn.Conv2d(256,4*4,kernel_size=3, padding=1)
        self.Head6 = nn.Conv2d(256,4*4,kernel_size=3, padding=1)
        
        self.class1 = nn.Conv2d(512,  4 * classes, kernel_size=3, padding=1)
        self.class2 = nn.Conv2d(1024, 6 * classes, kernel_size=3, padding=1)
        self.class3 = nn.Conv2d(512,  6 * classes, kernel_size=3, padding=1)
        self.class4 = nn.Conv2d(256,  6 * classes, kernel_size=3, padding=1)
        self.class5 = nn.Conv2d(256,  4 * classes, kernel_size=3, padding=1)
        self.class6 = nn.Conv2d(256,  4 * classes, kernel_size=3, padding=1)

    def forward(self,head1,head2,head3,head4,head5,head6):
        #size=head1.size(0)
        box1=self.Head1(head1)
        box1=box1.permute(0,2,3,1).contiguous()
        box1=box1.view(box1.size(0),-1,4)
        
        
        class1 = self.class1(head1)
        class1 = class1.permute(0, 2, 3, 1).contiguous()
        class1 = class1.view(class1.size(0),-1,self.classes)
        
        
        
        box2 = self.Head2(head2)
        box2 = box2.permute(0,2,3,1).contiguous()
        box2 = box2.view(box2.size(0),-1,4)
        
        class2 = self.class2(head2)
        class2 = class2.permute(0, 2, 3, 1).contiguous()
        class2 = class2.view(class2.size(0),-1,self.classes)
        
        box3 = self.Head3(head3)
        box3 = box3.permute(0,2,3,1).contiguous()
        box3 = box3.reshape(box3.size(0),-1,4)
        
        class3 = self.class3(head3)
        class3 = class3.permute(0, 2, 3, 1).contiguous()
        class3 = class3.view(class3.size(0),-1,self.classes)
        
        box4 = self.Head4(head4)
        box4 = box4.permute(0,2,3,1).contiguous()
        box4 = box4.view(box4.size(0),-1,4)
        
        
        class4 = self.class4(head4)
        class4 = class4.permute(0, 2, 3, 1).contiguous()
        class4 = class4.view(class4.size(0),-1,self.classes)
        
        box5 = self.Head5(head5)
        box5 = box5.permute(0,2,3,1).contiguous()
        box5 = box5.view(box5.size(0),-1,4)
        
        
        class5 = self.class5(head5)
        class5 = class5.permute(0, 2, 3, 1).contiguous()
        class5 = class5.view(class5.size(0),-1,self.classes)
        
        
        box6 = self.Head6(head6)
        box6 = box6.permute(0,2,3,1).contiguous()
        box6 = box6.view(box6.size(0),-1,4)
        
        
        class6 = self.class6(head6)
        class6 = class6.permute(0, 2, 3, 1).contiguous()
        class6 = class6.view(class6.size(0),-1,self.classes)
        
        
        boxes = torch.cat([box1,box2,box3,box4,box5,box6],dim=1)
        classess = torch.cat([class1,class2,class3,class4,class5,class6],dim=1)
        return boxes,classess



class ssd(nn.Module):
    def __init__(self,classes):
        super(ssd, self).__init__()
        self.classes = classes
        self.vgg = VGG16_NET()
        self.auxilary = Auxilarylayers()
        self.predection = Predection(classes)
    def forward(self,image):
        head1,head2 = self.vgg(image)
        head3,head4,head5,head6 = self.auxilary(head2)
        boxes,classes = self.predection(head1,head2,head3,head4,head5,head6)
        return boxes,classes