import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

class WHuberLoss(nn.Module):
    def __init__(self):
        super(WHuberLoss, self).__init__()

    def forward(self, input, target):
        return torch.mean(torch.ceil(target // 40) * F.smooth_l1_loss(input, target, reduction='none'))

class WeightMAELoss(nn.Module):
    def __init__(self):
        super(WeightMAELoss, self).__init__()

    def forward(self, input, target):
        return torch.mean(target // 5 * F.l1_loss(input, target, reduction='none'))

class WeightMSELoss(nn.Module):
    def __init__(self):
        super(WeightMSELoss, self).__init__()

    def forward(self, input, target):
        return torch.mean(target // 5 * F.mse_loss(input, target, reduction='none'))
## vgg19
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out        
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):   
        import pdb
        # pdb.set_trace()
        length = x.shape[1]
        loss = 0
        for t in range(length):           
            x_vgg, y_vgg = self.vgg(x[:, t, ...]), self.vgg(y[:, t, ...])
            for i in range(len(x_vgg)):
                loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss