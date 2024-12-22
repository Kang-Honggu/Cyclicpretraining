import torch
import torch.nn as nn
import timm

class Resnet50_ark(nn.Module):
    def __init__(self, num_classes_list):
        super(Resnet50_ark, self).__init__()
        # Create the base ResNet50d model
        self.model = timm.create_model('resnet50')
        self.projector = None
        #self.projector = nn.Linear(self.model.num_features,1376)
        #self.projector = nn.Sequential(nn.Linear(self.model.num_features,1376), nn.ReLU(inplace=True),nn.Linear(1376,1376))
        #self.model.num_features = 1376
        self.avg_pooling = nn.AvgPool2d(7)
        self.flatten = nn.Flatten(1)
        #self.batchnorm = nn.BatchNorm1d(num_features=self.model.num_features)
        # Multi-task heads
        self.omni_heads = []
        for num_classes in num_classes_list:
            self.omni_heads.append(nn.Linear(self.model.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)
        
    def forward(self, x, head_n=None):
        x = self.model.forward_features(x)
        x = self.avg_pooling(x)
        x = self.flatten(x)
        if self.projector:
            x = self.projector(x)
        #x = self.batchnorm(x)
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]

    def generate_embeddings(self, x, after_proj=True):
        x = self.model.forward_features(x)
        if self.projector and after_proj:
            x = self.projector(x)
        return x


def save_checkpoint(state,filename='model'):
    torch.save(state, filename + '.pth.tar')