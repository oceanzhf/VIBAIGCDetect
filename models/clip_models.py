from .clip import clip
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from numbers import Number
def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor
CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()
        self.k = 256
        self.model, self.preprocess = clip.load(name, device="cpu")
        self.fc_1 = nn.Linear(768, 1024)
        self.relu = nn.ReLU(True)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_3 = nn.Linear(1024, 2*self.k)
        self.decode = nn.Linear(self.k, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        num_sample = 1
        features = self.model.encode_image(x)

        if features.dim() > 2 : features = features.view(features.size(0),-1)

        features = self.dropout(features)
        statistics = self.fc_1(features)
        statistics = self.relu(statistics)
        statistics = self.fc_2(statistics)
        statistics = self.relu(statistics)
        statistics = self.fc_3(statistics)
        mu = statistics[:,:self.k]
        std = F.softplus(statistics[:,self.k:]-5,beta=1)

        encoding = self.reparametrize_n(mu,std,num_sample)
        logit = self.decode(encoding)

        if num_sample == 1 : pass
        elif num_sample > 1 : logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit

    def reparametrize_n(self, mu, std, n=1):
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std