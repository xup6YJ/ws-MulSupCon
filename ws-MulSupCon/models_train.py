import torch
import torch.nn as nn
import torchvision
from torchvision.models.densenet import DenseNet121_Weights

def DownSample(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels))

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_stride):
        super(BasicBlock, self).__init__()

        if downsample_stride == None:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.downsample = None
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.downsample = DownSample(in_channels, out_channels, downsample_stride)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        identity = inputs

        x = self.bn1(self.conv1(inputs))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))

        if self.downsample != None:
            identity = self.downsample(identity)
        x = self.relu(x + identity)

        return x
    
class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(BasicBlock(64, 64, None), BasicBlock(64, 64, None))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, (2, 2)), BasicBlock(128, 128, None))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, (2, 2)), BasicBlock(256, 256, None))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, (2, 2)), BasicBlock(512, 512, None))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, 2)
        self.fc = nn.Sequential(
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, 2)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x.reshape(x.shape[0], -1))
        return x

class Bottleneck(nn.Module):
    extend = 4
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels*self.extend, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels*self.extend)
        )
        self.relu = nn.ReLU()
        self.downsample = downsample
        
    def forward(self, inputs):

        identity = inputs
        x = self.block(inputs)
        
        if self.downsample != None:
            identity = self.downsample(inputs)

        x += identity
        x = self.relu(x)
        
        return x
        
class ResNet(nn.Module):
    def __init__(self, Block, layer, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(Block, layer[0], base=64)
        self.layer2 = self.make_layer(Block, layer[1], base=128, stride=2)
        self.layer3 = self.make_layer(Block, layer[2], base=256, stride=2)
        self.layer4 = self.make_layer(Block, layer[3], base=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512*Block.extend, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(512*Block.extend, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, num_classes)
        )
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x.reshape(x.shape[0], -1))
        return x
        
    def make_layer(self, Block, blocks, base, stride=1):
        downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != base*Block.extend:
            downsample = DownSample(self.in_channels, base*Block.extend, stride)
            
        layers.append(Block(self.in_channels, base, downsample=downsample, stride=stride))
        self.in_channels = base*Block.extend
        
        for _ in range(blocks-1):
            layers.append(Block(self.in_channels, base))
            
        return nn.Sequential(*layers)

def ResNet18(num_classes, channels=3):
    return resnet18()
        
def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)

def API_ResNet50(num_classes, channels=3):
    api_model = torchvision.models.resnet50(pretrained=False)
    api_model.conv1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
    api_model.fc = nn.Linear(api_model.fc.in_features, num_classes)
    return api_model

class CSRA(nn.Module): # one basic block 
    def __init__(self, input_dim, num_classes, T, lam):
        super(CSRA, self).__init__()
        self.T = T      # temperature       
        self.lam = lam  # Lambda                        
        self.head = nn.Conv2d(input_dim, num_classes, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x (B d H W)
        # normalize classifier
        # score (B C HxW)
        '''
        input: torch.Size([32, 1024, 7, 7])
        head: torch.Size([32, 14, 7, 7])
        score: torch.Size([32, 14, 7, 7])
        flatten: torch.Size([32, 14, 49])
        '''
        score = self.head(x) / torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0,1) # torch.Size([32, 14, 7, 7])/torch.Size([1, 14, 1, 1])
        score = score.flatten(2)
        base_logit = torch.mean(score, dim=2)

        if self.T == 99: # max-pooling
            att_logit = torch.max(score, dim=2)[0]
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = torch.sum(score * score_soft, dim=2)

        return base_logit + self.lam * att_logit
    
class MHA(nn.Module):  # multi-head attention
    temp_settings = {  # softmax temperature settings
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        6: [1, 2, 3, 4, 5, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(self, num_heads, lam, input_dim, num_classes):
        super(MHA, self).__init__()
        self.temp_list = self.temp_settings[num_heads]
        self.multi_head = nn.ModuleList([
            CSRA(input_dim, num_classes, self.temp_list[i], lam)
            for i in range(num_heads)
        ])

    def forward(self, x):
        logit = 0.
        for head in self.multi_head:
            logit += head(x)
        return logit

def API_DenseNet121_threeheads(num_heads, lam, num_classes, head, medium, tail):
    api_model = torchvision.models.densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1")
    # api_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # api_model.classifier = nn.Linear(api_model.classifier.in_features, num_classes)
    feature_extractor = nn.Sequential(*list(api_model.children())[:-1])
    # classifier = CSRA(api_model.classifier.in_features, num_classes, 1, 0.2)
    # classifier = MHA(num_heads, lam, api_model.classifier.in_features, num_classes)
    # model = nn.Sequential(feature_extractor, classifier)
    head_classifier = MHA(num_heads, lam, api_model.classifier.in_features, head)
    medium_classifier = MHA(num_heads, lam, api_model.classifier.in_features, medium)
    tail_classifier = MHA(num_heads, lam, api_model.classifier.in_features, tail)
    model = nn.ModuleDict({
        "feature_extractor": feature_extractor,
        "head_classifier": head_classifier,
        "medium_classifier": medium_classifier,
        "tail_classifier": tail_classifier
    })
    return model



def API_DenseNet121(num_classes):

    api_model = torchvision.models.densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1")
    # api_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    api_model.classifier = nn.Linear(api_model.classifier.in_features, num_classes)
    
    return api_model


def backbone_pretrain(num_classes, backbone = None):

    if backbone == 'DenseNet121':
        api_model = torchvision.models.densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1")
        api_model.classifier = nn.Linear(api_model.classifier.in_features, num_classes)
    elif backbone == 'resnet50':
        api_model = torchvision.models.resnet50(pretrained=False)
        dim = api_model.fc.in_features
        # delete the last fc layer
        api_model = nn.Sequential(*list(api_model.children())[:-1])
        api_model.classifier = nn.Sequential(nn.Flatten(start_dim = 1), nn.Linear(dim, num_classes))

        # api_model.fc = nn.Linear(api_model.fc.in_features, num_classes)

    return api_model

class Dense_pretrain(nn.Module):  # multi-head attention
    def __init__(self, encoder,  num_classes, enc_fix = False):
        super(Dense_pretrain, self).__init__()
        
        self.encoder = encoder
        self.classifier = nn.Linear(1024, num_classes)
        self.linear_probe = enc_fix

    def forward(self, x):
        if self.linear_probe:
            self.encoder.eval()
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)

        # x = self.encoder(x)
        x = x.flatten(1)
        x = self.classifier(x)

        return x
    
class Res_pretrain(nn.Module):  # multi-head attention
    def __init__(self, encoder,  num_classes, enc_fix = False):
        super(Res_pretrain, self).__init__()
        
        self.encoder = encoder
        self.classifier = nn.Linear(2048, num_classes)
        self.linear_probe = enc_fix

    def forward(self, x):
        if self.linear_probe:
            self.encoder.eval()
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)

        # x = self.encoder(x)
        x = x.flatten(1)
        x = self.classifier(x)

        return x
    
