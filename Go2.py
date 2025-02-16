# 文件2: neural_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x += residual
        return F.relu(x)

class DynamicPolicyHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, in_channels//2, 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels//2, 5, padding=2)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, 1)
        )

    def forward(self, x):
        x3 = F.relu(self.conv3(x))
        x5 = F.relu(self.conv5(x))
        x = torch.cat([x3, x5], dim=1)
        return self.fusion(x).view(x.size(0), -1)

class ProNetwork(nn.Module):
    def __init__(self, blocks=40, channels=512):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(17, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        
        # 动态策略头
        self.policy_head = DynamicPolicyHead(channels)
        
        # 价值网络
        self.value_net = nn.Sequential(
            nn.Conv2d(channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        # 初始化参数
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_net(x)
        return policy, value

    def predict(self, board):
        with torch.no_grad():
            features = torch.from_numpy(board.get_features()).unsqueeze(0).float()
            policy, value = self.forward(features)
            return F.softmax(policy[0], dim=0).numpy(), value.item()