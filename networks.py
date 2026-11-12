"""Image networks ("feature extractors") the policy can use instead of SB3's default NatureCNN.

Kept in their own module so saved models refer to `networks.ImpalaCNN` and load the same way from
train_treechop.py, bc_train.py and evaluate.py.

NatureCNN (default, "nature"): 3 conv layers -> 512 features, ~0.6M parameters.
ImpalaCNN ("impala"): the residual network from the IMPALA paper (Espeholt et al. 2018), widened.
  3 stages, each: 3x3 conv -> 3x3 max-pool (stride 2) -> 2 residual blocks (two 3x3 convs + skip).
  64x64 image -> 32x32 -> 16x16 -> 8x8, then 512 features. Much deeper than NatureCNN (13 conv
  layers instead of 3) and ~4x the parameters; it does far better than NatureCNN on games like
  Procgen, and VPT also uses a ResNet of this kind.
"""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResBlock(nn.Module):
    """x + scale * conv(relu(conv(relu(x)))), with `scale` learned and starting at 0 ("ReZero"), so each
    block starts out passing its input straight through. Without it, SB3's re-initialisation of every
    conv layer (orthogonal, gain sqrt(2)) makes the activations grow ~20x through the 6 blocks, and the
    first updates killed ~3/4 of the features: the Nov 11 run learned nothing (it only ever predicted
    the most common action). SB3 only re-initialises Conv/Linear weights, so `scale` keeps its 0."""

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.scale * self.conv2(torch.relu(self.conv1(torch.relu(x))))


class ImpalaCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, channels=(32, 64, 64)):
        super().__init__(observation_space, features_dim)
        layers, ch_in = [], observation_space.shape[0]
        for ch in channels:
            layers += [nn.Conv2d(ch_in, ch, 3, padding=1), nn.MaxPool2d(3, stride=2, padding=1),
                       ResBlock(ch), ResBlock(ch)]
            ch_in = ch
        self.cnn = nn.Sequential(*layers, nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.linear(self.cnn(obs))


ARCHS = {
    "nature": {},
    "impala": {"features_extractor_class": ImpalaCNN},
}


def arch_of(model):
    """Which ARCHS entry a loaded PPO model uses."""
    return "impala" if isinstance(model.policy.features_extractor, ImpalaCNN) else "nature"
