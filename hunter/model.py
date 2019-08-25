import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, h_sizes=[64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        layer_sizes = zip(h_sizes[:-1], h_sizes[1:])
        
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, h_sizes[0])])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(h_sizes[-1], action_size)

        # self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            # x = self.dropout(x)
        
        return self.output(x)
