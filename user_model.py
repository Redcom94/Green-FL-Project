import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

class DynamicNet(nn.Module): # Un modèle dynamique qui peut être configuré à partir d'un fichier .JSON. Idéal pour la sécurité (pas de code arbitraire à exécuter) et la flexibilité.
    def __init__(self, config):
        super(DynamicNet, self).__init__()
        layers = []
        
        for layer_cfg in config["layers"]:
            layer_type = layer_cfg["type"]
            params = layer_cfg["params"]
            
            # On récupère la classe dans torch.nn (ex: nn.Conv2d)
            layer_class = getattr(nn, layer_type)
            layers.append(layer_class(**params))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)