import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # On utilise un modèle certifié (celui de Google)
        # 'google/vit-base-patch16-224-in21k' est un modèle "vierge" de classification
        # On le configure spécifiquement pour 10 classes (MNIST)
        model_name = "google/vit-base-patch16-224-in21k"
        
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=10,
            id2label={str(i): str(i) for i in range(10)},
            label2id={str(i): i for i in range(10)}
        )

    def forward(self, x):
        # Hugging Face renvoie un objet SequenceClassifierOutput
        outputs = self.vit(x)
        return outputs.logits
