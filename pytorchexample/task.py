"""pytorchexample: A Flower / PyTorch app."""

import torch
from torchmetrics.classification import MulticlassF1Score
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, GaussianBlur
try:
    from pytorchexample.user_model import Net  # Modèle utilisateur
except ImportError:
    from pytorchexample.model import Net       # Modèle par défaut
from functools import partial 

fds = None  # Cache FederatedDataset

# Sigma max pour un flou à 100%
BLUR_SIGMA_MAX = 10.0


def apply_transforms(batch, img_size=32, num_channels=3, blur_percent=0.0):
    # 1. Détecter la clé d'image
    key = "image" if "image" in batch else "img"
    
    # 2. Choisir le mode de conversion et la normalisation selon num_channels
    if num_channels == 3:
        mode = "RGB"
        norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        mode = "L"  # Niveaux de gris
        norm = Normalize((0.5,), (0.5,))

    # 3. Construire les transforms dynamiquement (incluant le flou)
    transforms_list = [Resize((img_size, img_size)), ToTensor()]
    
    if blur_percent > 0:
        sigma = (blur_percent / 100.0) * BLUR_SIGMA_MAX
        kernel_size = max(3, int(6 * sigma + 1) | 1)
        transforms_list.append(GaussianBlur(kernel_size=kernel_size, sigma=sigma))
        
    transforms_list.append(norm)
    pytorch_transforms = Compose(transforms_list)
    
    batch["pixel_values"] = [pytorch_transforms(img.convert(mode)) for img in batch[key]]
    
    # --- GESTION DU LABEL INTELLIGENTE ---
    if "label" in batch:
        # Cas CIFAR-10 standard
        batch["label"] = batch["label"]
    elif "Cardiomegaly" in batch:
        # Cas CheXpert
        batch["label"] = [int(val) if val is not None else 0 for val in batch["Cardiomegaly"]]
    
    del batch[key]
    return batch


# -------------------------
# Client type generator
# -------------------------
def build_client_types(num_partitions, small_client, medium_client, big_client):
    types = (
        ["small"] * small_client +
        ["medium"] * medium_client +
        ["big"] * big_client
    )
    while len(types) < num_partitions:
        types.append("big")
    return types[:num_partitions]


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    small_client: int,
    medium_client: int,
    big_client: int,
    dataset_name="uoft-cs/cifar10",
    img_size=32,
    num_channels=3,
    alpha: float = 0.1,
    self_balancing: bool = True,
    blur_percent: float = 0.0
):
    """Load partition data with Quantity Skew and Feature Skew (Blur)."""
    global fds

    if fds is None:
        target_col = "Cardiomegaly" if dataset_name == "danjacobellis/chexpert" else "label"
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=target_col,
            alpha=alpha,
            self_balancing=self_balancing
        )

        fds = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)

    # -------------------------
    # Quantity skew (client size control)
    # -------------------------
    client_types = build_client_types(num_partitions, small_client, medium_client, big_client)
    client_type = client_types[partition_id]

    if client_type == "small":
        fraction = 0.1
    elif client_type == "medium":
        fraction = 0.5
    else:
        fraction = 1.0

    partition = partition.shuffle(seed=42).select(
        range(max(1, int(len(partition) * fraction)))
    )

    # Divide data: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # Intégration du flou spécifique au client via partial
    current_transforms = partial(
        apply_transforms, 
        img_size=img_size, 
        num_channels=num_channels, 
        blur_percent=blur_percent
    )
    
    partition_train_test = partition_train_test.with_transform(current_transforms)

    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)

    return trainloader, testloader


def load_centralized_dataset(dataset_name="uoft-cs/cifar10", img_size=32, num_channels=3):
    """Charge le set de test global (sans flou) pour l'évaluation serveur."""
    if "uoft-cs/cifar10" in dataset_name.lower():
        split_name = "test"
    elif "danjacobellis/chexpert" in dataset_name.lower():
        split_name = "train[-1000:]"
    else:
        split_name = "test"

    test_dataset = load_dataset(dataset_name, split=split_name)
    
    # Le serveur évalue sur des images nettes (blur_percent=0.0)
    server_transforms = partial(apply_transforms, img_size=img_size, num_channels=num_channels, blur_percent=0.0)
    dataset = test_dataset.with_transform(server_transforms)
    
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device, num_classes=10):
    """Validate the model on the test set with Accuracy and F1-Score."""
    net.to(device)
    f1_metric = MulticlassF1Score(num_classes, average='weighted').to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            preds = torch.argmax(outputs, dim=1)
            f1_metric.update(preds, labels)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    f1 = f1_metric.compute().item()
    f1_metric.reset()
    return loss, accuracy, f1
