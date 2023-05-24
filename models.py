# from typing import Tuple, Dict, Any, Union
# import torch
# import torchvision.models as models
# from torch import nn
# from torch.hub import load
# from sklearn.metrics import fbeta_score, roc_auc_score
# import torch
# import time
# import copy
# import wandb
# from tqdm import tqdm
# from torch import nn
# from collections import OrderedDict
# from enum import Enum
# import torch.optim as optim
# import torch.nn as nn



# class ModelName(Enum):
#     RESNET18 = 'resnet18'
#     RESNET50 = 'resnet50'
#     RESNET152 = 'resnet152'
#     RESNEXT50_32X4D = 'resnext50_32x4d'
#     RESNEXT101_32X8D = 'resnext101_32x8d'
#     RESNET18SSL = 'resnet18_ssl'
#     RESNET18SWSL = 'resnet18_swsl'
#     KIMIANET = 'kimianet'

# def get_fully_connected_resnet(model: nn.Module, num_classes: int) -> nn.Module:
#     """
#     Wraps a given Resnet model with a fully connected layer.

#     Args:
#     model (nn.Module): Model to be wrapped.
#     num_classes (int): Number of classes for the final output layer.

#     Returns:
#     nn.Module: Model wrapped with a fully connected layer.
#     """
#     model = copy.deepcopy(model)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model

# def get_kimianet(model: nn.Module, num_classes: int, weight_path: str) -> nn.Module:
#     """
#     Modifies a given DenseNet model and loads pretrained weights from a path.

#     Args:
#     model (nn.Module): Model to be modified.
#     num_classes (int): Number of classes for the final output layer.
#     weight_path (str): Path to the pretrained weights.

#     Returns:
#     nn.Module: Modified model loaded with pretrained weights.
#     """
#     model = copy.deepcopy(model)
#     model.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(output_size=(1,1)))
#     model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
#     state_dict = torch.load(weight_path)
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k[7:]  # remove 'module.' prefix
#         new_state_dict[name] = v
#     model.load_state_dict(new_state_dict, strict=False)
    
#     return model

# def load_ssl_model(model_name: ModelName) -> nn.Module:
#         return load('facebookresearch/semi-supervised-ImageNet1K-models', model_name.value)

# def load_swsl_model(model_name: ModelName) -> nn.Module:
#         return load('facebookresearch/semi-supervised-ImageNet1K-models', model_name.value)


# MODEL_BUILDERS = {
#     ModelName.RESNET18: lambda pretrained: models.resnet18(pretrained=pretrained),
#     ModelName.RESNET50: lambda pretrained: models.resnet50(pretrained=pretrained),
#     ModelName.RESNET152: lambda pretrained: models.resnet152(pretrained=pretrained),
#     ModelName.RESNEXT50_32X4D: lambda pretrained: models.resnext50_32x4d(pretrained=pretrained),
#     ModelName.RESNEXT101_32X8D: lambda pretrained: models.resnext101_32x8d(pretrained=pretrained),
#     ModelName.KIMIANET: lambda pretrained: models.densenet121(pretrained=pretrained),
#     ModelName.RESNET18SSL: lambda model_name: load_ssl_model(model_name = ModelName.RESNET18SSL),
#     ModelName.RESNET18SWSL: lambda model_name: load_swsl_model(model_name = ModelName.RESNET18SWSL),
    
# }

# def get_base_model(model_name: str, device: Any, num_classes: int, kimianet_weight_path: str = None, pretrained: bool = True) -> nn.Module:
#     """
#     Constructs a model given a model name.

#     Args:
#     model_name (ModelName): Enum specifying the model to be constructed.
#     num_classes (int): Number of classes for the final output layer.
#     weight_path (str, optional): If model_name is KimiaNet, path to the pretrained weights.
#     pretrained (bool, optional): If True, uses pretrained model.

#     Returns:
#     nn.Module: The constructed model.
#     """
#     base_model_builder = MODEL_BUILDERS[model_name]
#     base_model = base_model_builder(pretrained)
#     if model_name == ModelName.KIMIANET:
#         model = get_kimianet(base_model, num_classes, kimianet_weight_path).to(device)
#     else:
#         model = get_fully_connected_resnet(base_model, num_classes).to(device)
#     return model

# def train_step(inputs: torch.Tensor, labels: torch.Tensor, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
#     """
#     Performs a training step for the model.

#     Args:
#     inputs (torch.Tensor): The input data for the model.
#     labels (torch.Tensor): The true labels for the input data.
#     model (nn.Module): The model to be trained.
#     criterion (nn.Module): The loss function.
#     optimizer (torch.optim.Optimizer): The optimizer.

#     Returns:
#     Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]: A tuple containing the model outputs, predictions, softmax outputs, and the loss.
#     """
#     optimizer.zero_grad()

#     with torch.set_grad_enabled(True):
#         outputs = model(inputs)
        
#         # If your model returns embeddings and predictions, 
#         # outputs would be a tuple and you would need to unpack it
#         if isinstance(outputs, tuple):
#             embeddings, outputs = outputs
        
#         softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
#         _, preds = torch.max(softmax_outputs, 1)
#         loss = criterion(softmax_outputs, labels)
#         loss.backward()
#         optimizer.step()

#     return outputs, preds, softmax_outputs, loss.item()

# def train_model(model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler, dataloaders: Dict[str, torch.utils.data.DataLoader], device: torch.device, num_epochs: int=25) -> nn.Module:
#     """
#     Trains the model.

#     Args:
#     model (nn.Module): The model to be trained.
#     criterion (nn.Module): The loss function.
#     optimizer (torch.optim.Optimizer): The optimizer.
#     scheduler: Learning rate scheduler.
#     dataloaders (Dict[str, torch.utils.data.DataLoader]): The data loaders.
#     device (torch.device): The device to run the model on.
#     num_epochs (int, optional): Number of epochs. Default is 25.

#     Returns:
#     nn.Module: The trained model.
#     """
#     since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val_in', 'test_out']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode

#             running_loss = 0.0
#             running_corrects = 0
#             all_targets = []
#             all_scores = []
#             all_preds = []

#             # Iterate over data.
#             for inputs, labels in tqdm(dataloaders[phase]):
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     # perform a training step if in 'train' phase
#                     if phase == 'train':
#                         outputs, preds, softmax_outputs, loss = train_step(inputs, labels, model, criterion, optimizer)
#                     else:
#                         _, outputs = model(inputs)
#                         softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
#                         _, preds = torch.max(softmax_outputs, 1)
#                         loss = criterion(softmax_outputs, labels)
                    
#                 # statistics
#                 all_targets.extend(labels.cpu().detach().numpy().tolist())
#                 all_preds.extend(preds.cpu().detach().numpy().tolist())
#                 all_scores.extend(softmax_outputs[...,1].cpu().detach().numpy().tolist())
                
#                 running_loss += loss * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             if phase == 'train':
#                 scheduler.step()

#             val_roc_auc = roc_auc_score(all_targets, all_scores, average='weighted')
#             val_f1 = fbeta_score(all_targets, all_preds, average='weighted', beta=5)
#             epoch_loss = running_loss / len(dataloaders[phase].dataset)
#             epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

#             # logging metrics
#             log_metrics(phase, epoch_loss, epoch_acc, val_f1, val_roc_auc)

#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#             # deep copy the model
#             if phase == 'val_in' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#     time_elapsed = time.time() - since
#     print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#     print(f'Best val Acc: {best_acc:4f}')

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model

# def log_metrics(phase, epoch_loss, epoch_acc, f1_score, roc_auc):
#     metrics_dict = {
#         'epoch_loss': epoch_loss,
#         'epoch_acc': epoch_acc,
#         'f1': f1_score,
#         'auc': roc_auc,
#     }

#     for metric, value in metrics_dict.items():
#         wandb.log({f'{phase}_{metric}': value})
    

# def get_model(config, device):
#     torch.manual_seed(0)
#     model_name_str = config["model"].upper().replace("_", "")
#     model_name = ModelName[model_name_str]
#     kimianet_weight_path = config["kimianet_weight_path"]
#     model = get_base_model(model_name, device, num_classes=2, 
#                            kimianet_weight_path=kimianet_weight_path, 
#                            pretrained=config["pretrained"])
#     return model

# def freeze_layers(model, freeze_until_layer):
#     ct = 0
#     for child in model.children():
#         ct += 1
#         if ct < freeze_until_layer:
#             for param in child.parameters():
#                 param.requires_grad = False
#     return model

# def initialize_optimizer(model, config):
#     optimizer = optim.SGD(model.parameters(), 
#                           lr=config['learning_rate'], 
#                           momentum=config['momentum'])
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#     return optimizer, scheduler

# def get_criterion(criterion: str) -> Union[nn.Module, None]:
#     if criterion.lower() == 'cross_entropy':
#         return nn.CrossEntropyLoss()
#     else:
#         print(f"Criterion '{criterion}' not recognized. Please choose 'cross_entropy'.")
#         return None

# def save_model(model: torch.nn.Module, save_path: str) -> None:
#     """
#     Saves the model to the specified path.

#     Args:
#         model (torch.nn.Module): The model to be saved.
#         save_path (str): The path where the model should be saved.
#     """
#     try:
#         torch.save(model, save_path)
#         print(f"Model saved successfully at {save_path}")
#     except Exception as e:
#         print(f"Saving model failed: {e}")


# def get_device() -> torch.device:
#     """
#     Returns the device to use for computations (either CPU or GPU).

#     Returns:
#         device (torch.device): The device to use for computations.
#     """
#     return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





from typing import Tuple, Dict, Any, Union
import torch
import torchvision.models as models
from torch import nn
from torch.hub import load
from sklearn.metrics import fbeta_score, roc_auc_score
import torch
import time
import copy
import wandb
from tqdm import tqdm
from torch import nn
from collections import OrderedDict
from enum import Enum
import torch.optim as optim
import torch.nn as nn

class ModelName(Enum):
    RESNET18 = 'resnet18'
    RESNET50 = 'resnet50'
    RESNET152 = 'resnet152'
    RESNEXT50_32X4D = 'resnext50_32x4d'
    RESNEXT101_32X8D = 'resnext101_32x8d'
    RESNET18SSL = 'resnet18_ssl'
    RESNET18SWSL = 'resnet18_swsl'
    KIMIANET = 'kimianet'


class ModelHandler:
    MODEL_BUILDERS = {
        ModelName.RESNET18: lambda pretrained: models.resnet18(pretrained=pretrained),
        ModelName.RESNET50: lambda pretrained: models.resnet50(pretrained=pretrained),
        ModelName.RESNET152: lambda pretrained: models.resnet152(pretrained=pretrained),
        ModelName.RESNEXT50_32X4D: lambda pretrained: models.resnext50_32x4d(pretrained=pretrained),
        ModelName.RESNEXT101_32X8D: lambda pretrained: models.resnext101_32x8d(pretrained=pretrained),
        ModelName.KIMIANET: lambda pretrained: models.densenet121(pretrained=pretrained),
        ModelName.RESNET18SSL: lambda model_name: ModelHandler.load_ssl_model(model_name = ModelName.RESNET18SSL),
        ModelName.RESNET18SWSL: lambda model_name: ModelHandler.load_swsl_model(model_name = ModelName.RESNET18SWSL),
    }

    @classmethod
    def load_ssl_model(cls, model_name: ModelName) -> nn.Module:
        return load('facebookresearch/semi-supervised-ImageNet1K-models', model_name.value)

    @classmethod
    def load_swsl_model(cls, model_name: ModelName) -> nn.Module:
        return load('facebookresearch/semi-supervised-ImageNet1K-models', model_name.value)

    @staticmethod
    def get_fully_connected_resnet(model: nn.Module, num_classes: int) -> nn.Module:
        """
        Wraps a given Resnet model with a fully connected layer.

        Args:
        model (nn.Module): Model to be wrapped.
        num_classes (int): Number of classes for the final output layer.

        Returns:
        nn.Module: Model wrapped with a fully connected layer.
        """
        model = copy.deepcopy(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    @staticmethod
    def get_kimianet(model: nn.Module, num_classes: int, weight_path: str) -> nn.Module:
        """
        Modifies a given DenseNet model and loads pretrained weights from a path.

        Args:
        model (nn.Module): Model to be modified.
        num_classes (int): Number of classes for the final output layer.
        weight_path (str): Path to the pretrained weights.

        Returns:
        nn.Module: Modified model loaded with pretrained weights.
        """
        model = copy.deepcopy(model)
        model.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(output_size=(1,1)))
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
        state_dict = torch.load(weight_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        
        return model

    @classmethod
    def get_base_model(cls, model_name: str, device: Any, num_classes: int, kimianet_weight_path: str = None, pretrained: bool = True) -> nn.Module:
        """
        Constructs a model given a model name.

        Args:
        model_name (ModelName): Enum specifying the model to be constructed.
        num_classes (int): Number of classes for the final output layer.
        weight_path (str, optional): If model_name is KimiaNet, path to the pretrained weights.
        pretrained (bool, optional): If True, uses pretrained model.

        Returns:
        nn.Module: The constructed model.
        """
        base_model_builder = cls.MODEL_BUILDERS[model_name]
        base_model = base_model_builder(pretrained)
        if model_name == ModelName.KIMIANET:
            model = cls.get_kimianet(base_model, num_classes, kimianet_weight_path).to(device)
        else:
            model = cls.get_fully_connected_resnet(base_model, num_classes).to(device)
        return model
    
    @classmethod
    def train_step(cls, inputs: torch.Tensor, labels: torch.Tensor, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Performs a training step for the model.

        Args:
        inputs (torch.Tensor): The input data for the model.
        labels (torch.Tensor): The true labels for the input data.
        model (nn.Module): The model to be trained.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.

        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]: A tuple containing the model outputs, predictions, softmax outputs, and the loss.
        """
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            
            # If your model returns embeddings and predictions, 
            # outputs would be a tuple and you would need to unpack it
            if isinstance(outputs, tuple):
                embeddings, outputs = outputs
            
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(softmax_outputs, 1)
            loss = criterion(softmax_outputs, labels)
            loss.backward()
            optimizer.step()

        return outputs, preds, softmax_outputs, loss.item()
    
    @classmethod
    def train_model(cls, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler, dataloaders: Dict[str, torch.utils.data.DataLoader], device: torch.device, num_epochs: int=25) -> nn.Module:
        """
        Trains the model.

        Args:
        model (nn.Module): The model to be trained.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler: Learning rate scheduler.
        dataloaders (Dict[str, torch.utils.data.DataLoader]): The data loaders.
        device (torch.device): The device to run the model on.
        num_epochs (int, optional): Number of epochs. Default is 25.

        Returns:
        nn.Module: The trained model.
        """
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val_in', 'test_out']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                all_targets = []
                all_scores = []
                all_preds = []

                # Iterate over data.
                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # perform a training step if in 'train' phase
                        if phase == 'train':
                            outputs, preds, softmax_outputs, loss = cls.train_step(inputs, labels, model, criterion, optimizer)
                        else:
                            model_output = model(inputs)
                            if isinstance(model_output, tuple):
                                _, outputs = model_output
                            else:
                                outputs = model_output
                            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
                            _, preds = torch.max(softmax_outputs, 1)
                            loss = criterion(softmax_outputs, labels)
                        
                    # statistics
                    all_targets.extend(labels.cpu().detach().numpy().tolist())
                    all_preds.extend(preds.cpu().detach().numpy().tolist())
                    all_scores.extend(softmax_outputs[...,1].cpu().detach().numpy().tolist())
                    
                    running_loss += loss * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                val_roc_auc = roc_auc_score(all_targets, all_scores, average='weighted')
                val_f1 = fbeta_score(all_targets, all_preds, average='weighted', beta=5)
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                # logging metrics
                cls.log_metrics(phase, epoch_loss, epoch_acc, val_f1, val_roc_auc)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val_in' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
    
    @staticmethod
    def log_metrics(phase, epoch_loss, epoch_acc, f1_score, roc_auc):
        metrics_dict = {
            'epoch_loss': epoch_loss,
            'epoch_acc': epoch_acc,
            'f1': f1_score,
            'auc': roc_auc,
        }

        for metric, value in metrics_dict.items():
            wandb.log({f'{phase}_{metric}': value})
    
    @classmethod
    def get_model(cls, config, device):
        torch.manual_seed(0)
        model_name_str = config["model"].upper().replace("_", "")
        model_name = ModelName[model_name_str]
        kimianet_weight_path = config["kimianet_weight_path"]
        model = cls.get_base_model(model_name, device, num_classes=2, 
                            kimianet_weight_path=kimianet_weight_path, 
                            pretrained=config["pretrained"])
        return model
    
    @classmethod
    def freeze_layers(cls, model, freeze_until_layer):
        ct = 0
        for child in model.children():
            ct += 1
            if ct < freeze_until_layer:
                for param in child.parameters():
                    param.requires_grad = False
        return model
    
    @classmethod
    def initialize_optimizer(cls, model, config):
        optimizer = optim.SGD(model.parameters(), 
                            lr=config['learning_rate'], 
                            momentum=config['momentum'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return optimizer, scheduler
    
    @classmethod
    def get_criterion(cls, criterion: str) -> Union[nn.Module, None]:
        if criterion.lower() == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            print(f"Criterion '{criterion}' not recognized. Please choose 'cross_entropy'.")
            return None
    @classmethod
    def save_model(cls, model: torch.nn.Module, save_path: str) -> None:
        """
        Saves the model to the specified path.

        Args:
            model (torch.nn.Module): The model to be saved.
            save_path (str): The path where the model should be saved.
        """
        try:
            torch.save(model, save_path)
            print(f"Model saved successfully at {save_path}")
        except Exception as e:
            print(f"Saving model failed: {e}")

    @classmethod
    def get_device(cls) -> torch.device:
        """
        Returns the device to use for computations (either CPU or GPU).

        Returns:
            device (torch.device): The device to use for computations.
        """
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # other methods like train_step, train_model, log_metrics, get_model, freeze_layers, initialize_optimizer, get_criterion, save_model, get_device go here
        # remember to change the method decorators from @staticmethod to @classmethod if the method uses any class level attributes
