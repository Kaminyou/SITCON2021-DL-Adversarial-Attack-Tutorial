import torchvision.models as models
import torch.nn as nn

def get_model(model_name, classes = 10):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 10)
    
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 10)
    
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, 10)
    
    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(pretrained=True)
        model.fc = nn.Linear(2048, 10)
    
    return model