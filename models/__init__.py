from .vgg import *
from .resnet import *
from .resnet112 import resnet18x112
from .resnet50_scratch_dims_2048 import resnet50_pretrained_vgg
from .centerloss_resnet import resnet18_centerloss
from .resatt import *
from .alexnet import *
from .densenet import *
from .googlenet import *
from .inception import *
from .inception_resnet_v1 import *
from .residual_attention_network import *
from .fer2013_models import *
from .res_dense_gle import *
from .masking import masking
from .resmasking import (
    resmasking,
    resmasking_dropout1,
    resmasking_dropout2,
    resmasking50_dropout1,
)
from .resmasking_naive import resmasking_naive_dropout1
from .brain_humor import *
from .runet import *
from pytorchcv.model_provider import get_model as ptcv_get_model


def resattnet56(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("resattnet56", pretrained=False)
    model.output = nn.Linear(2048, num_classes)
    return model


def cbam_resnet50(in_channels, num_classes, pretrained=1):
    if pretrained==1:
        model = ptcv_get_model("cbam_resnet50", pretrained=True)
        model.output = nn.Linear(2048, num_classes)
        print('using pretrained imagenet weights')
    else:
        model = ptcv_get_model("cbam_resnet50", pretrained=False)
        model.output = nn.Linear(2048, num_classes)
        if pretrained==2:
            print('using pretrained lfw weights')
            state_dict = torch.load('saved/checkpoints/cbam_resnet50__n_2021Apr20_00.24')['net']
            model.load_state_dict(state_dict)

    # freeze params
    # for name,param in model.named_parameters():
        # if "stage4" not in name:
        #     param.requires_grad = False
    # model.output = nn.Linear(2048, num_classes)
    # model.output = nn.Sequential(nn.Dropout(p=0.5, inplace=False), nn.Linear(2048, num_classes))

    return model


def bam_resnet50(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("bam_resnet50", pretrained=True)
    model.output = nn.Linear(2048, num_classes)
    return model


def efficientnet_b7b(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b7b", pretrained=True)
    model.output = nn.Sequential(nn.Dropout(p=0.5, inplace=False), nn.Linear(2560, num_classes))
    return model


def efficientnet_b3b(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b3b", pretrained=True)
    model.output = nn.Sequential(nn.Dropout(p=0.3, inplace=False), nn.Linear(1536, num_classes))
    return model


def efficientnet_b2b(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b2b", pretrained=True)
    model.output = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False), nn.Linear(1408, num_classes, bias=True)
    )
    return model


def efficientnet_b1b(in_channels, num_classes, pretrained=True):
    model = ptcv_get_model("efficientnet_b1b", pretrained=True)
    print(model)
    model.output = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False), nn.Linear(1280, num_classes, bias=True)
    )
    return model
