import argparse
import os
import random
import json
import imgaug
import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import *
import matplotlib.pyplot as plt
import cv2

seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from tqdm import tqdm
import models
import torch.nn.functional as F
from utils.datasets.fer2013dataset import fer2013
from utils.datasets.lfw_dataset import lfw
from utils.generals import make_batch

model_dict = [

    # best models for M-LFW
    ("cbam_resnet50","cbam_resnet50__n_2021Apr18_23.46"),
    ("resmasking_dropout1", "resmasking_dropout1__n_2021Apr18_21.31"),

    # best models for LFW
    ("resmasking_dropout1", "resmasking_dropout1__n_2021Apr18_22.13"),
    ("cbam_resnet50","cbam_resnet50__n_2021Apr20_00.24"),

]


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", help="path to config file")
    argparser.add_argument("--type", help="must be one of: train, test, val")
    argparser.add_argument("--model", type=str, help="model name, architecture name used in config file")
    argparser.add_argument("--checkpoint", type=str, help="checkpoint name of model")
    argparser.add_argument("--save_samples",default=0,type=int,help="save samples misclassified by BOTH cbam and resmaskingnet. Default = 0. \n"
                                                          "this requires running the script TWICE: first with cbam, then with resmaskingnet")
    args = argparser.parse_args()
    with open(args.config) as f:
    # with open("./configs/fer2013_config.json") as f:
        configs = json.load(f)

    # test_set = fer2013("test", configs, tta=True, tta_size=8)
    test_set = lfw(args.type,configs)

    model_name = args.model
    checkpoint_path = args.checkpoint

    # for model_name, checkpoint_path in model_dict:
    prediction_list = []  # each item is 7-ele array

    print("Processing", checkpoint_path)
    # if os.path.exists("./saved/results/{}.npy".format(checkpoint_path)):
    #     continue
    if args.save_samples:
        if "m_lfw_" in checkpoint_path:
            cbam = np.load(r"./saved\results\cbam_resnet50__n_2021Apr18_23.46.npy")
        else:
            cbam = np.load(r"./saved\results\cbam_resnet50__n_2021Apr20_00.24.npy")

    model = getattr(models, model_name)
    model = model(in_channels=configs['in_channels'], num_classes=configs['num_classes'])

    state = torch.load(os.path.join(configs['checkpoint_dir'], checkpoint_path))
    model.load_state_dict(state["net"])

    model.cuda()
    model.eval()

    gt = []
    pred = []
    corr = 0
    with torch.no_grad():
        for idx in tqdm(range(len(test_set)), total=len(test_set), leave=False):
            images, targets = test_set[idx]
            images_copy = images
            images = make_batch(images)

            images = images.cuda(non_blocking=True)

            outputs = model(images).cpu()
            outputs = F.softmax(outputs, 1)
            outputs = torch.sum(outputs, 0)  # outputs.shape [tta_size, 7]

            outputs2 = torch.unsqueeze(outputs, 0)
            pred_class = torch.argmax(outputs2, dim=1)
            pred.append(pred_class.numpy()[0])
            gt.append(targets)
            outputs = [round(o, 4) for o in outputs.numpy()]
            prediction_list.append(outputs)
            if targets == pred_class.numpy()[0]:
                corr += 1
            elif args.save_samples and pred_class.numpy()[0]==cbam[idx]:
                cv2.imwrite(f"./debug/target_{targets}pred_{cbam[idx]}_{idx}.png", cv2.cvtColor(255 * np.transpose(images_copy.numpy(), (1, 2, 0)), cv2.COLOR_RGB2BGR))

    np.save("./saved/results/{}.npy".format(checkpoint_path), pred)

    sns.set_style('whitegrid')
    emo_labels = ["negative", "neutral", "positive"]
    # gt = list(map(lambda x: emo_dict[x], gt))
    # pred = list(map(lambda x: emo_dict[x], pred))

    cf_matrix = confusion_matrix(gt, pred)

    print(f"accuracy: {corr} / {len(gt)} = {round(corr / len(gt) * 100, 3)}%")

    plt.figure(figsize=(8, 5))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(cf_matrix, annot=True, square=True, annot_kws={"size": 18})

    if "m_lfw_" in checkpoint_path:
        dataname = "M-LFW"
    else:
        dataname = "LFW"
    if "cbam" in model_name:
        ax.set_title(f"Confusion matrix for Cbam_Resnet50 ({dataname} dataset)", weight='bold')
    else:
        ax.set_title(f"Confusion matrix for ResMaskingNet ({dataname} dataset)", weight='bold')
    ax.set_xticklabels(emo_labels + [''], fontsize=15, weight='bold')
    ax.set_yticklabels(emo_labels + [''], fontsize=15, weight='bold')
    ax.set_xlabel('Predicted label', weight='bold')
    ax.set_ylabel('True label', weight='bold')
    # ax.set_xticks(['negative','neutral','positive'])
    # ax.get_xticks()
    # ax.set_xticks(['negative', 'neutral', 'positive'])
    plt.tight_layout()
    plt.savefig(f"cm_{model_name}_{dataname}.png")
    plt.show()



if __name__ == "__main__":
    main()
