import os
import argparse

import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn

from data.multi_class_build_data import build_dataloader
from utils.trainer import Trainer
from transformer.vit import Transformer
from utils.optimizer import get_optimizer

# --- args parsing

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--path", type=str, default='multi_class_output')
parser.add_argument("--opt", type=str, default='adam')


args = parser.parse_args()

EPOCHS = args.epochs
PATH =  args.path 
LR= args.lr
BATCH_SIZE=args.batch_size

criteriation = nn.CrossEntropyLoss()
optimizer_fn = get_optimizer(args.opt)

# --- args parsing

os.environ['CUDA_VISIABLE_DEVICES']='0'
device = torch.device('cuda')

transfrom = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

if not os.path.exists(PATH):
    os.mkdir(PATH)
    
    
# DATA
train_dataloader, test_dataloader, num_train, num_test, NUM_CLASS = build_dataloader(transfrom, BATCH_SIZE)

# MODELS
resnet = torchvision.models.resnet34(pretrained=False, num_classes=NUM_CLASS)
resnet = resnet.to(device)
# ---
resnet_modified = torchvision.models.resnet34(pretrained=False, num_classes=NUM_CLASS)

resnet_modified.conv1 = torch.nn.Conv2d(3, 64, 
                                        kernel_size=(3, 3), 
                                        stride=(1, 1), 
                                        padding=(1, 1), 
                                        bias=False)

resnet_modified.maxpool = torch.nn.Identity()

resnet_modified = resnet_modified.to(device)
# ---

effnet = torchvision.models.efficientnet_b0(pretrained=False, num_classes=NUM_CLASS)
effnet = effnet.to(device)

# ---

transformer = Transformer(img_size=(240, 320),
                          patch_size=(8, 8),
                          in_channels=3,
                          n_classes=NUM_CLASS,
                          embed_dim=128,
                          depth=6,
                          n_heads=16,
                          mlp_ratio=4.,
                          qkv_bias=True,
                          p=0.3,
                          attn_p=0.3
                         )

transformer = transformer.to(device)
# ---

models = [resnet, resnet_modified, effnet, transformer]
names = ['resnet', 'resnet_modified', 'effnet', 'transformer']


for model_name, model in zip(names, models):
    print()
    print('-'*20 + model_name + '-'*20)
    
    OUTPUT_DIR = f'{PATH}/{model_name}.pt'
    optimizer = optimizer_fn(model.parameters(), lr=LR)


    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criteriation=criteriation,
        device=device,
        epochs=EPOCHS,
        path_output=OUTPUT_DIR,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        trainset_len=num_train, 
        testset_len=num_test
    )
    
    trainer.training()