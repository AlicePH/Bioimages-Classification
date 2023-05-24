import os
import gc

import torch
import torchvision
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import random
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything()



class Trainer:
    def __init__(self, 
                 model, 
                 criteriation,
                 device,
                 train_dataloader,
                 test_dataloader,
                 trainset_len,
                 testset_len,
                 optimizer=None,
                 epochs=None,
                 path_output=None,
                 multi_label=False
                ):
    
        self.model = model
        self.optimizer = optimizer
        self.criteriation = criteriation
        self.device = device
        self.epochs = epochs
        self.path_output = path_output
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.trainset_len = trainset_len
        self.testset_len = testset_len

        self.multi_label = multi_label
        
    
    def training(self, start=0, show_treshold=250):
        
    
        metrics = []
        max_val_accuracy = 0

        for epoch in range(start, self.epochs):
            print(f'[{epoch+1}]/[{self.epochs}] Epoch starts')
            train_loss = 0
            train_acc = 0
            self.model.train()

            for b_ind, batch in enumerate(self.train_dataloader):


                imgs, labels = batch
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs = self.model(imgs)

                if self.multi_label:
                    outputs = torch.sigmoid(outputs)
                    pred = outputs
                    train_acc += self.compute_mAP((labels, pred), self.multi_label) * labels.size(0)

                else:
                    pred = torch.max(outputs, 1)[1]
                    train_acc +=  (pred == labels).sum()


                Loss = self.criteriation(outputs, labels)
                train_loss += Loss.item() * labels.size(0)



                if b_ind % show_treshold==0:
                    b_loss = Loss.item()
                    b_acc = self.compute_mAP((labels, pred), self.multi_label)
                    print(f'\t Batch train loss: {b_loss}, accuracy {b_acc}')


                self.optimizer.zero_grad()
                Loss.backward()
                self.optimizer.step()


                del imgs, labels



            val_loss, val_accuracy, _= self.val()
    
            train_loss = train_loss / self.trainset_len
            train_acc = train_acc / self.trainset_len



            print(f'[{epoch+1}]/[{self.epochs}] End epoch: train loss: {train_loss}, val loss: {val_loss}')
            print(f'\t Epoch train accuracy: {train_acc}, val accuracy: {val_accuracy}\n')

            torch.cuda.empty_cache()
            gc.collect()

            if val_accuracy > max_val_accuracy:
                max_val_accuracy = val_accuracy
                self.save_model(self.path_output, epoch, self.model, self.optimizer, self.criteriation)

            metrics.append([train_loss, val_loss, train_acc, val_accuracy])
            pd.DataFrame(metrics, columns=['train_loss',
                                           'val_loss', 
                                           'train accuracy',
                                           'val accuracy'
                                          ]).to_csv(self.path_output.split('.')[0] + '_metrics.csv', index=False)
            
            
    def val(self):
        self.model.eval()

        val_loss = 0
        val_mAP = 0

        true_label = []
        pred_label = []

        for batch in self.test_dataloader:
            img, label = batch

            img, label = img.to(self.device), label.to(self.device)

            with torch.no_grad():
                output = self.model(img)

                if self.multi_label:
                    output = torch.sigmoid(output)
                    pred_label.append(output)

                else:
                    _, preds = torch.max(output, 1)
                    pred_label.append(preds.item())


            loss = self.criteriation(output, label)
            val_loss += loss.item()

            true_label.append(label)

            del img, label

        if self.multi_label:
            prediction = (torch.cat(true_label), torch.cat(pred_label))
        else:
            prediction = (torch.tensor(true_label), torch.tensor(pred_label))

        val_loss = val_loss / self.testset_len
        return val_loss, self.compute_mAP(prediction, self.multi_label), prediction
        
        
            
    def compute_mAP(self, pred, multi_class=False):
        if multi_class:
            labels, outputs = pred
            y_true = labels.long().cpu().detach().numpy()
            y_pred = outputs.cpu().detach().numpy()
            AP = []
            for i in range(y_true.shape[0]):
                AP.append(average_precision_score(y_true[i], y_pred[i]))
            accuracy = np.mean(AP)

        else:
            outputs, labels = pred
            y_true = labels.cpu().detach().numpy()
            y_pred = outputs.cpu().detach().numpy()
            accuracy = accuracy_score(y_true, y_pred)

        return accuracy
    
    def save_model(self, output_dir, epochs, model, optimizer, criterion):
        torch.save({
                    'epoch': epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    }, output_dir)
