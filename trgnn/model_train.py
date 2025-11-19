import os
import random
import threading
import queue
import wandb
import torch
import numpy as np
from typing_extensions import Literal
from tqdm import tqdm
from .data_utils import DataUtils
from .model_train_utils import ModelTrainUtils,EarlyStopping
from .metrics import Metrics

class ModelTrainer:
    @staticmethod
    def train(model,train_data_loader_list,val_data_loader_list,validate:bool=False,config:dict=None):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['lr']) if config['optimizer']=='adam' else torch.optim.SGD(model.parameters(),lr=config['lr'])

        """
        Early stopping
        """
        if config['early_stop']:
            early_stop=EarlyStopping(patience=config['patience'])

        """
        model train
        """
        for epoch in tqdm(range(config['epochs']),desc=f"Training..."):
            loss_list=[]
            model.train()
            for data_loader in tqdm(train_data_loader_list,desc=f"Epoch {epoch+1}..."):
                label_list=[batch['label'] for batch in data_loader] # List of [B,1], B는 각 element마다 다를 수 있음
                label_list=[label.to(device) for label in label_list]

                output=model(data_loader=data_loader,device=device) # List of [B,1], B는 각 element마다 다를 수 있음

                loss=Metrics.compute_tR_loss(logit_list=output,label_list=label_list)
                loss_list.append(loss)

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            epoch_loss=torch.stack(loss_list).mean().item()

            """
            Early stopping
            """
            if config['early_stop']:
                val_loss=epoch_loss
                pre_model=early_stop(val_loss=val_loss,model=model)
                if early_stop.early_stop:
                    model=pre_model
                    print(f"Early Stopping in epoch {epoch+1}")
                    break

            """
            wandb log
            """
            if config['wandb']:
                wandb.log({
                    f"loss":epoch_loss,
                },step=epoch)
            
            """
            validate
            """
            if validate:
                acc=ModelTrainer.test(model=model,data_loader_list=val_data_loader_list)
                print(f"{epoch+1} epoch tR validation acc: {acc}")

    @staticmethod
    def test(model,data_loader_list):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        """
        model test
        """
        acc_list=[]
        with torch.no_grad():
            for data_loader in tqdm(data_loader_list,desc=f"Evaluating..."):
                label_list=[batch['label'] for batch in data_loader] # List of [B,1], B는 각 element마다 다를 수 있음
                label_list=[label.to(device) for label in label_list]

                output=model(data_loader=data_loader,device=device) # List of [B,1], B는 각 element마다 다를 수 있음

                acc=Metrics.compute_tR_acc(logit_list=output,label_list=label_list)
                acc_list.append(acc)
        return float(np.mean(acc_list))
