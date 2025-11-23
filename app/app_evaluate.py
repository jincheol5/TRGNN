import os
import random
import numpy as np
import argparse
import wandb
import torch
from torch.utils.data import DataLoader
from trgnn import DataUtils,ModelTrainer,ModelTrainUtils,TGAT,TGN,TRGNN,TRGAT

def app_evaluate(config:dict):
    match config['app_num']:
        case 1:
            """
            App 1.
            evaluate models
                test_20
                test_50
                test_100
                test_500
            """
            wandb.init(project="tRGNN",name=f"test_{config['num_nodes']}_result")

            """
            load dataset_list
            """
            dataset_list=DataUtils.load_from_pickle(file_name=f"test_{config['num_nodes']}",dir_type="test",num_nodes=config['num_nodes'])
            test_data_loader_list=[]
            for dataset in dataset_list:
                data_loader=ModelTrainUtils.get_data_loader(dataset=dataset,batch_size=config['batch_size'])
                test_data_loader_list.append(data_loader)
            
            model_list=['tgat','tgn','trgnn','trgat']
            seed_list=[1,2,3]
            lr_list=[0.001,0.0005]
            batch_size=16
            latent_dim=32

            for seed in seed_list:
                for lr in lr_list:
                    for model in model_list:
                        if model=='tgn':
                            emb_list=['time','sum','attn']
                        else:
                            emb_list=[None]
                        for emb in emb_list:
                            """
                            seed setting
                            """
                            random.seed(seed)
                            np.random.seed(seed)
                            torch.manual_seed(seed) 
                            os.environ["PYTHONHASHSEED"]=str(seed)
                            torch.cuda.manual_seed(seed)
                            torch.cuda.manual_seed_all(seed)
                            torch.backends.cudnn.deterministic=True 
                            torch.backends.cudnn.benchmark=False

                            """
                            model setting and evaluating
                            """
                            match model:
                                case 'tgat':
                                    model_name=f"tgat_{seed}_{lr}_{batch_size}"
                                    model=TGAT(node_dim=1,latent_dim=latent_dim)
                                    model=DataUtils.load_model_parameter(model=model,model_name=model_name)
                                case 'tgn':
                                    model_name=f"tgn_{emb}_{seed}_{lr}_{batch_size}"
                                    model=TGN(node_dim=1,latent_dim=latent_dim,emb=emb)
                                    model=DataUtils.load_model_parameter(model=model,model_name=model_name)
                                case 'trgnn':
                                    model_name=f"trgnn_{seed}_{lr}_{batch_size}"
                                    model=TRGNN(node_dim=1,latent_dim=latent_dim)
                                    model=DataUtils.load_model_parameter(model=model,model_name=model_name)
                                case 'trgat':
                                    model_name=f"trgat_{seed}_{lr}_{batch_size}"
                                    model=TRGAT(node_dim=1,latent_dim=latent_dim)
                                    model=DataUtils.load_model_parameter(model=model,model_name=model_name)
                            ModelTrainer.test(model=model,data_loader_list=test_data_loader_list)