import os
import random
import pickle
import torch
from tqdm import tqdm
from typing_extensions import Literal
from .graph_utils import GraphUtils

class DataUtils:
    dataset_path=os.path.join('..','data','trgnn')
    
    @staticmethod
    def save_to_pickle(data,file_name:str,dir_type:Literal['graph','train','val','test'],num_nodes:Literal[20,50,100,500,1000]=20):
        file_name=file_name+".pkl"
        file_path=os.path.join(DataUtils.dataset_path,dir_type,file_name)
        if dir_type=='test':
            file_path=os.path.join(DataUtils.dataset_path,dir_type,f"{num_nodes}",file_name)
        with open(file_path,'wb') as f:
            pickle.dump(data,f)
        print(f"Save {file_name}")
    
    @staticmethod
    def load_from_pickle(file_name:str,dir_type:Literal['graph','train','val','test'],num_nodes:Literal[20,50,100,500,1000]=20):
        file_name=file_name+".pkl"
        file_path=os.path.join(DataUtils.dataset_path,dir_type,file_name)
        if dir_type=='test':
            file_path=os.path.join(DataUtils.dataset_path,dir_type,f"{num_nodes}",file_name)
        with open(file_path,'rb') as f:
            data=pickle.load(f)
        print(f"Load {file_name}")
        return data

    @staticmethod
    def save_graph_list_to_dataset_list(graph_list:list,num_nodes:int,dir_type:Literal['train','val','test']):
        dataset_list=[]
        for graph_id,graph in tqdm(enumerate(graph_list),desc=f"Convert {dir_type}_{num_nodes} graph_list..."):
            event_stream=GraphUtils.get_event_stream(graph=graph)
            for source_id in tqdm(graph.nodes,desc=f"Convert {graph_id} graph to dataset..."):
                dataset=GraphUtils.convert_event_stream_to_dataset(event_stream=event_stream,num_nodes=num_nodes,source_id=source_id)
                dataset_list.append(dataset)
        random.shuffle(dataset_list)
        DataUtils.save_to_pickle(data=dataset_list,file_name=f"{dir_type}_{num_nodes}",dir_type=dir_type,num_nodes=num_nodes)

    @staticmethod
    def save_graph_list_to_dataset_list_chunk(graph_list:list,graph_type:str,num_nodes:int,chunk_size:int,dir_type:Literal['train','val','test']):
        dataset_list=[]
        for graph_id,graph in tqdm(enumerate(graph_list),desc=f"Convert {dir_type} {graph_type} graph_list..."):
            event_stream=GraphUtils.get_event_stream(graph=graph)
            for source_id in tqdm(graph.nodes,desc=f"Convert {graph_id} graph to dataset..."):
                dataset=GraphUtils.convert_event_stream_to_dataset(event_stream=event_stream,num_nodes=num_nodes,source_id=source_id)
                dataset_list.append(dataset)

        file_path=os.path.join(DataUtils.dataset_path,dir_type)
        if dir_type=='test':
            file_path=os.path.join(DataUtils.dataset_path,dir_type,f"{num_nodes}")
        exist_chunk_files=[f for f in os.listdir(file_path) if f.startswith(f"{dir_type}_{num_nodes}_chunk_{chunk_size}_")]
        idx_offset=len(exist_chunk_files)

        chunk_list=[dataset_list[i:i+chunk_size] for i in range(0,len(dataset_list),chunk_size)]
        for idx,chunk in tqdm(enumerate(chunk_list),total=len(chunk_list),desc=f"Saving {dir_type}_{num_nodes}_chunk_{chunk_size}..."):
            DataUtils.save_to_pickle(data=chunk,file_name=f"{dir_type}_{num_nodes}_chunk_{chunk_size}_{idx+idx_offset}",dir_type=dir_type,num_nodes=num_nodes)
        print(f"Finish to save {graph_type} {dir_type}_{num_nodes}_chunk_{chunk_size}!")
    
    @staticmethod
    def save_model_parameter(model,model_name:str):
        file_name=model_name+".pt"
        file_path=os.path.join(DataUtils.dataset_path,"inference",file_name)
        torch.save(model.state_dict(),file_path)
        print(f"Save {model_name} model parameter")

    @staticmethod
    def load_model_parameter(model,model_name:str):
        file_name=model_name+".pt"
        file_path=os.path.join(DataUtils.dataset_path,"inference",file_name)
        model.load_state_dict(torch.load(file_path))
        return model