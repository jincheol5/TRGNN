# @staticmethod
# def save_graph_list_to_dataset_list(graph_list:list,graph_type:str,num_nodes:int,dir_type:Literal['train','val','test']):
#     dataset_list=[]
#     match dir_type:
#         case 'train'|'val':
#             for graph in tqdm(graph_list,desc=f"Convert {dir_type} {graph_type} graph_list..."):
#                 event_stream=GraphUtils.get_event_stream(graph=graph)
#                 source_id=random.randrange(num_nodes)
#                 dataset=GraphUtils.convert_event_stream_to_dataset(event_stream=event_stream,num_nodes=num_nodes,source_id=source_id)
#                 dataset_list.append(dataset)
#         case 'test':
#             for graph_id,graph in tqdm(enumerate(graph_list),desc=f"Convert {dir_type} {graph_type} graph_list..."):
#                 event_stream=GraphUtils.get_event_stream(graph=graph)
#                 for source_id in tqdm(graph.nodes,desc=f"Convert {graph_id} graph to dataset..."):
#                     dataset=GraphUtils.convert_event_stream_to_dataset(event_stream=event_stream,num_nodes=num_nodes,source_id=source_id)
#                     dataset_list.append(dataset)
#     DataUtils.save_to_pickle(data=dataset_list,file_name=f"{dir_type}_{num_nodes}_{graph_type}",dir_type=dir_type,num_nodes=num_nodes)