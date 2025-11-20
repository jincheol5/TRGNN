import networkx as nx
import numpy as np
import argparse
import torch
from trgnn import DataUtils,GraphAnalysis

def app_analysis(config:dict):
    match config['app_num']:
        case 1:
            """
            App 1.
            check_elements
            """
            print(f"<<Check {config['mode']}_{config['num_nodes']} graphs elements>>")
            graph_list_dict=DataUtils.load_from_pickle(file_name=f"{config['mode']}_{config['num_nodes']}",dir_type="graph",num_nodes=config['num_nodes'])
            for graph_type,graph_list in graph_list_dict.items():
                N_list=[]
                E_s_list=[]
                E_list=[]
                for graph in graph_list:
                    graph.remove_edges_from(nx.selfloop_edges(graph))
                    N,E_s,E=GraphAnalysis.check_elements(graph=graph)
                    N_list.append(N)
                    E_s_list.append(E_s)
                    E_list.append(E)
                print(f"{config['mode']}_{config['num_nodes']}_{graph_type} graphs mean of num_nodes: {np.mean(N_list)}")
                print(f"{config['mode']}_{config['num_nodes']}_{graph_type} graphs mean of num_static_edgs: {np.mean(E_s_list)}")
                print(f"{config['mode']}_{config['num_nodes']}_{graph_type} graphs mean of num_edge_events: {np.mean(E_list)}")
                print()

        case 2:
            """
            App 2.
            check_tR_ratio
                train_20
                val_20
                test_20
                test_50
                test_100
                test_500
            """
            print(f"<<Check {config['mode']}_{config['num_nodes']} graphs tR ratio>>")

            """
            dataset:
                raw: [seq_len,N,1]
                r: [seq_len,N,1]
                t: [seq_len,N,1]
                src: [seq_len,1]
                tar: [seq_len,1]
                n_mask: [seq_len,N]
                label: [seq_len,1]
            """
            dataset_list=DataUtils.load_from_pickle(file_name=f"{config['mode']}_{config['num_nodes']}",dir_type=config['mode'],num_nodes=config['num_nodes'])
            all_ratios=[]
            for dataset in dataset_list:
                label=dataset['label'] # [seq_len,1]
                tR_ratio=GraphAnalysis.check_tR_ratio(r=label)
                all_ratios.append(tR_ratio)
            lst=np.array(all_ratios,dtype=float)
            mean_ratio=lst.mean()
            max_ratio=lst.max()
            min_ratio=lst.min()
            print(f"{config['mode']}_{config['num_nodes']} graphs tR_ratio:{mean_ratio} max:{max_ratio} min:{min_ratio}")

if __name__=="__main__":
    """
    Execute app_analysis
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--graph_type",type=str,default="ladder")
    parser.add_argument("--mode",type=str,default="train")
    parser.add_argument("--num_nodes",type=int,default=20)
    args=parser.parse_args()

    config={
        # app 관련
        'app_num':args.app_num,
        "graph_type":args.graph_type,
        "mode":args.mode,
        "num_nodes":args.num_nodes
    }
    app_analysis(config=config)