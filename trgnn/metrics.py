import torch
import torch.nn.functional as F

class Metrics:
    @staticmethod
    def compute_tR_loss(logit_list:list,label_list:list):
        """
        Input:
            logit: List of [B,1]
            label: List of [B,1]
        Output:
            loss scalar tensor: [] (0차원)
        """
        loss_list=[]
        for logit,label in zip(logit_list,label_list):
            step_loss=F.binary_cross_entropy_with_logits(
                logit,label.float(),reduction='mean'
            )
            loss_list.append(step_loss)
        loss=torch.stack(loss_list).mean()
        return loss

    @staticmethod
    def compute_tR_acc(logit_list:list,label_list:list):
        """
        Input:
            logit: List of [B,1]
            label: List of [B,1]
        Output:
            Accuracy
        """
        acc_list=[]
        for logit,label in zip(logit_list,label_list):
            pred=torch.sigmoid(logit) 
            pred_label=(pred>=0.5).float()   
            correct=(pred_label==label).float()
            step_acc=correct.mean()
            acc_list.append(step_acc)
        acc=torch.stack(acc_list).mean() 
        return acc.cpu().item()
    
    @staticmethod
    def compute_tR_MCC(logit_list:list,label_list:list,eps=1e-8):
        """
        Input:
            logit: List of [B,1]
            label: List of [B,1]
        Output:
            MCC
        """
        # 1) 리스트를 하나로 쭉 이어 붙이기 (B가 서로 달라도 문제 없음)
        logit_all=torch.cat(logit_list,dim=0) # [all_B,1]
        label_all=torch.cat(label_list,dim=0) # [all_B,1]

        # 2) 시그모이드 후 0/1로 변환
        pred=torch.sigmoid(logit_all)
        pred_label=(pred>=0.5).float()

        # 3) TP,TN,FP,FN 계산
        tp=((pred_label==1)&(label_all==1)).sum().float()
        tn=((pred_label==0)&(label_all==0)).sum().float()
        fp=((pred_label==1)&(label_all==0)).sum().float()
        fn=((pred_label==0)&(label_all==1)).sum().float()

        # 4) MCC numerator & denominator
        numerator=(tp*tn)-(fp*fn)
        denominator=torch.sqrt(
            (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
        )

        # 5) 분모가 0이면 MCC = 0 처리 (관례적 방식)
        mcc=numerator/(denominator+eps)
        return mcc.cpu().item()