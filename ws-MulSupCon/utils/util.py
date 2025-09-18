
import torch
import numpy as np
from glob import glob
import h5py 
import yaml
import torch
import sys, os
import models_pretrain
from loguru import logger
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from models_pretrain import PretrainModel
import torch.nn.functional as F


def genlogger(file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if file:
        logger.add(file, enqueue=True, format=log_format)
    return logger
    
class Logger():
    def __init__(self, file, rank=0):
        self.logger = None
        self.rank = rank
        if not rank:
            self.logger = genlogger(file)
    def info(self, msg):
        if not self.rank:
            self.logger.info(msg)


# how to generate mask and score
def get_output_func(pattern='MulSupCon', with_weight=False):
    """
    Paremeters
        pattern: how to generate mask and score
            - all: labels exactly matched with the anchor
            - any: labels with at least one common label with the anchor
            - MulSupCon: treat each of anchor's label separately
        with_weight: argument for sep pattern, whether to use 1/|y| to weight the loss
    """
    # assert pattern in ['all', 'any', 'MulSupCon']

    def generate_output_MulSupCon(batch_labels, ref_labels, scores):
        """
        MulSupCon
        
        Parameters:
            batch_labels: B x C tensor, labels of the anchor
            ref_labels: Q x C tensor, labels of samples from queue (BxC)
            scores: B x Q tensor, cosine similarity between the anchor and samples from queue (BXB)
        """
        B = len(batch_labels) # batch size
        indices = torch.where(batch_labels == 1) # get desease indices, [0]: which case has desease in bs, [1] what desease it has
        scores = scores[indices[0]] # get score of the desease indices, shape: case x C
        labels = torch.zeros(len(scores), batch_labels.shape[1], device=scores.device)
        labels[range(len(labels)), indices[1]] = 1 # shape: [case, C]
        masks = (labels @ ref_labels.T).to(torch.bool)
        n_score_per_sample = batch_labels.sum(dim=1).to(torch.int16).tolist() # how many labels each sample has
        if with_weight:
            weights_per_sample = [1/(n * B) for n in n_score_per_sample for _ in range(n)] 
        else:
            # generate n times of 1/total disease for each case depend on how many desease it has
            weights_per_sample = [1 / len(scores) for n in n_score_per_sample for _ in range(n)]
        weights_per_sample = torch.tensor(
            weights_per_sample,
            device=scores.device,
            dtype=torch.float32
        )
        return scores, [masks.to(torch.long), weights_per_sample]
    
    def generate_output_SoftCon(batch_labels, ref_labels, scores):
        """
        SoftCon
        
        Parameters:
            batch_labels: B x C tensor, labels of the anchor
            ref_labels: Q x C tensor, labels of samples from queue (BxC)
            scores: B x Q tensor, cosine similarity between the anchor and samples from queue (BXB)
        """
        B = len(batch_labels) # batch size
        # construct the diagonal matrix
        mask_diag = torch.eye(batch_labels.shape[0], device=batch_labels.device)
        soft_mask = batch_labels @ ref_labels.T
        # normalize the mask
        soft_mask  = (soft_mask - soft_mask.min()) / (soft_mask.max() - soft_mask.min() + 1e-10)

        return scores, [mask_diag.to(torch.long), soft_mask]
    

    def generate_output_SimDiss(batch_label, ref_label, scores):

        mul_matrix = (batch_label @ ref_label.T)
        # calculate the similarity between the anchor and samples from queue
        S = ref_label.sum(1) # ref (anchor) label
        T = batch_label.sum(1)
        Ks = torch.stack([mul_matrix[:, x] / S[x] for x in range(len(S))], dim=1)
        kt = torch.stack([  torch.abs(mul_matrix[x] - T[x]) for x in range(len(T))  ], dim=0)
        Kt = 1 / (1 + kt )
        weight_sim_dis = Ks * Kt # shape B x Q (64*64)

        return scores, [(mul_matrix > 0).to(torch.long), weight_sim_dis]

    def generate_output_MulSupCon_iwash(batch_labels, ref_labels, scores, args):

        dataset = args.dataset
        wtype = args.output_func_w
        single_h = args.single_health

        B = len(batch_labels) # batch size
        # spit single and multi desease case
        single_desease = (batch_labels.sum(dim=1) == 1).to(torch.bool)
        single_desease_indices = torch.where(single_desease)
        single_d_indices = torch.where(batch_labels[single_desease_indices] == 1)
        multi_desease = (batch_labels.sum(dim=1) > 1).to(torch.bool)
        multi_desease_indices = torch.where(multi_desease)
        multi_d_indices = torch.where(batch_labels[multi_desease_indices] == 1)
        health = (batch_labels.sum(dim=1) == 0).to(torch.bool)
        health_indices = torch.where(health)

        # split single and multi desease case in ref_labels
        ref_single = (ref_labels.sum(dim=1) <= 1).to(torch.bool)
        ref_single_indices = torch.where(ref_single)
        ref_multi = (ref_labels.sum(dim=1) > 1).to(torch.bool)
        ref_health = (ref_labels.sum(dim=1) == 0).to(torch.bool)
        ref_health_indices = torch.where(ref_health)
        ref_multi_indices = torch.where(ref_multi)

        # health sample contrast
        scores_h = scores[health_indices[0]] # get score of the health indices, shape: case x C
        health_labels = torch.ones(len(scores_h), device=scores.device).unsqueeze(1).to(torch.float32)
        health_ref_label = (ref_labels.sum(dim=1) ==0 ).to(torch.bool)
        health_ref_label = torch.where(health_ref_label, 1, 0).unsqueeze(1).to(torch.float32)
        masks_h = (health_labels @ health_ref_label.T).to(torch.bool)

        # single 2 single   
        scores_s2s = scores[single_desease_indices[0]] # get score of the desease indices, shape: case x C
        scores_s2s = scores_s2s[:, ref_single_indices[0]]

        labels_single = torch.zeros(len(scores_s2s), batch_labels.shape[1], device=scores.device)
        labels_single[range(len(labels_single)), single_d_indices[1]] = 1 # shape: [case, C]
        ref_labels_single = ref_labels[ref_single]

        masks_s2s = (labels_single @ ref_labels_single.T).to(torch.bool)
        s2s_IoU = get_IoU(ref_labels_single, labels_single)
        s2s_IoU = torch.tensor(s2s_IoU).cuda()

        if 0 in masks_s2s.sum(1):
            zero_index_s2s = torch.where(masks_s2s.sum(1) == 0)[0].cpu()
            keep_mask = ~torch.isin(torch.arange(len(masks_s2s)), zero_index_s2s)
            masks_s2s = masks_s2s[keep_mask]
            scores_s2s = scores_s2s[keep_mask]
            s2s_IoU = s2s_IoU[keep_mask]
        else:
            zero_index_s2s = None

        n_score_per_sample_single = batch_labels[single_desease].sum(dim=1).to(torch.int16).tolist() # how many labels each sample has

        # single 2 multi
        scores_s2m = scores[single_desease_indices[0]] # get score of the desease indices, shape: case x C
        scores_s2m = scores_s2m[:, ref_multi_indices[0]]
        
        if not single_h:
            scores_s2h = scores[single_desease_indices[0]] # get score of the desease indices, shape: case x C
            scores_s2h = scores_s2h[:, ref_health_indices[0]]
            scores_s2mh = torch.cat((scores_s2m, scores_s2h), dim=1)
            ref_labels_multi = torch.cat((ref_labels[ref_multi], ref_labels[ref_health]), dim=0)
        else:
            scores_s2mh = scores_s2m
            ref_labels_multi = ref_labels[ref_multi]

        masks_s2m = (labels_single @ ref_labels_multi.T).to(torch.bool)
        s2m_IoU = torch.tensor(get_IoU(ref_labels_multi, labels_single)).cuda()

        if 0 in masks_s2m.sum(1):
            zero_index_s2m = torch.where(masks_s2m.sum(1) == 0)[0].cpu()
            keep_mask = ~torch.isin(torch.arange(len(masks_s2m)), zero_index_s2m)
            masks_s2m = masks_s2m[keep_mask]
            scores_s2mh = scores_s2mh[keep_mask]
            s2m_IoU = s2m_IoU[keep_mask]
        else:
            zero_index_s2m = None

        # multi 2 single
        # multi desease case
        indices = torch.where(batch_labels == 1)
        # muli desease indices  = all indices- single desease indices
        indices_multi = indices[0][~torch.isin(indices[0], single_desease_indices[0])] # multi desease case index in all case
        scores_m2s = scores[indices_multi] # get score of the desease indices, shape: case x C
        scores_m2s = scores_m2s[:, ref_single_indices[0]]

        labels_multi = torch.zeros(len(scores_m2s), batch_labels.shape[1], device=scores.device)
        labels_multi[range(len(labels_multi)), multi_d_indices[1]] = 1 # shape: [case, C]

        masks_m2s = (labels_multi @ ref_labels_single.T).to(torch.bool)
        multi_case_label = batch_labels[multi_desease_indices]
        m2s_IoU = get_multi_IoU(ref_labels_single, multi_case_label)
        m2s_IoU = torch.tensor(m2s_IoU).cuda()

        # original label
            
        if 0 in masks_m2s.sum(1):
            zero_index_m2s = torch.where(masks_m2s.sum(1) == 0)[0].cpu()
            keep_mask = ~torch.isin(torch.arange(len(masks_m2s)), zero_index_m2s)
            masks_m2s = masks_m2s[keep_mask]
            scores_m2s = scores_m2s[keep_mask]
            m2s_IoU = m2s_IoU[keep_mask]

        else:
            zero_index_m2s = None

        # multi 2 multi
        scores_m2m = scores[indices_multi] # get score of the desease indices, shape: case x C
        scores_m2m = scores_m2m[:, ref_multi_indices[0]]
        
        if not single_h:
            scores_m2h = scores[indices_multi] # get score of the desease indices, shape: case x C
            scores_m2h = scores_m2h[:, ref_health_indices[0]]
            scores_m2mh = torch.cat((scores_m2m, scores_m2h), dim=1)
        else:
            scores_m2mh = scores_m2m

        masks_m2m = (labels_multi @ ref_labels_multi.T).to(torch.bool)
        m2m_IoU = get_multi_IoU(ref_labels_multi, multi_case_label)
        m2m_IoU = torch.tensor(m2m_IoU).cuda()


        if 0 in masks_m2m.sum(1):
            zero_index_m2m = torch.where(masks_m2m.sum(1) == 0)[0].cpu()
            keep_mask = ~torch.isin(torch.arange(len(masks_m2m)), zero_index_m2m)
            masks_m2m = masks_m2m[keep_mask]
            scores_m2mh = scores_m2mh[keep_mask]
            m2m_IoU = m2m_IoU[keep_mask]

        else:
            zero_index_m2m = None

        n_score_per_sample_multi = batch_labels[multi_desease].sum(dim=1).to(torch.int16).tolist() # how many labels each sample has

        # get the weight for each desease
        singel_label_w, multi_label_w = args.single_w, args.mul_w

        # weight the single desease case accoding to singel_label_w and labels_single
        n_weight_s = [singel_label_w[n] for n in single_d_indices[1] ]
        # weight the multi desease case accoding to multi_label_w and labels_multi
        n_weight_m = [multi_label_w[n] for n in multi_d_indices[1] ]

        if zero_index_s2s is not None:
            n_weight_s2s = [n_weight_s[n] for n in range(len(n_weight_s)) if n not in zero_index_s2s]
        else:
            n_weight_s2s = n_weight_s

        if zero_index_s2m is not None:
            n_weight_s2m = [n_weight_s[n] for n in range(len(n_weight_s)) if n not in zero_index_s2m]
        else:
            n_weight_s2m = n_weight_s

        if zero_index_m2s is not None:
            n_weight_m2s = [n_weight_m[n] for n in range(len(n_weight_m)) if n not in zero_index_m2s]
        else:
            n_weight_m2s = n_weight_m

        if zero_index_m2m is not None:
            n_weight_m2m = [n_weight_m[n] for n in range(len(n_weight_m)) if n not in zero_index_m2m]
        else:
            n_weight_m2m = n_weight_m

        # generate n times of 1/n for each case depend on how many desease it has
        weights_per_sample_single = [1 / len(scores_s2s) for n in n_score_per_sample_single for _ in range(n)]
        weights_per_sample_single = torch.tensor(
            weights_per_sample_single,
            device=scores.device,
            dtype=torch.float32
        )

        weights_per_sample_multi = [1 / len(scores_m2m) for n in n_score_per_sample_multi for _ in range(n)]
        weights_per_sample_multi = torch.tensor(
            weights_per_sample_multi,
            device=scores.device,
            dtype=torch.float32
        )


        # output
        ref = {}
        if single_desease.sum() != 0:
            ref['scores_s2s'] = scores_s2s
            ref['scores_s2m'] = scores_s2mh
        else:
            ref['scores_s2s'] = None
            ref['scores_s2m'] = None

        if multi_desease.sum() != 0:
            ref['scores_m2s'] = scores_m2s
            ref['scores_m2m'] = scores_m2mh
        else:
            ref['scores_m2s'] = None
            ref['scores_m2m'] = None

        if health.sum() != 0:
            ref['scores_h'] = scores_h
        else:
            ref['scores_h'] = None

        ref['masks_s2s'] = masks_s2s.to(torch.long)
        ref['masks_s2m'] = masks_s2m.to(torch.long)
        ref['masks_m2s'] = masks_m2s.to(torch.long)
        ref['masks_m2m'] = masks_m2m.to(torch.long)
        ref['masks_h'] = masks_h.to(torch.long)
        
        ref['IoU_s2s'] = s2s_IoU
        ref['IoU_s2m'] = s2m_IoU
        ref['IoU_m2s'] = m2s_IoU
        ref['IoU_m2m'] = m2m_IoU

        ref['weight_single'] = weights_per_sample_single
        ref['weight_multi'] = weights_per_sample_multi

        ref['ref_labels_s'] = ref_labels_single
        ref['ref_labels_m'] = ref_labels_multi


        ref['n_weight_s2s'] = n_weight_s2s
        ref['n_weight_s2m'] = n_weight_s2m
        ref['n_weight_m2s'] = n_weight_m2s
        ref['n_weight_m2m'] = n_weight_m2m
        

        return ref
    

    def generate_output_all(batch_label, ref_label, scores):
        """
        positives: labels exactly matched with the anchor
        """
        mul_matrix = (batch_label @ ref_label.T).to(torch.int16)
        mask1 = torch.sum(batch_label, dim=1).unsqueeze(1).to(torch.int16) == mul_matrix
        mask2 = torch.sum(ref_label, dim=1).unsqueeze(1).to(torch.int16) == mul_matrix.T
        mask = mask1 & mask2.T
        return scores, mask.to(torch.long)

    def generate_output_any(batch_label, ref_label, scores):
        """
        positives: labels with at least one common label with the anchor
        """
        mul_matrix = (batch_label @ ref_label.T)
        return scores, (mul_matrix > 0).to(torch.long)

    if pattern == 'all':
        return generate_output_all
    elif pattern == 'any':
        return generate_output_any
    elif pattern == 'MulSupCon':
        return generate_output_MulSupCon
    elif pattern == 'SimDiss':
        return generate_output_SimDiss
    elif pattern == 'MulSupCon_iwash':
        return generate_output_MulSupCon_iwash
    elif pattern == 'SoftCon':
        return generate_output_SoftCon
    else:
        raise NotImplementedError
    




def get_IoU(ref_labels, batch_labels):
    """
    Calculate the intersection over union between the anchor and samples from queue
    """
    mul_matrix = (batch_labels @ ref_labels.T)
    S = ref_labels.sum(1) # ref (anchor) label
    T = batch_labels.sum(1)
    union = T.unsqueeze(1) + S.unsqueeze(0) - mul_matrix
    IoU_matrix = mul_matrix / (union + 1e-10)

    return IoU_matrix

def get_multi_IoU(ref_labels, multi_case_label):
    # cal IoU
    m_sum = multi_case_label.sum(dim=1)
    ref_sum = ref_labels.sum(dim=1)
    union = m_sum.unsqueeze(1) + ref_sum.unsqueeze(0) - (multi_case_label @ ref_labels.T)
    IoU_m = (multi_case_label @ ref_labels.T) / (union + 1e-10)
    
    # repeat the IoU_m according to m_sum
    IoU_multi = torch.tensor([], device=ref_labels.device)
    for x in range(len(IoU_m)):
        IoU_multi = torch.concat( (IoU_multi, IoU_m[x].repeat(int(m_sum.tolist()[x]), 1) ) , dim = 0)
    
    return IoU_multi





def get_model_from_pretrain(
    args,
    model_path: str,
    config: dict,
    resume: bool = False,
    load_level: str = 'backbone',
    **kwargs
):
    """
    Load model, optimizer, and scheduler from saved model
    
    """

    if model_path is None:
        model = getattr(models, config['model'])(**config['model_args'], **kwargs)
        return model, {}, {}
    
    pretrain_config = torch.load(
        glob(os.path.join(model_path, '*config*'))[0], map_location='cpu')
    if resume:
        config = pretrain_config

    model = getattr(models, config['model'])(**config['model_args'], **kwargs)

    saved = torch.load(
        glob(os.path.join(model_path, '*best*'))[0], map_location='cpu')
    params, optim_params, scheduler_params\
         = saved['model'], saved.get('optimizer', {}), saved.get('scheduler', {})
    

    pretrain_model.load_state_dict(params)
    encoder_params = pretrain_model.get_params(level=load_level)
    model.load_params(encoder_params, load_level=load_level)

    return model, optim_params, scheduler_params


def get_task_name(args):

    # task_name += f'_{args.dataset}'

    task_name = f'ws-MulSupCon_{args.backbone}_{args.dataset}_{args.mode}'

    if args.mode == 'pretrain':
        task_name += f'_{args.lr}_e{args.num_epochs}_bs{args.batch_size}_{args.scheduler}_{args.sch_step}_{args.output_func}'
     

    else:
        task_name += f'_{args.train_lr}_e{args.train_epochs}_bs{args.train_batch_size}_{args.train_scheduler}_{args.train_scheduler_mode}_{args.train_sch_step}'
        if args.load_pretrain:
            task_name += f'_lp'
        if args.enc_fixed:
            task_name += f'_ef'
        if args.train_pretrain:
            task_name += f'_pre_{args.freeze_epochs}'
        
        task_name += f'_{args.loss}'
        
        if args.load_pretrain:
            task_name += f'_{args.pretrain_path}'

                
    return task_name