import torch
import torch.nn as nn
import torch.nn.functional as F

nINF = -100
class TwoWayLoss(nn.Module):
    def __init__(self, Tp=4., Tn=1):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x, y):
        class_mask = (y > 0).any(dim=0) #torch.Size([num_classes]) bool
        sample_mask = (y > 0).any(dim=1) #torch.Size([batch_size]) bool

        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0)) #torch.Size([batch_size, num_classes]) neg/pos => -100/0
        plogit_class = torch.logsumexp(-x/self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x/self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]
    
        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0)) #torch.Size([batch_size, num_classes]) neg/pos => 0/-100
        nlogit_class = torch.logsumexp(x/self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x/self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        return torch.nn.functional.softplus(nlogit_class + plogit_class).mean() + \
                torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        num_label = 14
        l = logits.reshape(-1)
        t = targets.reshape(-1)
        p = torch.sigmoid(l)
        p = torch.where(t >= 0.5, p, 1-p)
        logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
        loss = logp*((1-p)**self.gamma)
        loss = num_label*loss.mean()
        return loss
    
    
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, eps=1e-4, shift=0, num_classes=14):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        # self.clip = clip
        self.eps = eps
        self.shift = shift
        self.num_label = num_classes

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.shift > 0:
             xs_neg = (xs_neg + self.shift).clamp(max=1)

        '''# Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            
            loss *= one_sided_w'''

        log_pos = y * torch.log(xs_pos.clamp(min=self.eps, max=1-self.eps))
        log_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps, max=1-self.eps))
        loss_pos = -log_pos * (1 - xs_pos) ** self.gamma_pos
        loss_neg = -log_neg * (1 - xs_neg) ** self.gamma_neg

        return self.num_label*loss_pos.mean() + self.num_label*loss_neg.mean()
    

class TwoWayASL(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, eps=1e-4, shift=0, num_classes=14):
        super(TwoWayASL, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.eps = eps
        self.shift_pos = shift[0]
        # self.pos_shift = class_distribution('train+val_new', shift[0], slope)
        self.shift_neg = shift[1]
        self.num_classes = num_classes

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.shift_pos > 0:
            xs_pos = torch.sigmoid(x - self.shift_pos)
        # xs_pos = torch.sigmoid(x - shift_pos)
        if self.shift_neg > 0:
            xs_neg = (xs_neg + self.shift_neg).clamp(max=1)

        # Clamp the values to avoid log(0)
        log_pos = y * (torch.log(xs_pos.clamp(min=self.eps, max=1-self.eps)))
        log_neg = (1 - y) * (torch.log(xs_neg.clamp(min=self.eps, max=1-self.eps)))
        
        focal_loss_pos = -log_pos * (1 - xs_pos) ** self.gamma_pos
        focal_loss_neg = -log_neg * (1 - xs_neg) ** self.gamma_neg
        
        loss = focal_loss_pos + focal_loss_neg
        
        class_wise_loss = loss.sum(dim=0)
        sample_wise_loss = loss.sum(dim=1)
        return torch.nn.functional.softplus(class_wise_loss).mean() + torch.nn.functional.softplus(sample_wise_loss).mean()


def get_criterion(args):
    if args.loss == 'TwoWayLoss':
        return TwoWayLoss(Tp=args.twoway_Tp, Tn=args.twoway_Tn)
    elif args.loss == 'FocalLoss':
        return FocalLoss(gamma=args.focal_gamma)
    elif args.loss == 'BCE':
        return nn.BCEWithLogitsLoss()
    elif args.loss == 'ASL':
        return AsymmetricLoss(gamma_neg=args.asl_gamma_neg, gamma_pos=args.asl_gamma_pos, eps=args.asl_eps, shift=args.asl_shift, num_classes=args.num_classes)
    elif args.loss == 'TwoWayASL':
        return TwoWayASL(gamma_neg=args.twa_gamma_neg, gamma_pos=args.twa_gamma_pos, eps=args.twa_eps, shift=[args.twa_pos_shift, args.twa_neg_shift], num_classes=args.num_classes)
    else:
        raise ValueError(f"Not supported loss {args.loss}")





class WeightedSupCon(nn.Module):
    def __init__(self, temperature=0.1):
        super(WeightedSupCon, self).__init__()
        self.temperature = temperature

    def forward(self, score, ref, ind_weight = None):
        mask, weight = ref[0], ref[1]
        num_pos = mask.sum(1)

        if ind_weight is None:
            loss = - (torch.log((F.softmax(score / self.temperature, dim=1))) * mask).sum(1) / num_pos
            return (loss * weight).sum()
        else:
            ind_weight = torch.tensor(ind_weight).cuda()
            loss_s = (- (torch.log((F.softmax(score / self.temperature, dim=1))) * mask).sum(1) * ind_weight) / num_pos
            return (loss_s * weight).sum()
    


class SupCon(nn.Module):
    """
    Supervised Contrastive Loss
    """
    def __init__(self, temperature=0.1):
        super(SupCon, self).__init__()
        self.temperature = temperature

    def forward(self, score, mask, ind_weight = None):
        num_pos = mask.sum(1)
    
        if ind_weight is None:
            loss = - (torch.log((F.softmax(score / self.temperature, dim=1))) * mask).sum(1) / (num_pos+1e-10)
            return loss.mean()
        
        else:
            ind_weight = torch.nan_to_num(ind_weight, nan=0.0)
            loss = (- (  (torch.log((F.softmax(score / self.temperature, dim=1))) * mask) * ind_weight) .sum(1) ) / (num_pos+1e-10)

            return loss.mean()
        


class SoftCon(nn.Module):
    def __init__(self, temperature=0.1):
        super(SoftCon, self).__init__()
        self.temperature = temperature

    def forward(self, score, mask):
        diag_mask = mask[0]
        soft_mask = mask[1]

        contrast_loss = - (torch.log(  (F.softmax(score / self.temperature, dim=1))   ) * diag_mask).sum(1) 
        contrast_loss = contrast_loss.mean()

        softcon_loss = -( torch.log(F.sigmoid(score)) * soft_mask + torch.log(1 - F.sigmoid(score)) * (1 - soft_mask) )
        softcon_loss = softcon_loss.mean()

        
        return contrast_loss + softcon_loss * 0.1
    


class WeightedSupCon_sm_IoU_s(nn.Module):
    def __init__(self, temperature=0.1,  args = None):
        super(WeightedSupCon_sm_IoU_s, self).__init__()
        self.temperature = temperature
        self.output_func = args.output_func
        self.weight_health = args.weight_health


    def forward(self, ref):
        scores_s2s = ref['scores_s2s']
        scores_s2m = ref['scores_s2m']
        scores_m2s = ref['scores_m2s']
        scores_m2m = ref['scores_m2m']
        masks_s2s = ref['masks_s2s']
        masks_s2m = ref['masks_s2m']
        masks_m2s = ref['masks_m2s']
        masks_m2m = ref['masks_m2m']
        iou_s2s = ref['IoU_s2s']
        iou_s2m = ref['IoU_s2m']
        iou_m2s = ref['IoU_m2s']
        iou_m2m = ref['IoU_m2m']

        weight_s = ref['weight_single']
        weight_m = ref['weight_multi']

        ref_labels = ref['ref_labels_s'] 
        ref_labelm = ref['ref_labels_m']

        num_pos_s2s = masks_s2s.sum(1)
        num_pos_s2m = masks_s2m.sum(1)
        num_pos_m2s = masks_m2s.sum(1)
        num_pos_m2m = masks_m2m.sum(1)

        if ref.get('masks_h') is not None:
            mask_h = ref['masks_h']
            num_pos_h = mask_h.sum(1)
            scores_h = ref['scores_h']


        # if 'n_weight_s' this key is in ref, then use it
        n_weight_s2s = ref['n_weight_s2s'] 
        n_weight_s2m = ref['n_weight_s2m'] 
        n_weight_m2s = ref['n_weight_m2s']
        n_weight_m2m = ref['n_weight_m2m']
        n_weight_s2s = torch.tensor(n_weight_s2s).cuda().unsqueeze(1)
        n_weight_s2m = torch.tensor(n_weight_s2m).cuda().unsqueeze(1)
        n_weight_m2s = torch.tensor(n_weight_m2s).cuda().unsqueeze(1)
        n_weight_m2m = torch.tensor(n_weight_m2m).cuda().unsqueeze(1)


        if scores_s2s is not None:
            # class average on s2s
            soft_s2s = F.softmax(scores_s2s / self.temperature, dim=1)
            soft_s2m = F.softmax(scores_s2m / self.temperature, dim=1)

        if scores_m2s is not None:
            # class average on m2s
            soft_m2s = F.softmax(scores_m2s / self.temperature, dim=1)
            soft_m2m = F.softmax(scores_m2m / self.temperature, dim=1)

        if ref.get('masks_h') is not None:
            if scores_h is not None:
                soft_h = F.softmax(scores_h / self.temperature, dim=1)


        #single desease case
        if scores_s2s is not None:
            loss_s2s = (- (torch.log(  soft_s2s  ) *n_weight_s2s * masks_s2s).sum(1) ) / num_pos_s2s
            loss_s2m = (- (torch.log(  soft_s2m  )* iou_s2m *n_weight_s2m * masks_s2m).sum(1) ) / num_pos_s2m
                

        #multi desease case
        if scores_m2s is not None:
            loss_m2s = (- (torch.log(  soft_m2s  )* iou_m2s *n_weight_m2s * masks_m2s).sum(1) ) / num_pos_m2s
            loss_m2m = (- (torch.log(  soft_m2m  )* iou_m2m *n_weight_m2m * masks_m2m).sum(1) ) / num_pos_m2m
            
            

        loss = {}
        if scores_s2s is not None and scores_s2m is not None:
            loss['s2s'] = (loss_s2s).sum()/loss_s2s.shape[0]
            loss['s2m'] = (loss_s2m).sum()/loss_s2m.shape[0]
        if scores_m2s is not None and scores_m2m is not None:
            loss['m2s'] = (loss_m2s).sum()/loss_m2s.shape[0]
            loss['m2m'] = (loss_m2m).sum()/loss_m2m.shape[0]

        if ref.get('masks_h') is not None:
            if scores_h is not None:
                loss_h = (- (torch.log(  soft_h  ) * mask_h).sum(1) ) / num_pos_h
                loss['h'] = ((loss_h).sum()/loss_h.shape[0]) 


        return loss
    



