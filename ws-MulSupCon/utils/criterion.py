import torch
import torch.nn as nn
import pandas as pd
import numpy as np

nINF = -100

def class_balanced_weight(mode: str, beta: float = 0.999):
    df = pd.read_csv(f'order_{mode}.csv')
    sum = df.sum(axis=0)
    sum = sum[1:]
    sum = sum.tolist()

    effective_num = 1.0 - np.power(beta, sum)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(weights)

    return torch.tensor(weights).float()


class TwoWayLoss(nn.Module):
    def __init__(self, Tp=4., Tn=1):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x, y, CB_weight=1):
        class_mask = (y > 0).any(dim=0) #torch.Size([num_classes]) bool
        sample_mask = (y > 0).any(dim=1) #torch.Size([batch_size]) bool

        # x = torch.sigmoid(x)
        # CB_class = CB_weight
        # CB_sample = CB_weight.unsqueeze(0).repeat(y.shape[0], 1) * y

        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0)) #torch.Size([batch_size, num_classes]) neg/pos => -100/0
        plogit_class = torch.logsumexp(-x/self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x/self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]
        # plogit_class = torch.logsumexp(-x/self.Tp + pmask, dim=0).mul(self.Tp*CB_class)[class_mask]
        # exp_values = torch.exp(-x/self.Tp + pmask).mul(CB_sample)
        # plogit_sample = torch.log(exp_values.sum(dim=1)).mul(self.Tp)[sample_mask]
    
        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0)) #torch.Size([batch_size, num_classes]) neg/pos => 0/-100
        nlogit_class = torch.logsumexp(x/self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x/self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        # return torch.nn.functional.softplus(nlogit_class + plogit_class).mean() + \
        #         torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()

        loss = {}
        loss['plogit_class'] = plogit_class
        loss['nlogit_class'] = nlogit_class
        loss['plogit_sample'] = plogit_sample
        loss['nlogit_sample'] = nlogit_sample
        loss['class_wise'] = torch.nn.functional.softplus(nlogit_class + plogit_class).mean()
        loss['sample_wise'] = torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()

        return loss
    

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

        return self.num_label*loss_pos.mean(), self.num_label*loss_neg.mean()
    

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

        loss = {}
        loss['p_class'] = torch.nn.functional.softplus(focal_loss_pos.sum(dim=0)).mean()
        loss['n_class'] = torch.nn.functional.softplus(focal_loss_neg.sum(dim=0)).mean()
        loss['p_sample'] = torch.nn.functional.softplus(focal_loss_pos.sum(dim=1)).mean()
        loss['n_sample'] = torch.nn.functional.softplus(focal_loss_neg.sum(dim=1)).mean()
        loss['class_wise'] = torch.nn.functional.softplus(class_wise_loss).mean()
        loss['sample_wise'] = torch.nn.functional.softplus(sample_wise_loss).mean()

        return loss


def compute_loss(args, loss, writer, epoch, i):
    if args.loss == 'TwoWayLoss':
        # writer.add_scalars('Loss/logit/class', {
        #     'plogit_class': loss['plogit_class'].mean(),
        #     'nlogit_class': loss['nlogit_class'].mean()
        #     }, args.num_epochs*(epoch-1)+i+1)
        # writer.add_scalars('Loss/logit/sample', {
        #     'plogit_sample': loss['plogit_sample'].mean(),
        #     'nlogit_sample': loss['nlogit_sample'].mean()
        #     }, args.num_epochs*(epoch-1)+i+1)
        # writer.add_scalars('Loss/logit/class_and_sample', {
        #     'plogit_class': loss['plogit_class'].mean(),
        #     'nlogit_class': loss['nlogit_class'].mean(),
        #     'plogit_sample': loss['plogit_sample'].mean(),
        #     'nlogit_sample': loss['nlogit_sample'].mean()
        #     }, args.num_epochs*(epoch-1)+i+1)
        # writer.add_scalars('Loss/softplus/class_and_sample', {
        #     'class': loss['class_wise'],
        #     'sample': loss['sample_wise']
        #     }, args.num_epochs*(epoch-1)+i+1)
        loss = loss['class_wise'] + loss['sample_wise']
        writer.add_scalar('Loss', loss, args.num_epochs*(epoch-1)+i+1)
       
    if args.loss == 'ASL':
        if args.asl_weight:
            writer.add_scalars('Loss/pos_and_neg', {
            'pos': loss[0]*args.asl_weight_pos,
            'neg': loss[1]*args.asl_weight_neg
            }, args.num_epochs*(epoch-1)+i+1)
            loss = args.asl_weight_pos * loss[0] + args.asl_weight_neg * loss[1]
        else:
            writer.add_scalars('Loss/pos_and_neg', {
            'pos': loss[0],
            'neg': loss[1]
            }, args.num_epochs*(epoch-1)+i+1)
            loss = loss[0] + loss[1]
        writer.add_scalar('Loss', loss, args.num_epochs*(epoch-1)+i+1)  

    if args.loss == 'TwoWayASL':

        # writer.add_scalars('Loss/softplus/class', {
        #     'p_class': loss['p_class'],
        #     'n_class': loss['n_class']
        #     }, args.num_epochs*(epoch-1)+i+1)
        # writer.add_scalars('Loss/softplus/sample', {
        #     'p_sample': loss['p_sample'],
        #     'n_sample': loss['n_sample']
        #     }, args.num_epochs*(epoch-1)+i+1)
        # writer.add_scalars('Loss/softplus/class_and_sample', {
        #     'class': loss['class_wise'],
        #     'sample': loss['sample_wise']
        #     }, args.num_epochs*(epoch-1)+i+1)
        # loss = loss['class_wise'] + loss['sample_wise']
        # writer.add_scalar('Loss', loss, args.num_epochs*(epoch-1)+i+1)

        writer.add_scalar('Loss/class_p',  loss['p_class'] , args.num_epochs*(epoch-1)+i+1)
        writer.add_scalar('Loss/class_n',  loss['n_class'], args.num_epochs*(epoch-1)+i+1)
        writer.add_scalar('Loss/sample_p',  loss['p_sample'], args.num_epochs*(epoch-1)+i+1)
        writer.add_scalar('Loss/sample_n',  loss['n_sample'], args.num_epochs*(epoch-1)+i+1)
        writer.add_scalar('Loss/class',  loss['class_wise'], args.num_epochs*(epoch-1)+i+1)
        writer.add_scalar('Loss/sample',  loss['sample_wise'], args.num_epochs*(epoch-1)+i+1)
        loss = loss['class_wise'] + loss['sample_wise']
        writer.add_scalar('Loss/c+s', loss, args.num_epochs*(epoch-1)+i+1)


        # writer.add_scalars('Loss/Head/softplus/class', {
        #     'p_class': loss['head']['p_class'],
        #     'n_class': loss['head']['n_class']
        #     }, args.num_epochs*(epoch-1)+i+1)
        # writer.add_scalars('Loss/Head/softplus/sample', {
        #     'p_sample': loss['head']['p_sample'],
        #     'n_sample': loss['head']['n_sample']
        #     }, args.num_epochs*(epoch-1)+i+1)
        # writer.add_scalars('Loss/Head/softplus/class_and_sample', {
        #     'class': loss['head']['class_wise'],
        #     'sample': loss['head']['sample_wise']
        #     }, args.num_epochs*(epoch-1)+i+1)
        # writer.add_scalars('Loss/Mediun/softplus/class', {
        #     'p_class': loss['medium']['p_class'],
        #     'n_class': loss['medium']['n_class']
        #     }, args.num_epochs*(epoch-1)+i+1)
        # writer.add_scalars('Loss/Mediun/softplus/sample', {
        #     'p_sample': loss['medium']['p_sample'],
        #     'n_sample': loss['medium']['n_sample']
        #     }, args.num_epochs*(epoch-1)+i+1)
        # writer.add_scalars('Loss/Mediun/softplus/class_and_sample', {
        #     'class': loss['medium']['class_wise'],
        #     'sample': loss['medium']['sample_wise']
        #     }, args.num_epochs*(epoch-1)+i+1)
        # writer.add_scalars('Loss/Tail/softplus/class', {
        #     'p_class': loss['tail']['p_class'],
        #     'n_class': loss['tail']['n_class']
        #     }, args.num_epochs*(epoch-1)+i+1)
        # writer.add_scalars('Loss/Tail/softplus/sample', {
        #     'p_sample': loss['tail']['p_sample'],
        #     'n_sample': loss['tail']['n_sample']
        #     }, args.num_epochs*(epoch-1)+i+1)
        # writer.add_scalars('Loss/Tail/softplus/class_and_sample', {
        #     'class': loss['tail']['class_wise'],
        #     'sample': loss['tail']['sample_wise']
        #     }, args.num_epochs*(epoch-1)+i+1)
        # loss = [loss['head']['class_wise'] + loss['head']['sample_wise'],
        #         loss['medium']['class_wise'] + loss['medium']['sample_wise'],
        #         loss['tail']['class_wise'] + loss['tail']['sample_wise']]
        # writer.add_scalars('Loss/ALL_Loss', 
        #     {'Head': loss[0], 'Mediun': loss[1], 'Tail': loss[2]}, 
        #     args.num_epochs*(epoch-1)+i+1)
        # writer.add_scalar('Loss', sum(loss), args.num_epochs*(epoch-1)+i+1)
    return loss


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
        # TWASL['head'] = TwoWayASL(gamma_neg=args.twa_gamma_neg, gamma_pos=args.twa_gamma_pos, eps=args.twa_eps, shift=[args.twa_pos_shift*0, args.twa_neg_shift], num_classes=args.num_head)
        # TWASL['medium'] = TwoWayASL(gamma_neg=args.twa_gamma_neg, gamma_pos=args.twa_gamma_pos, eps=args.twa_eps, shift=[args.twa_pos_shift*1, args.twa_neg_shift], num_classes=args.num_medium)
        # TWASL['tail'] = TwoWayASL(gamma_neg=args.twa_gamma_neg, gamma_pos=args.twa_gamma_pos, eps=args.twa_eps, shift=[args.twa_pos_shift*2, args.twa_neg_shift], num_classes=args.num_tail)
        
        TWASL = TwoWayASL(gamma_neg=args.twa_gamma_neg, gamma_pos=args.twa_gamma_pos, eps=args.twa_eps, shift=[args.twa_pos_shift, args.twa_neg_shift])
        
        return TWASL
    else:
        raise ValueError(f"Not supported loss {args.loss}")
