import os
import csv
import warnings
import numpy as np
from tqdm import tqdm
import random
from argparse import ArgumentParser
import logging

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torchvision

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch_lr_finder import LRFinder

from dataloader import NIHChestLoader
from utils.metrics import measurement, sub_measurement, auc_roc_curve, compute_class_freqs
from utils.criterion import get_criterion, compute_loss, class_balanced_weight
from utils.util import *
from findLR import find_lr, get_current_lr
import init
import CSV
import datetime
from glob import glob
from loss import *
from models_pretrain import Encoder
from models_train import Dense_pretrain, backbone_pretrain, Res_pretrain
from metrics import get_mlc_metrics
import math
import pandas as pd
from utils import ramps

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resume_training(args, model, optimizer):
    resume_path= './model/' + args.resume_path
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))

        checkpoint = torch.load(resume_path)
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                resume_path, checkpoint["epoch"]
            )
        )
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
            
def freeze_BNlayers(model, freeze_bn=True):     
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval() if freeze_bn else m.train()


def board_measurement(acc, recall, precision, f1_score, epoch, mode):
    for i, feature in enumerate(args.features):
        # writer.add_scalar('Loss/MulSupCon', loss, iter_num)

        writer.add_scalar(f'Accuracy/{mode}_{feature}', acc[i], epoch)
        writer.add_scalar(f'Recall/{mode}_{feature}', recall[i], epoch)
        writer.add_scalar(f'Precision/{mode}_{feature}', precision[i], epoch)
        writer.add_scalar(f'F1_score/{mode}_{feature}', f1_score[i], epoch)

def auc_board(AUC, mAUC, epoch):
    writer.add_scalar('mAUC', mAUC, epoch)
    for i, feature in enumerate(args.features):
        writer.add_scalar(f'AUC/{feature}', AUC[i], epoch)

def TrainTest_board(train_value, test_value, epoch, title):

    writer.add_scalar(f'{title}_Comparison/train', train_value, epoch)
    writer.add_scalar(f'{title}_Comparison/test', test_value, epoch)


def append_csv(file_name, data):
    with open(file_name, 'a') as f:
        CSVwriter = csv.writer(f)
        CSVwriter.writerow(data)


def get_current_consistency_weight(weight, current_iter):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return weight * ramps.sigmoid_rampup(current_iter, 40)


def pretrain(
    train_loader,
    resume: bool = False,
    model_path: str = None,
    is_image: bool = False,
    load_level: str = 'all',
    **kwargs):
    '''
    write the tensorboard logger
    '''

    """
    pretrain on Multi-label Dataset

    """
    assert load_level in ['all', 'encoder', 'backbone']

    n_class = args.num_classes

    logger.info('<============== Device Info ==============>')
    logger.info(f'Using device: {torch.cuda.get_device_name(0)}')
    logger.info(f'Using {args.seed} as seed')

    logger.info('<============== PreTraining ==============>')
    
    output_func = get_output_func(args.output_func, False)
    model = PretrainModel(args = args, out_dim= 256, cap_Q= 256, n_hidden= 1,
                            momentum= 0.999, n_class= args.num_classes,
                            output_func = output_func, output_func_name= args.output_func, backbone = args.backbone)
    model.cuda()

    if args.output_func == 'MulSupCon':
        criterion = WeightedSupCon()
    elif args.output_func == 'SimDiss' or args.output_func == 'any':
        criterion = SupCon()
    elif args.output_func == 'MulSupCon_iwash':    
        criterion = WeightedSupCon_sm_IoU_s(args = args)
    elif args.output_func == 'SoftCon':
        criterion = SoftCon()

    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay=0.00001)


    iter_num = 0
    best_loss = np.inf
    early_stop = 0
    # pretrain
    for epoch in range(1, args.num_epochs):

        epoch_loss = 0
        for i, data in enumerate(pbar := tqdm(train_loader)):
            inputs, labels = data

            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
            labels = labels.to(device)

            if args.output_func == 'MulSupCon_iwash':
                ref = model(im_q=inputs[0], im_k=inputs[1], labels = labels)
                loss = criterion(ref)
            else:
                score, mask = model(im_q=inputs[0], im_k=inputs[1], labels = labels)
                if args.output_func == 'SimDiss':
                    total_loss = criterion(score, mask[0], mask[1])
                elif args.output_func == 'SoftCon':
                    total_loss = criterion(score, mask)
                elif args.output_func == 'MulSupCon':
                    total_loss = criterion(score, mask)
                else:
                    total_loss = criterion(score, mask)


            if args.output_func == 'MulSupCon_iwash':
                for key in loss:
                    writer.add_scalar(f'Loss/{key}', loss[key] , iter_num)
                ls = 0
                lm = 0
                lh = 0
                if 's2s' in loss.keys():
                    ls += loss['s2s']
                if 's2m' in loss.keys():
                    ls += loss['s2m']
                if 'm2s' in loss.keys():
                    lm += loss['m2s']
                if 'm2m' in loss.keys():
                    lm += loss['m2m']
                if 'h' in loss.keys():
                    lh = loss['h']
                writer.add_scalar('Loss/single_loss', ls, iter_num)
                writer.add_scalar('Loss/multi_loss', lm, iter_num)
                writer.add_scalar('Loss/health_loss', lh, iter_num)

                dloss = 0
                for key in loss:
                    if key != 'h':
                        dloss += loss[key]
                # dloss = loss['s2s'] + loss['s2m'] + loss['m2s'] + loss['m2m']
                if 'h' in loss.keys():
                    total_loss = dloss*(1-args.weight_health) + loss['h']*args.weight_health
                else:
                    total_loss = dloss

                if torch.isnan(total_loss).any():
                    logging.info('Loss is nan')
                    logging.info(f'Loss: {total_loss:.4f}, s2s: {loss["s2s"]:.4f}, s2m: {loss["s2m"]:.4f}, m2s: {loss["m2s"]:.4f}, m2m: {loss["m2m"]:.4f}')
                    if args.output_func == 'MulSupCon_iwash': 
                        logging.info(f'h: {loss["h"]:.4f}')

                    assert not torch.isnan(total_loss).any()
                
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            for param_group in optimizer.param_groups:
                lr_ = param_group['lr'] 

            iter_num = iter_num + 1

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('Loss/MulSupCon', total_loss, iter_num)

            pbar.set_description(f'Epoch: {epoch}, Loss: {total_loss:.4f}')
            epoch_loss += total_loss.item()


        if epoch_loss < best_loss:
            best_loss = epoch_loss
            early_stop = 0
            print('Best model saved, Current epoch: ', epoch)

            save_model_path = os.path.join(snapshot_path, 'pretrain_model_dic.pth')
            logging.info("save model to {}".format(save_model_path))
            logging.info(f'Current epoch: {epoch}, Best loss: {best_loss:.4f}')

            torch.save(
                {'encoder': model.encoder_q.backbone.state_dict(),
                'proj': model.encoder_q.proj_head.state_dict()}, save_model_path)
        else:
            early_stop += 1
            print('Current Early stopping: ', early_stop)
        
        if args.lamdba_ramp and epoch < args.num_epochs / 2:
            early_stop = 0

        if early_stop == 30 and args.scheduler == 'RP':
            print('Early stopping')
            break
        if early_stop == 15 and args.scheduler != 'RP':
            print('Early stopping')
            break
        
        if args.scheduler == 'COS':
            lr_ *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.num_epochs))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_
    
    
    logger.info('<============== Finished ==============>')




def train(device, train_loader, test_loader, valid_loader, criterion, model_name = None):
    logger.info('<============== Training ==============>')
    logger.info(f'<============== LR: {args.train_lr} ==============>')
    best_f1, best_mAUC = 0.0, 0.0
    early_stop = 0
    valid_mAUC = 0.0
    best_valid_loss = np.inf

    if args.load_pretrain:

        encoder = Encoder(backbone= args.backbone, if_pretrain = False).backbone

        pretrain_path= './model/' + args.pretrain_path
        pretrain_path = os.path.join(pretrain_path, 'pretrain_model_dic.pth')
        encoder.load_state_dict(torch.load(pretrain_path)['encoder'], strict=True) 

        if args.backbone == 'DenseNet121':
            model = Dense_pretrain(encoder, args.num_classes, args.enc_fixed)
        else:
            model = Res_pretrain(encoder, args.num_classes, args.enc_fixed)
        
    else:
        model = backbone_pretrain(args.num_classes, args.backbone)


    model.cuda()
    model.train()
    scaler = GradScaler()
        
    optimizer = optim.Adam(model.parameters(), lr = args.train_lr)

    if args.train_scheduler == 'RP':
        if args.train_scheduler_mode == 'trainauc' or args.train_scheduler_mode == 'validauc':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3 , verbose=True)
        elif args.train_scheduler_mode == 'validloss' or args.train_scheduler_mode == 'trainloss':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    elif args.train_scheduler == 'OC':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.train_lr, steps_per_epoch=len(train_loader), epochs=args.train_epochs)

        
    # optionally resume from a checkpoint
    if args.resume:
        resume_training(args, model, optimizer)
        start_epoch = args.start_epoch
    else:
        start_epoch = 1

    iter_num = 0
    early_stop = 0

    if args.train_pretrain or args.enc_fixed:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    for epoch in range(start_epoch, args.train_epochs+1):
        
        # pretrain classifier when epoch < freeze_epochs
        if args.train_pretrain and not args.enc_fixed:
            if epoch == args.freeze_epochs+1:
                for param in model.parameters():
                    param.requires_grad = True


        # with torch.set_grad_enabled(True):
        train_epoch_loss = 0.0
        train_acc = 0.0
        tp, tn, fp, fn = 1e-10, 1e-10, 1e-10, 1e-10
        single_tp, single_tn, single_fp, single_fn = [1e-10]*args.num_classes, [1e-10]*args.num_classes, [1e-10]*args.num_classes, [1e-10]*args.num_classes
        preds_list = torch.tensor([])
        true_list = torch.tensor([])

        for i, data in enumerate(pbar := tqdm(train_loader)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)

                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            sub_tp, sub_tn, sub_fp, sub_fn, sub_single_tp, sub_single_tn, sub_single_fp, sub_single_fn = \
                measurement(torch.round(F.sigmoid(outputs)), labels)
            
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn
            single_tp = np.sum([single_tp, sub_single_tp], axis=0).tolist()
            single_tn = np.sum([single_tn, sub_single_tn], axis=0).tolist()
            single_fp = np.sum([single_fp, sub_single_fp], axis=0).tolist()
            single_fn = np.sum([single_fn, sub_single_fn], axis=0).tolist()

            preds_list = torch.cat((preds_list, F.sigmoid(outputs).cpu().detach()), 0)
            true_list = torch.cat((true_list, labels.cpu().detach()), 0)
            pbar.set_description(f'Epoch: {epoch}, Loss: {loss:.4f}')

            if args.train_sch_step == 'iter' and args.train_scheduler != 'RP':
                scheduler.step()

            iter_num = iter_num + 1
            train_epoch_loss += loss.item()
            writer.add_scalar('Loss/loss', loss, iter_num)


        if epoch > 0:
            # validation
            test_metrics = test(valid_loader, model, epoch, criterion, mode = 'valid')
            
            for m in test_metrics:
                writer.add_scalar(f'Metrics/{m}', test_metrics[m], epoch)

            test_f1 = test_metrics['miF1']
            test_mAUC = test_metrics['mAUC']
            valid_loss = test_metrics['valid_loss']
            if test_f1 > best_f1:
                best_f1 = test_f1
                torch.save(model.state_dict(), 
                           os.path.join(snapshot_path , 'best_f1.pt' ))
            
            if args.train_scheduler_mode == 'validauc':
                if test_mAUC > best_mAUC:
                    best_mAUC = test_mAUC
                    mAUC_best_f1 = test_f1

                    # save epoch, best mAUC, best F1-score
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, 
                            os.path.join(snapshot_path, 'best_mAUC.pth'))
                    
                    early_stop = 0
                else:
                    early_stop += 1
                print(f'↳ <<<Valid mAUC>>>: {test_mAUC:.4f}, Current best mAUC: {best_mAUC:.4f}, (F1 score: {mAUC_best_f1:.4f})')

            elif args.train_scheduler_mode == 'validloss':
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, 
                            os.path.join(snapshot_path, 'best_valid_loss.pth'))
                    logging.info(f'Current epoch: {epoch}, Best valid loss: {valid_loss:.4f}')
                    early_stop = 0

                    loss_best_f1 = test_f1
                    loss_best_mAUC = test_mAUC

                else:
                    early_stop += 1
                print(f'↳ <<<Valid loss>>>: {valid_loss:.4f}, Current best loss:{best_valid_loss:.4f}, Current mAUC: {loss_best_mAUC:.4f}, (F1 score: {loss_best_f1:.4f})')
                

            if early_stop == 7 and args.train_scheduler == 'RP':
                print('Early stopping')
                break

            
            print('Current Early stopping: ', early_stop, 'valid_loss: ', valid_loss)
            model.train()

        if args.train_scheduler == 'RP':
            if args.train_scheduler_mode == 'trainauc':
                scheduler.step(train_mAUC)
            elif args.train_scheduler_mode == 'validloss':
                scheduler.step(valid_loss)
            elif args.train_scheduler_mode == 'trainloss':
                scheduler.step(train_epoch_loss/len(train_loader))

        elif args.train_sch_step == 'epoch' and args.train_scheduler != 'RP':
            scheduler.step()
        
        for param_group in optimizer.param_groups:
            lr_ = param_group['lr'] 
        writer.add_scalar('lr', lr_, epoch)

        train_epoch_loss = train_epoch_loss / len(train_loader) 
        train_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        f1_score = (2*tp) / (2*tp+fp+fn)
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)

        single_acc, single_recall, single_precision, single_f1 = sub_measurement(single_tp, single_tn, single_fp, single_fn)
        board_measurement(single_acc, single_recall, single_precision, single_f1, epoch, 'train')

        AUC, train_mAUC = auc_roc_curve(preds_list, true_list.long(), args.num_classes)
        print(f'↳ Train Acc.(%): {train_acc:.2f}%, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}, mAUC: {train_mAUC:.4f}')
        # print(f'↳ Train )

        TrainTest_board(train_acc, test_metrics['cACC'], epoch, 'Accuracy')
        TrainTest_board(f1_score, test_metrics['miF1'], epoch, 'F1_score')
        TrainTest_board(train_epoch_loss, test_metrics['valid_loss'], epoch, 'Loss')
        TrainTest_board(train_mAUC, test_metrics['mAUC'], epoch, 'mAUC')

    print(model_name)
    
    logger.info('<============== Finished ==============>')



def test(test_loader, model, epoch, criterion, csv_path = None, mode = 'valid'):
    if mode == 'test':
        logger.info('<============== Testing ==============>')
    tp, tn, fp, fn = 1e-10, 1e-10, 1e-10, 1e-10
    single_tp, single_tn, single_fp, single_fn = [1e-10]*args.num_classes, [1e-10]*args.num_classes, [1e-10]*args.num_classes, [1e-10]*args.num_classes

    preds_list = torch.tensor([])
    true_list = torch.tensor([])
    
    with torch.set_grad_enabled(False):
        model.eval()
        avg_loss = 0.0
        for i, data in enumerate(pbar := tqdm(test_loader)):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            if mode == 'valid':
                loss = criterion(outputs, labels)
                avg_loss += loss.item()

            sub_tp, sub_tn, sub_fp, sub_fn, sub_single_tp, sub_single_tn, sub_single_fp, sub_single_fn = \
                measurement(torch.round(F.sigmoid(outputs)), labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn
            single_tp = np.sum([single_tp, sub_single_tp], axis=0).tolist()
            single_tn = np.sum([single_tn, sub_single_tn], axis=0).tolist()
            single_fp = np.sum([single_fp, sub_single_fp], axis=0).tolist()
            single_fn = np.sum([single_fn, sub_single_fn], axis=0).tolist()

            preds_list = torch.cat((preds_list, F.sigmoid(outputs).cpu().detach()), 0)
            true_list = torch.cat((true_list, labels.cpu().detach()), 0)
            

        val_acc = (tp+tn) / (tp+tn+fp+fn+ 1e-10) * 100
        f1_score = (2*tp) / (2*tp+fp+fn+ 1e-10) 
        recall = tp / (tp+fn+ 1e-10) 
        precision = tp / (tp+fp+ 1e-10)
        
        
        single_acc, single_recall, single_precision, single_f1 = sub_measurement(single_tp, single_tn, single_fp, single_fn)
        AUC, mAUC = auc_roc_curve(preds_list, true_list.long(), args.num_classes)

        if mode == 'valid':
            board_measurement(single_acc, single_recall, single_precision, single_f1, epoch, 'valid')
            auc_board(AUC, mAUC, epoch)

        # paper all metrics
        metrics = {**get_mlc_metrics(preds_list, true_list)}
        for name, metric in metrics.items():
            print('{}: {:<5.3f}'.format(name, metric))

        print (f'↳ {mode} mAUC: {mAUC:.4f}, F1-score: {f1_score:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}')

        if mode == 'test':
            # create a dataframe using featrues as coloumns
            test_result_dict = {}
            test_result_dict['disease'] = args.features
            test_result_dict['recall'] = np.round(single_recall, 4)*100
            test_result_dict['precision'] = np.round(single_precision, 4)*100
            test_result_dict['f1'] = np.round(single_f1, 4)*100
            test_result_dict['AUC'] = np.round(AUC.numpy(), 4)*100
            df_test = pd.DataFrame(test_result_dict)
            df_test.to_csv(csv_path + 'test_metrics_single.csv', index=False)


        metrics_dict = {}
        metrics_dict['cACC'] = np.round(val_acc, 4)*100
        metrics_dict['HA'] = np.round(metrics['HA'], 4)*100
        metrics_dict['ebF1'] = np.round(metrics['ebF1']*100, 2)
        metrics_dict['miF1'] = np.round(metrics['miF1']*100, 2)
        metrics_dict['maF1'] = np.round(metrics['maF1']*100, 2)
        metrics_dict['miP'] = np.round(precision*100, 2)
        metrics_dict['maP'] = np.round(np.mean(single_precision)*100, 2)
        metrics_dict['miR'] = np.round(recall*100, 2)
        metrics_dict['maR'] = np.round(np.mean(single_recall)*100, 2)
        metrics_dict['ACC'] = np.round(metrics['ACC'], 4)*100
        metrics_dict['mAUC'] = np.round(mAUC.item(), 4)*100

        if mode == 'valid':
            avg_loss = avg_loss / len(test_loader)
            metrics_dict['valid_loss'] = avg_loss
            writer.add_scalar('Loss/valid_loss', avg_loss, epoch)
        
        # create csv file
        if mode == 'test':
            df_result = pd.DataFrame(metrics_dict, index=[0])
            df_result.to_csv(csv_path + 'test_metrics.csv', index=False)
        
    return metrics_dict


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=85)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--mode', type=str, default='pretrain', choices=['train', 'pretrain', 'test'])

    # for model
    parser.add_argument('--num_classes', type=int, required=False, default=14)
    parser.add_argument('--num_heads', type=int, required=False, default=1)
    parser.add_argument('--lam', type=float, required=False, default=0.2)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='')
    parser.add_argument('--backbone', type=str, default='resnet50') 

    # for pretraining
    parser.add_argument('--num_epochs', type=int, required=False, default=150)
    parser.add_argument('--batch_size', type=int, required=False, default=64)
    parser.add_argument('--lr', type=float, default=0.0005) 
    parser.add_argument('--scheduler', type=str, default='COS', choices=['CAW', 'OC', 'RP', 'COS'])
    parser.add_argument('--sch_step', type=str, default='iter', choices=['iter', 'epoch'])
    parser.add_argument('--color_jitter', action='store_true', default= False)
    parser.add_argument('--color_jitter_value', type=float, default=0.5) 

    parser.add_argument('--output_func', type=str, default='MulSupCon_iwash', 
                        choices=['MulSupCon', 'SimDiss', 'any', 'MulSupCon_iwash', 'SoftCon'])
    parser.add_argument('--sm_lamdba', type=float, default=1.0) 
    parser.add_argument('--weight_health', type=float, default=0.75) 


    # for training
    parser.add_argument('--load_pretrain', action='store_true', default= False)
    parser.add_argument('--pretrain_path', type=str, default='resnet50_CXR14_pretrain_0.0005_e150_bs64_COS_iter_MulSupCon')
    parser.add_argument('--train_epochs', type=int, required=False, default=10)
    parser.add_argument('--train_pretrain', action='store_true', default= False)
    parser.add_argument('--freeze_epochs', type=int, required=False, default=3)
    parser.add_argument('--train_batch_size', type=int, required=False, default=32)
    parser.add_argument('--train_lr', type=float, default=5e-3) 
    parser.add_argument('--enc_fixed', action='store_true', default= False)
    parser.add_argument('--train_scheduler', type=str, default='RP', choices=['CAW', 'OC', 'RP'])
    parser.add_argument('--train_scheduler_mode', type=str, default='validloss', choices=['trainauc', 'trainloss', 'validloss', 'validauc'])
    parser.add_argument('--train_sch_step', type=str, default='iter', choices=['iter', 'epoch'])


    # for testing
    parser.add_argument('--test_base', action='store_true', default= False)
    parser.add_argument('--model_path', type=str, 
                        default='DenseNet121_CXR14_train_0.0005_e30_bs32_RP_validloss_iter_nc1_lp_pre_3_BCE_DenseNet121_CXR14_pretrain_0.0005_e150_bs64_COS_iter_MulSupCon_ss_smw_wB_smlb_0.5')

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='CXR14', choices=['CXR14', 'mimic'])
    parser.add_argument('--features', type=list, default=[])
    parser.add_argument('--train_df', type=str, default='') 
    parser.add_argument('--valid_df', type=str, default='')
    parser.add_argument('--base_path', type=str, default='')

    # for data augmentation
    parser.add_argument('--degree', type=int, default=20)
    parser.add_argument('--resize', type=int, default=224)

    # for loss function
    parser.add_argument('--loss', type=str, default='BCE', choices=['BCE', 'TwoWayLoss', 'ASL', 'TwoWayASL'])


    args = parser.parse_args()
    seed_everything(args.seed)

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')
    torch.set_float32_matmul_precision('high')

    if args.dataset == 'CXR14':
        args.features = ['Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Mass', 'Pneumothorax', 'Consolidation', 
                         'Pleural_Thickening', 'Cardiomegaly', 'Emphysema', 'Edema', 'Fibrosis', 'Pneumonia', 'Hernia']
        args.train_df = 'cxr14_train+val'
        args.valid_df = 'cxr14_test'
        args.base_path = ''
        args.num_classes = 14
    elif args.dataset == 'mimic' or args.dataset == 'mimic_all' or args.dataset == 'mimic_front':
        args.features = ['Lung opacity', 'Pleural effusion', 'Atelectasis', 'Pneumonia', 'Cardiomegaly', 'Edema', 'Support devices', 
                         'Lung lesion', 'Enlarged cardiomediastinum', 'Consolidation', 'Pneumothorax', 'Fracture', 'Pleural other']
        args.train_df = 'mimic_train+val'
        args.valid_df = 'mimic_test'
        args.num_classes = 13


    if args.mode != 'test':
        model_name = get_task_name(args)
        model_path = "./"+'model'+"/" 
        snapshot_path = os.path.join(model_path, model_name)
        print('model name: ', model_name)
        if not os.path.exists(snapshot_path) :
            os.makedirs(snapshot_path)

        writer = SummaryWriter("./log/"+model_name )
        logging.basicConfig(filename=snapshot_path + "/log.txt",
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    else:
        model_path = "./"+'model'+"/" 
        snapshot_path = os.path.join(model_path, args.model_path)


    # training
    if args.mode == 'train':
        # Data loader
        train_data = NIHChestLoader(args, args.base_path, 'train', args.num_classes, args.dataset)
        test_data = NIHChestLoader(args, args.base_path, 'test', args.num_classes, args.dataset)
        valid_data = NIHChestLoader(args, args.base_path, 'valid', args.num_classes, args.dataset)

        train_loader = DataLoader(dataset = train_data, batch_size = args.train_batch_size, num_workers=8, shuffle=True)
        test_loader = DataLoader(dataset = test_data, batch_size = 32, num_workers=8, shuffle=False)
        valid_loader = DataLoader(dataset = valid_data, batch_size = 32, num_workers=8, shuffle=False)

        train(device, train_loader, test_loader, valid_loader, criterion = get_criterion(args), model_name = model_name)
        

    elif args.mode == 'pretrain':

        pretrain_single_data = NIHChestLoader(args, args.base_path, 'pretrain', args.num_classes, args.dataset)

        '''
        df format: img_path, disease1, disease2, ..., diseasen
        '''
        df = pretrain_single_data.df
        # df drop img_path col
        df = df.drop(columns = ['img_path'])
        # sum over disease columns
        disease_prevelance = df.sum(axis=0)
        total_disease = disease_prevelance.sum()
        args.single_w = ((total_disease - disease_prevelance) / total_disease).tolist()

        # calculate single disease in each disease
        single_d = df[df.sum(axis=1) == 1]
        single_d_num = single_d.sum(axis=0)
        mul_d_num = disease_prevelance - single_d_num
        mul_d_total = mul_d_num.sum()
        args.mul_w = ((mul_d_total - mul_d_num) / mul_d_total).tolist()


        train_sampler = None
        pretrain_loader = torch.utils.data.DataLoader(
            pretrain_single_data,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=8,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
    )
        pretrain(pretrain_loader)



    elif args.mode == 'test':
        test_data = NIHChestLoader(args, args.base_path, 'test', args.num_classes, args.dataset)
        test_loader = DataLoader(dataset = test_data, batch_size = 1, num_workers=8, shuffle=False)
        
        if args.test_base:
            model = backbone_pretrain(args.num_classes, args.backbone)
        else:
            encoder = Encoder(backbone= args.backbone, if_pretrain = False).backbone
            if args.backbone == 'DenseNet121':
                model = Dense_pretrain(encoder, args.num_classes)
            else:
                model = Res_pretrain(encoder, args.num_classes)

        print('Load model name: ', args.model_path)
        model.load_state_dict(torch.load(os.path.join(snapshot_path, 'best_valid_loss.pth'))['state_dict'])
        model.cuda()
        
        # csv_path = 'result/' + model_name + '/'
        csv_path = os.path.join(snapshot_path, 'result/')
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        # CSV.CreateCSVFile(args.features, csv_path)

        test(test_loader, model, 0, criterion = get_criterion(args), csv_path= csv_path, mode = 'test')
        # CSV.MoveCSVFile(snapshot_path)
