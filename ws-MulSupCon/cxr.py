import os
import csv
import warnings
import numpy as np
from tqdm import tqdm
import random
from argparse import ArgumentParser

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch_lr_finder import LRFinder

from models import API_DenseNet121
from dataloader import NIHChestLoader
from utils.metrics import measurement, sub_measurement, auc_roc_curve, compute_class_freqs
from utils.criterion import get_criterion, compute_loss, class_balanced_weight
from findLR import find_lr, get_current_lr
import init
import CSV


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_BNlayers(model, freeze_bn=True):     
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval() if freeze_bn else m.train()


def board_measurement(acc, recall, precision, f1_score, epoch, mode):
    for i, feature in enumerate(args.features):
        writer.add_scalar(f'Accuracy/{mode}_{feature}', acc[i], epoch)
        writer.add_scalar(f'Recall/{mode}_{feature}', recall[i], epoch)
        writer.add_scalar(f'Precision/{mode}_{feature}', precision[i], epoch)
        writer.add_scalar(f'F1_score/{mode}_{feature}', f1_score[i], epoch)

def auc_board(AUC, mAUC, epoch):
    writer.add_scalar('mAUC', mAUC, epoch)
    for i, feature in enumerate(args.features):
        writer.add_scalar(f'AUC/{feature}', AUC[i], epoch)

def TrainTest_board(train_value, test_value, epoch, title):
    writer.add_scalars(f'{title}/comparision', {
        'train': train_value,
        'test': test_value
    }, epoch)

# def Loss_board(train_loss, test_loss, epoch, title):
#     writer.add_scalars(f'Head{title}/comparision', {
#         'train': train_loss[0],
#         'test': test_loss[0]
#     }, epoch)
#     writer.add_scalars(f'Medium{title}/comparision', {
#         'train': train_loss[1],
#         'test': test_loss[1]
#     }, epoch)
#     writer.add_scalars(f'Tail{title}/comparision', {
#         'train': train_loss[2],
#         'test': test_loss[2]
#     }, epoch)
#     writer.add_scalars(f'HMT{title}/comparision', {
#         'train': sum(train_loss),
#         'test': sum(test_loss)
#     }, epoch)

def append_csv(file_name, data):
    with open(file_name, 'a') as f:
        CSVwriter = csv.writer(f)
        CSVwriter.writerow(data)


def train(device, train_loader, test_loader, model, criterion):
    best_f1, best_mAUC = 0.0, 0.0
    early_stop = 0

    model.train()
    scaler = GradScaler()

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.num_epochs)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.num_epochs*len(train_loader)), eta_min=1e-6)

    for epoch in range(1, args.num_epochs):

        # pretrain
        if epoch == args.freeze_epochs+1:
            for param in model.parameters():
                param.requires_grad = True

  
        with torch.set_grad_enabled(True):
            avg_loss = [0.0, 0.0, 0.0]
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

                    if args.loss == 'TwoWayLoss' or args.loss == 'ASL' or args.loss == 'TwoWayASL':
                        loss = compute_loss(args, loss, writer, epoch, i)

                scaler.scale(loss).backward()
                # scaler.scale(loss[0]).backward(retain_graph=True)
                # scaler.scale(loss[1]).backward(retain_graph=True)
                # scaler.scale(loss[2]).backward()
                scaler.step(optimizer)
                scaler.update()
                # loss.backward()
                # optimizer.step()

                avg_loss += loss.item()
                # avg_loss[0] += loss[0].item()
                # avg_loss[1] += loss[1].item()
                # avg_loss[2] += loss[2].item()
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

            scheduler.step()
        
        avg_loss = [loss / len(train_loader) for loss in avg_loss]
        train_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        f1_score = (2*tp) / (2*tp+fp+fn)
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)

        single_acc, single_recall, single_precision, single_f1 = sub_measurement(single_tp, single_tn, single_fp, single_fn)
        board_measurement(single_acc, single_recall, single_precision, single_f1, epoch, 'train')

        AUC, train_mAUC = auc_roc_curve(preds_list, true_list.long(), args.num_classes)
        print(f'↳ Train Acc.(%): {train_acc:.2f}%, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')
        print(f'↳ Train mAUC: {train_mAUC:.4f}')

        append_csv('result/train_all.csv', [round(train_acc, 2)] + (np.round([f1_score, recall, precision, train_mAUC], 4)*100).tolist() + \
                   round(avg_loss, 4))
        append_csv('result/train_single_acc.csv', np.round(single_acc, 2))
        append_csv('result/train_single_recall.csv', np.round(single_recall, 4) * 100)
        append_csv('result/train_single_precision.csv', np.round(single_precision, 4) * 100)
        append_csv('result/train_single_f1.csv', np.round(single_f1, 4) * 100)
        append_csv('result/train_true&false.csv',  [tp, tn, fp, fn, single_tp, single_tn, single_fp, single_fn])
        append_csv('result/train_AUC.csv', np.round(AUC.numpy(), 4) * 100)

        if epoch >= 5:
            test_acc, test_f1, test_loss, test_mAUC = test(test_loader, model, epoch, criterion)
            print(f'↳ Test mAUC: {test_mAUC:.4f}')

            TrainTest_board(train_acc, test_acc, epoch, 'Accuracy')
            TrainTest_board(f1_score, test_f1, epoch, 'F1_score')
            TrainTest_board(avg_loss, test_loss, epoch, 'Loss')
            TrainTest_board(train_mAUC, test_mAUC, epoch, 'mAUC')

            # early_stop = early_stop + 1 if test_mAUC < best_mAUC else 0
            # if early_stop == 5:
            #     print('Early stopping')
            #     break

            if test_f1 > best_f1:
                best_f1 = test_f1
                torch.save(model.state_dict(), f'result/best_f1_{best_f1:.4f}.pt')
            if test_mAUC > best_mAUC:
                best_mAUC = test_mAUC
                torch.save(model.state_dict(), f'result/best_mAUC_{best_mAUC:.4f}.pt')
            model.train()


        # for test loss
        # if epoch >= 5:
        #     early_stop += 1 if test_loss > best_loss else early_stop * 0
        #     if early_stop == 5:
        #         print('Early stopping')
        #         break


def test(test_loader, model, epoch, criterion):
    tp, tn, fp, fn = 1e-10, 1e-10, 1e-10, 1e-10
    single_tp, single_tn, single_fp, single_fn = [1e-10]*args.num_classes, [1e-10]*args.num_classes, [1e-10]*args.num_classes, [1e-10]*args.num_classes

    preds_list = torch.tensor([])
    true_list = torch.tensor([])
    
    with torch.set_grad_enabled(False):
        model.eval()
        avg_loss = [0.0, 0.0, 0.0]
        for i, data in enumerate(pbar := tqdm(test_loader)):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # outputs = model["feature_extractor"](images)
            # head_outputs = model["head_classifier"](outputs)
            # medium_outputs = model["medium_classifier"](outputs)
            # tail_outputs = model["tail_classifier"](outputs)
            # head_loss = criterion['head'](head_outputs, labels)
            # medium_loss = criterion['medium'](medium_outputs, labels)
            # tail_loss = criterion['tail'](tail_outputs, labels)

            # outputs = torch.cat((head_outputs[:, :4], medium_outputs[:, 4:10], tail_outputs[:, 10:]), 1)

            if args.loss == 'TwoWayLoss' or args.loss == 'TwoWayASL':
                loss = loss['class_wise'] + loss['sample_wise']
                # head_loss = head_loss['class_wise'] + head_loss['sample_wise']
                # medium_loss = medium_loss['class_wise'] + medium_loss['sample_wise']
                # tail_loss = tail_loss['class_wise'] + tail_loss['sample_wise']

            if args.loss == 'ASL':
                if args.asl_weight:
                    loss = args.asl_weight_pos * loss[0] + args.asl_weight_neg * loss[1]
                else:
                    loss = loss[0] + loss[1]

            avg_loss += loss.item()
            # avg_loss[0] += head_loss.item()
            # avg_loss[1] += medium_loss.item()
            # avg_loss[2] += tail_loss.item()

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

        val_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        f1_score = (2*tp) / (2*tp+fp+fn)
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        avg_loss = [loss / len(test_loader) for loss in avg_loss]
        
        single_acc, single_recall, single_precision, single_f1 = sub_measurement(single_tp, single_tn, single_fp, single_fn)
        board_measurement(single_acc, single_recall, single_precision, single_f1, epoch, 'test')

        AUC, mAUC = auc_roc_curve(preds_list, true_list.long(), args.num_classes)
        auc_board(AUC, mAUC, epoch)

        print (f'↳ Test Acc.(%): {val_acc:.2f}%, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')

        append_csv('result/test_all.csv', [round(val_acc, 2)] + (np.round([f1_score, recall, precision, mAUC], 4)*100).tolist() + \
                    [round(avg_loss, 4)])
        append_csv('result/test_single_acc.csv', np.round(single_acc, 2))
        append_csv('result/test_single_recall.csv', np.round(single_recall, 4)*100)
        append_csv('result/test_single_precision.csv', np.round(single_precision, 4)*100)
        append_csv('result/test_single_f1.csv', np.round(single_f1, 4)*100)
        append_csv('result/test_true&false.csv', [tp, tn, fp, fn, single_tp, single_tn, single_fp, single_fn])
        append_csv('result/test_AUC.csv', np.round(AUC.numpy(), 4)*100)

    return val_acc, f1_score, avg_loss, mAUC


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=85)
    parser.add_argument('--deterministic', action='store_true')

    # for model
    parser.add_argument('--num_classes', type=int, required=False, default=14)
    parser.add_argument('--num_head', type=int, required=False, default=14)
    parser.add_argument('--num_medium', type=int, required=False, default=14)
    parser.add_argument('--num_tail', type=int, required=False, default=14)
    parser.add_argument('--num_heads', type=int, required=False, default=1)
    parser.add_argument('--lam', type=float, required=False, default=0.2)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='')
    parser.add_argument('--logdir', default='log/temp')

    # for training
    parser.add_argument('--num_epochs', type=int, required=False, default=20)
    parser.add_argument('--freeze_epochs', type=int, required=False, default=3)
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--lr', type=float, default=1e-3) # 1e-3 0.002511886414140463 0.0030199517495930195
    parser.add_argument('--wd', type=float, default=0.001) # 1e-4

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='CXR14')
    parser.add_argument('--features', type=list, default=[])
    parser.add_argument('--train_df', type=str, default='') 
    parser.add_argument('--valid_df', type=str, default='')
    parser.add_argument('--base_path', type=str, default='')

    # for data augmentation
    parser.add_argument('--degree', type=int, default=20)
    parser.add_argument('--resize', type=int, default=224)

    # for loss function
    parser.add_argument('--loss', type=str, default='TwoWayLoss')
    parser.add_argument('--twoway_Tp', type=float, default=4.0)
    parser.add_argument('--twoway_Tn', type=float, default=1.0)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--asl_gamma_neg', type=float, default=4.0)
    parser.add_argument('--asl_gamma_pos', type=float, default=1.0)
    parser.add_argument('--asl_eps', type=float, default=1e-4)
    parser.add_argument('--asl_shift', type=float, default=0.0)
    parser.add_argument('--asl_weight', type=bool, default=False)
    parser.add_argument('--asl_weight_pos', type=float, default=1)
    parser.add_argument('--asl_weight_neg', type=float, default=0.5)
    parser.add_argument('--twa_gamma_neg', type=float, default=4.0)
    parser.add_argument('--twa_gamma_pos', type=float, default=1.0)
    parser.add_argument('--twa_eps', type=float, default=1e-4)
    parser.add_argument('--twa_neg_shift', type=float, default=0.0)
    parser.add_argument('--twa_pos_shift', type=float, default=0.0)
    parser.add_argument('--twa_slope', type=float, default=1.0)

    args = parser.parse_args()
    seed_everything(args.seed)

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

    # if args.deterministic:
        # cudnn.benchmark = False
        # cudnn.deterministic = True
        # random.seed(args.seed)
        # np.random.seed(args.seed)
        # torch.manual_seed(args.seed)
        # torch.cuda.manual_seed(args.seed)
        # os.environ["PYTHONHASHSEED"] = str(args.seed)

    torch.set_float32_matmul_precision('high')

    writer = SummaryWriter(args.logdir)

    if args.dataset == 'CXR14':
        args.features = ['Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Mass', 'Pneumothorax', 'Consolidation', 
                         'Pleural_Thickening', 'Cardiomegaly', 'Emphysema', 'Edema', 'Fibrosis', 'Pneumonia', 'Hernia']
        args.train_df = 'cxr14_train+val'
        args.valid_df = 'cxr14_test'
        args.base_path = ''
    elif args.dataset == 'MIMIC':
        args.features = ['Lung opacity', 'Pleural effusion', 'Atelectasis', 'Pneumonia', 'Cardiomegaly', 'Edema', 'Support devices', 
                         'Lung lesion', 'Enlarged cardiomediastinum', 'Consolidation', 'Pneumothorax', 'Fracture', 'Pleural other']
        args.train_df = 'mimic_train+val'
        args.valid_df = 'mimic_test'
        args.base_path = '/home/peng/workspace/DATA/mimic-cxr-jpg-2.1.0/'
    CSV.CreateCSVFile(args.features)

    # /home/peng/workspace/DATA/mimic-cxr-jpg-2.1.0/ for MIMIC
    train_data = NIHChestLoader(args.base_path, 'train', args.num_classes, args.dataset)
    # valid_data = NIHChestLoader('img', 'valid', args.num_classes)
    test_data = NIHChestLoader(args.base_path, 'test', args.num_classes, args.dataset)
    train_loader = DataLoader(dataset = train_data, batch_size = args.batch_size, num_workers=8, shuffle=True)
    # valid_loader = DataLoader(dataset = valid_data, batch_size = args.batch_size)
    test_loader = DataLoader(dataset = test_data, batch_size = args.batch_size, num_workers=8, shuffle=False)

    # pos, neg = compute_class_freqs(train_data.get_labels())
    # print(pos, neg)
    # pos, neg = compute_class_freqs(valid_data.get_labels())
    # print(pos, neg)
    # pos, neg = compute_class_freqs(test_data.get_labels())
    # print(pos, neg)

    model = API_DenseNet121(num_classes = args.num_classes)
    model.to(device)
    if args.model:
        model.load_state_dict(torch.load(args.model))

    # init_func = init.inititalize_parameters(model,init_func=torch.nn.init.xavier_normal_)

    # CBweight_train = class_balanced_weight('train').to(device)
    # CBweight_test = class_balanced_weight('test').to(device)

    loss = get_criterion(args)
    # loss.to(device)

    # lr_finder = LRFinder(model, optimizer, loss, device=device)
    # lr_finder.range_test(train_loader, end_lr=1, num_iter=1000)
    # lr_finder.plot()
    # lr_finder.reset()

    # find lr
    args.lr = find_lr(args)

    # training
    train(device, train_loader, test_loader, model, loss)
    CSV.MoveCSVFile(args.logdir)
