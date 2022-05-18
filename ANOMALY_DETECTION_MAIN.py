import os 
import copy
import random
import argparse
import datetime
import warnings
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ANOMALY_DETECTION_MODEL import Network
from ANOMALY_DETECTION_UTILS import str2bool, img_load, Custom_dataset, score_function, CosineAnnealingWarmUpRestarts, EarlyStopping, SmoothCrossEntropyLoss

warnings.filterwarnings(action='ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch Training', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='efficientnet_b3', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--pretrain', default=True, type=str2bool)

    # Optimizer parameters
    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_t', default=10, type=int)
    parser.add_argument('--lr_scheduler', default='CosineAnnealingLR', type=str)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--loss_function', default='CE_with_Lb', type=str)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--weight_decay', default=0.005, type=float)
    parser.add_argument('--label_smoothing', default=0.3, type=float)

    # Training parameters
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--n_fold', default=5, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--text', default='default', type=str)
    parser.add_argument('--device', default='0,1,2,3', type=str)

    return parser


def main(args):

    seed = 10
    suffix = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%y%m%d_%H%M")

    config = {
        # Model parameters
        'model': args.model,
        'batch_size': args.batch_size,
        'pretrain': args.pretrain,
        
        # Optimizer parameters
        'optimizer': args.optimizer,
        'lr': args.lr,
        'lr_t': args.lr_t,
        'lr_scheduler': args.lr_scheduler,
        'gamma': args.gamma,
        'loss_function': args.loss_function,
        'patience': args.patience,
        'weight_decay': args.weight_decay,
        'label_smoothing': args.label_smoothing,
        
        # Training parameters
        'epochs': args.epochs,
        'n_fold': args.n_fold,
        'num_workers': args.num_workers,
        'text': args.text,
        'device': args.device
        }
    
    model_save_name='./RESULTS/'+config['text']+"_"+suffix+"("+ str(config['model'])+"_"+\
                                                                str(config['batch_size'])+"_"+\
                                                                str(config['pretrain'])+"__"+\
                                                                str(config['optimizer'])+"_"+\
                                                                str(config['lr'])+"_"+\
                                                                str(config['lr_t'])+"_"+\
                                                                str(config['lr_scheduler'])+"_"+\
                                                                str(config['gamma'])+"_"+\
                                                                str(config['loss_function'])+"_"+\
                                                                str(config['patience'])+"_"+\
                                                                str(config['weight_decay'])+"_"+\
                                                                str(config['label_smoothing'])+")_fold_"
                                                            
    config['model_save_name'] = model_save_name
    print('model_save_name: '+config['model_save_name'].split("/")[-1])
    # -------------------------------------------------------------------------------------------

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print('Device: %s' % device)
    if (device.type == 'cuda') or (torch.cuda.device_count() > 1):
        print('GPU activate --> Count of using GPUs: %s' % torch.cuda.device_count())
    config['device'] = device
    # -------------------------------------------------------------------------------------------

    # Dataload
    train_png = sorted(glob('./open/train/*.png'))    
    train_y = pd.read_csv("./open/train_df.csv")

    train_labels = train_y["label"]
    label_unique = sorted(np.unique(train_labels))
    label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
    train_labels = [label_unique[k] for k in train_labels]

    train_imgs = [img_load(m) for m in tqdm(train_png)]
    config['class_num'] = len(label_unique)

    # Cross Validation
    kfold = StratifiedKFold(n_splits=config['n_fold'],shuffle=True,random_state=seed)
    n_fold = config['n_fold']
    k_train_f1, k_valid_f1 = [], []   
    
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_imgs,train_labels)):

        Train_set = [train_imgs[i] for i in train_idx]
        Valid_set = [train_imgs[i] for i in valid_idx]
        Train_label_set = [train_labels[i] for i in train_idx]
        Valid_label_set = [train_labels[i] for i in valid_idx]

        # Train
        Train_dataset = Custom_dataset(np.array(Train_set), np.array(Train_label_set), mode='train')
        Train_loader = DataLoader(Train_dataset, batch_size=config['batch_size'], pin_memory=True,
                                num_workers=config['num_workers'], prefetch_factor=config['batch_size']*2, 
                                shuffle=True)

        # Valid
        Valid_dataset = Custom_dataset(np.array(Valid_set), np.array(Valid_label_set), mode='test')
        Valid_loader = DataLoader(Valid_dataset, batch_size=config['batch_size'], pin_memory=True,
                                num_workers=config['num_workers'], prefetch_factor=config['batch_size']*2, 
                                shuffle=True)
        
        model = Network(config).to(config['device'])
        model = nn.DataParallel(model).to(config['device'])

        if config['lr_scheduler'] == 'CosineAnnealingLR':
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['lr_t'], eta_min=0)
            
        elif config['lr_scheduler'] == 'CosineAnnealingWarmUpRestarts':
            optimizer = torch.optim.AdamW(model.parameters(), lr=0, weight_decay=config['weight_decay'])
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=config['lr_t'], eta_max=config['lr'], gamma=config['gamma'], T_mult=1, T_up=0)
        
        criterion = SmoothCrossEntropyLoss(smoothing=config['label_smoothing']).to(config['device'])
        scaler = torch.cuda.amp.GradScaler() 
        early_stopping = EarlyStopping(patience=config['patience'], mode='max')
        early_stopping_loss = EarlyStopping(patience=config['patience'], mode='min')
        
        best=0.5
        best_loss=100
        each_fold_train_loss, each_fold_train_f1 = [], []
        each_fold_valid_loss, each_fold_valid_f1 = [], []
        epochs = config['epochs']
        
        for epoch in range(epochs):
            train_loss, train_pred, train_real = 0, [], []
            valid_loss, valid_pred, valid_real = 0, [], []

            model.train()
            for batch_id, batch in tqdm(enumerate(Train_loader), total=len(Train_loader)):
                
                optimizer.zero_grad()
                x = torch.tensor(batch['img'], dtype=torch.float32).to(config['device'])
                y = torch.tensor(batch['label'], dtype=torch.long).to(config['device'])

                with torch.cuda.amp.autocast():
                    pred = model(x)
                loss = criterion(pred, y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                train_real += y.detach().cpu().numpy().tolist()
                
            train_loss = train_loss/len(Train_loader)
            train_f1 = score_function(train_real, train_pred)
            each_fold_train_loss.append(train_loss)
            each_fold_train_f1.append(train_f1)
            scheduler.step()

            model.eval()
            for batch_id, val_batch in tqdm(enumerate(Valid_loader), total=len(Valid_loader)):
                with torch.no_grad():
                    val_x = torch.tensor(val_batch['img'], dtype=torch.float32).to(config['device'])
                    val_y = torch.tensor(val_batch['label'], dtype=torch.long).to(config['device'])

                    val_pred = model(val_x)
                    val_loss = criterion(val_pred, val_y)

                valid_loss += val_loss.item()
                valid_pred += val_pred.argmax(1).detach().cpu().numpy().tolist()
                valid_real += val_y.detach().cpu().numpy().tolist()

            
            valid_loss = valid_loss/len(Valid_loader)
            valid_f1 = score_function(valid_real, valid_pred)
            each_fold_valid_loss.append(valid_loss)
            each_fold_valid_f1.append(valid_f1)
            
            print_best = 0    
            if (each_fold_valid_f1[-1] >= best) or (each_fold_valid_loss[-1] <= best_loss):
                difference = each_fold_valid_f1[-1] - best
                if (each_fold_valid_f1[-1] >= best):
                    best = each_fold_valid_f1[-1] 
                if (each_fold_valid_loss[-1] <= best_loss):
                    best_loss = each_fold_valid_loss[-1]
                
                pprint_best = each_fold_valid_f1[-1]
                pprint_best_loss = each_fold_valid_loss[-1]
                
                best_idx = epoch+1
                model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.module.state_dict()
                best_model_wts = copy.deepcopy(model_state_dict)
                
                # load and save best model weights
                model.module.load_state_dict(best_model_wts)
                torch.save(best_model_wts, config['model_save_name'] + str(fold+1) + ".pt")
                print_best = '==> best model saved %d epoch / acc: %.5f  loss: %.5f  /  difference %.5f'%(best_idx, pprint_best, pprint_best_loss, difference)

            print(f'Fold : {fold+1}/{n_fold}    epoch : {epoch+1}/{epochs}')
            print(f'TRAIN_Loss : {train_loss:.5f}    TRAIN_F1 : {train_f1:.5f}')
            print(f'VALID_Loss : {valid_loss:.5f}    VALID_F1 : {valid_f1:.5f}    BEST : {pprint_best:.5f}    BEST_LOSS : {pprint_best_loss:.5f}')
            print('\n') if type(print_best)==int else print(print_best,'\n')

            if early_stopping.step(torch.tensor(each_fold_valid_f1[-1])) and early_stopping_loss.step(torch.tensor(each_fold_valid_loss[-1])):
                break
            
        print("VALID Loss: ", pprint_best_loss, ", VALID F1: ", pprint_best)
            
        k_train_f1.append(pprint_best_loss)
        k_valid_f1.append(pprint_best)
        
    
    print(config['model_save_name'].split("/")[-1] + ' is saved!')
    
    print("1Fold - VALID Loss: ", k_train_f1[0], ", 1Fold - VALID F1: ", k_valid_f1[0])
    print("2Fold - VALID Loss: ", k_train_f1[1], ", 2Fold - VALID F1: ", k_valid_f1[1])
    print("3Fold - VALID Loss: ", k_train_f1[2], ", 3Fold - VALID F1: ", k_valid_f1[2])
    print("4Fold - VALID Loss: ", k_train_f1[3], ", 4Fold - VALID F1: ", k_valid_f1[3])
    print("5Fold - VALID Loss: ", k_train_f1[4], ", 5Fold - VALID F1: ", k_valid_f1[4])
    
    print("k-fold Valid Loss: ",np.mean(k_train_f1),", k-fold Valid F1: ",np.mean(k_valid_f1))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)


