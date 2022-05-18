import os 
import random
import argparse
import warnings
import numpy as np
import pandas as pd
import ttach as tta 

from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ANOMALY_DETECTION_MODEL import Network
from ANOMALY_DETECTION_UTILS import str2bool, img_load, Custom_dataset 

warnings.filterwarnings(action='ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch Inference', add_help=False)

    # Inference parameters
    parser.add_argument('--model_save_name', nargs='+', default='load_model', type=str)
    parser.add_argument('--model', default='efficientnet_b7', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--pretrain', default=True, type=str2bool)
    parser.add_argument('--n_fold', default=5, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--device', default='0,1,2,3', type=str)
    parser.add_argument('--tta', default=True, type=str2bool)
    parser.add_argument('--save_name', default='default', type=str)

    return parser


def main(args):

    seed = 10   
    config = {
        # Inference parameters
        'model_save_name': args.model_save_name,
        'model': args.model,
        'batch_size': args.batch_size,
        'pretrain': args.pretrain,
        'n_fold': args.n_fold,
        'num_workers': args.num_workers,
        'device': args.device,
        'tta': args.tta,
        'save_name': args.save_name,
        }
    
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
    test_png = sorted(glob('./open/test/*.png'))
    train_y = pd.read_csv("./open/train_df.csv")
    train_labels = train_y["label"]
    label_unique = sorted(np.unique(train_labels))
    label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
    train_labels = [label_unique[k] for k in train_labels]

    test_imgs = [img_load(n) for n in tqdm(test_png)]
    config['class_num'] = len(label_unique)
    
    # Test
    Test_dataset = Custom_dataset(np.array(test_imgs), np.array(["tmp"]*len(test_imgs)), mode='test')
    Test_loader = DataLoader(Test_dataset, batch_size=config['batch_size'], pin_memory=True,
                            num_workers=config['num_workers'], prefetch_factor=config['batch_size']*2, 
                            shuffle=False)
    
    transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
        ])
    
    models = []
    tta_models = []
    
    for model_name in config['model_save_name']:
        for fold in range(config['n_fold']):
            model_dict = torch.load('./RESULTS/'+model_name + str(fold+1) + ".pt")
            model = Network(config).to(config['device']) 
            model = nn.DataParallel(model).to(config['device'])
            model.module.load_state_dict(model_dict) if torch.cuda.device_count() > 1 else model.load_state_dict(model_dict)
            
            tta_model = tta.ClassificationTTAWrapper(model, transforms, merge_mode='sum').to(config['device'])
            models.append(model)
            tta_models.append(tta_model)

    results = []
    for batch_id, batch in tqdm(enumerate(Test_loader), total=len(Test_loader)):
        x = torch.tensor(batch['img'], dtype = torch.float32, device = device)
        raw_x = torch.tensor(batch['raw_img'], dtype = torch.float32, device = device)
        
        if config['tta'] == True:
            for fold, (model, tta_model) in enumerate(zip(models, tta_models)):
                model.eval()
                tta_model.eval() 
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        if fold == 0:
                            
                            output = model(x)+tta_model(batch['raw_img'])
                        else:
                            output = output+model(x)+tta_model(batch['raw_img'])
                            
        elif config['tta'] == False:
            for fold, model in enumerate(models):
                model.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        if fold == 0:
                            output = model(x)
                        else:
                            output = output+model(x)
        output = len(models)*output
        output = torch.tensor(torch.argmax(output, axis=-1), dtype=torch.int32).cpu().numpy()
        results.extend(output)
    
    label_decoder = {val:key for key, val in label_unique.items()}
    results = [label_decoder[result] for result in results]
    submission = pd.read_csv("./open/sample_submission.csv")
    submission["label"] = results
    
    if config['tta'] == True:
        submission.to_csv("./RESULTS/{}.csv".format(config['save_name'] + 'TTA_TRUE'), index=False)
    else:
        submission.to_csv("./RESULTS/{}.csv".format(config['save_name']), index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
