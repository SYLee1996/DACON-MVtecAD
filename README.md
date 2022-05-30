# DACON-MVtecAD

## Final 5th | 0.89246 | EfficientNet-b7 & blending
+ 주최 및 주관: 데이콘 
+ 링크: https://dacon.io/competitions/official/235894/overview/description

----

## Summary
+ Augmentation      
    + 기본적으로 Albumentation 사용
    + Sobel filter를 이용한 증강을 추가(filter를 이용하여 edge를 추출 후 unsharp image와 합친 이미지를 만들어 사용)
        + 여러 filter 및 LBP 를 이용한 augmentation을 시도했는데, 결과적으로 Sobel filter를 이용한 경우 가장 높은 성능이 나왔습니다.
    + Train, Test 시에도 augmentation 수행     
        + (train 시 Sharpen, HueSaturationValue, FancyPCA, Emboss을 통해 이미지를 선명하게 만든 후 다른 aumgmentation을 수행)    
        + (test 시 Sharpen, HueSaturationValue, FancyPCA, Emboss만 수행)    

</br>

+ Train     
    + CosineAnnealingWarmUpRestarts - Scheduler(warmup을 하지 않고 학습)      
    + LabelSmoothing_with_CrossEntropy - Loss  
    + AdamW - optimizer
    + amp.GradScaler        
    + EarlyStopping(Loss, Acc 동시에 적용 |, & 대신 -> and, or 사용)      

</br>

+ Model     
    + EfficientNet_b7 사용
    + 5-fold 진행
    + Public 기준 상위 2개 모델 blending      
        -> EfficientNet_b7 (Public-LB: 0.89066, LR: 2e-4)      
        -> EfficientNet_b7 (Public-LB: 0.88658, LR: 1.5e-4)        
    + TTA
        -> tta.HorizontalFlip()     
        -> tta.VerticalFlip()       
        -> tta.Rotate90(angles=[0, 90, 180, 270])    

----
## Directory
        .
        ├── ANOMALY_DETECTION_INFERENCE.py
        ├── ANOMALY_DETECTION_MAIN.ipynb
        ├── ANOMALY_DETECTION_MAIN.py
        ├── ANOMALY_DETECTION_MODEL.py
        ├── ANOMALY_DETECTION_UTILS.py
        ├── RESULTS
        └── open
            ├── sample_submission.csv
            ├── test
            ├── test_df.csv
            ├── train
            └── train_df.csv

        4 directories, 8 files
---- 
## Environment 
+ (cuda10.2, cudnn7, ubuntu18.04), (cuda11.2.0, cudnn8, ubuntu18.04)
+ 사용한 Docker image는 Docker Hub에 첨부하며, 두 버전의 환경을 제공합니다.
  + https://hub.docker.com/r/lsy2026/anomaly_detection/tags
  
  
  
## Libraries
+ (cuda10.2, cudnn7, ubuntu18.04) 기준
  + python==3.9.7
  + pandas==1.3.4
  + numpy==1.20.3
  + tqdm==4.62.3
  + sklearn==0.24.2
  + cv2==4.5.5
  + albumentations==1.1.0
  + torch==1.11.2+cu102
  + torchvision==0.12.0+cu102
  + efficientnet_pytorch==0.7.1

---- 

## Usage
+ ipynb 파일을 이용하는 경우, '[Private 6th, 0.89246] EfficientNet & blending.ipynb' 파일을 실행시키면 됩니다.
+ py 파일을 이용하는 경우, 'ANOMALY_DETECTION_MAIN.py' 파일을 실행시키면 학습이 진행됩니다.



### Terminal Command Example for train
```
!python3 ANOMALY_DETECTION_MAIN.py \
--model 'efficientnet_b7' \
--batch_size 32 \
--pretrain True \
--epochs 200 \
\
\
--optimizer 'AdamW' \
--lr 2e-4 \
--lr_t 15 \
--lr_scheduler 'CosineAnnealingWarmUpRestarts' \
--gamma 0.524 \
--loss_function 'CE_with_Lb' \
--patience 10 \
--weight_decay 0.002157 \
--label_smoothing 0.8283  \
\
\
--text 'A'
```

Result: 
  
        model_save_name: A_220510_0142(efficientnet_b7_32_True__AdamW_0.0002_15_CosineAnnealingWarmUpRestarts_0.524_CE_with_Lb_10_0.002157_0.8283)_fold_
        Device: cuda
        GPU activate --> Count of using GPUs: 4
        100%|███████████████████████████████████████| 4277/4277 [01:33<00:00, 45.70it/s]
        Loaded pretrained weights for efficientnet-b7
        100%|█████████████████████████████████████████| 107/107 [02:12<00:00,  1.24s/it]
        100%|███████████████████████████████████████████| 27/27 [00:13<00:00,  1.94it/s]
        Fold : 1/5    epoch : 1/200
        TRAIN_Loss : 4.27603    TRAIN_F1 : 0.14825
        VALID_Loss : 4.26185    VALID_F1 : 0.13426    BEST : 0.13426    BEST_LOSS : 4.26185
        ==> best model saved 1 epoch / acc: 0.13426  loss: 4.26185  /  difference -0.36574 

        100%|█████████████████████████████████████████| 107/107 [01:44<00:00,  1.02it/s]
        100%|███████████████████████████████████████████| 27/27 [00:13<00:00,  1.94it/s]
        Fold : 1/5    epoch : 2/200
        TRAIN_Loss : 4.22950    TRAIN_F1 : 0.18017
        ...
        3Fold - VALID Loss:  4.167903229042336 , 3Fold - VALID F1:  0.8702906976744186
        4Fold - VALID Loss:  4.169158511691624 , 4Fold - VALID F1:  0.8850384026086272
        5Fold - VALID Loss:  4.170196762791386 , 5Fold - VALID F1:  0.9248873895024164
        k-fold Valid Loss:  4.168642227737992 , k-fold Valid F1:  0.9011274312187172


### Terminal Command Example for inference
```
!python3 ANOMALY_DETECTION_INFERENCE.py \
--model_save_name \
'A_220510_0142(efficientnet_b7_32_True__AdamW_0.0002_15_CosineAnnealingWarmUpRestarts_0.524_CE_with_Lb_10_0.002157_0.8283)_fold_' \
--model 'efficientnet_b7' \
--batch_size 32 \
--pretrain True \
--tta True \
--save_name 'A_220510_0142_fold_'
```

### Terminal Command Example for inference(blending)
```
!python3 ANOMALY_DETECTION_INFERENCE.py \
--model_save_name \
'A_220510_0142(efficientnet_b7_32_True__AdamW_0.0002_15_CosineAnnealingWarmUpRestarts_0.524_CE_with_Lb_10_0.002157_0.8283)_fold_' \
'A_220511_2011(efficientnet_b7_32_True__AdamW_0.00015_15_CosineAnnealingWarmUpRestarts_0.524_CE_with_Lb_10_0.002157_0.8283)_fold_' \
--model 'efficientnet_b7' \
--batch_size 32 \
--pretrain True \
--tta True \
--save_name 'b7(2e-4+1.5e-4)_blending'
```

