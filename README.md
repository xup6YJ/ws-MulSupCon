# 【MICCAI'2025】 ws-MulSupCon for Long-Tailed Medical Image Classification

Official implementation of "Weighted Stratification in Multi-Label Contrastive Learning for Long-Tailed Medical Image Classification". 

## Quick Implement
```bash
bash main.sh
```

## Pretrain
```bash
python pretrain.py --dataset CXR14 --mode pretrain  --backbone resnet50 --batch_size 64 --num_epochs 150 --scheduler COS  --lr 0.0005 \
    --output_func MulSupCon_iwash  --sm_lamdba 1.0  --weight_health 0.7
```

## Pretrain weight on Google Cloud
Pretrain weight for CXR14 and MIMIC dataset (ResNet50)
```bash
https://drive.google.com/file/d/1OC9KQmlGRh9NV-xrE7lEKWGruB1sOYPI/view?usp=sharing
```


## Train & Evaluation
```bash
python pretrain.py --dataset mimic --mode train  --backbone resnet50 --train_epochs 100   --train_batch_size 32 --train_scheduler RP --train_lr 0.0005 --train_pretrain \
    --loss 'BCE' --train_scheduler_mode validloss \
    --load_pretrain --pretrain_path ws-MulSupCon_resnet50_mimic_pretrain_0.0005_e150_bs64_COS_iter_MulSupCon_iwash
                                    
python pretrain.py --dataset mimic --mode test  --backbone resnet50  \
    --model_path ws-MulSupCon_resnet50_mimic_train_0.0005_e100_bs32_RP_validloss_iter_lp_pre_3_BCE_ws-MulSupCon_resnet50_mimic_pretrain_0.0005_e150_bs64_COS_iter_MulSupCon_iwash

```
## Citation
```bash
@inproceedings{lin2025weighted,
  title={Weighted Stratification in Multi-label Contrastive Learning for Long-Tailed Medical Image Classification},
  author={Lin, Ying-Chih and Chen, Yong-Sheng},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={677--687},
  year={2025},
  organization={Springer}
}
```

## Acknowledgements
Our code is adapted from ["MulSupCon"](https://github.com/williamzhangsjtu/MulSupCon). Great appreciation to these authors for their efforts in building the research community in multi-label contrastive learning.
