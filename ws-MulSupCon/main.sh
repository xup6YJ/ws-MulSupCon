


python pretrain.py --dataset mimic --mode pretrain  --backbone resnet50 --batch_size 64 --num_epochs 150 --scheduler COS  --lr 0.0005 \
    --output_func MulSupCon_iwash  --sm_lamdba 1.0  --weight_health 0.75
                                  
python pretrain.py --dataset mimic --mode train  --backbone resnet50 --train_epochs 100   --train_batch_size 32 --train_scheduler RP --train_lr 0.0005 --train_pretrain \
    --loss 'BCE' --train_scheduler_mode validloss \
    --load_pretrain --pretrain_path ws-MulSupCon_resnet50_mimic_pretrain_0.0005_e150_bs64_COS_iter_MulSupCon_iwash
                                    
python pretrain.py --dataset mimic --mode test  --backbone resnet50  \
    --model_path ws-MulSupCon_resnet50_mimic_train_0.0005_e100_bs32_RP_validloss_iter_lp_pre_3_BCE_ws-MulSupCon_resnet50_mimic_pretrain_0.0005_e150_bs64_COS_iter_MulSupCon_iwash



python pretrain.py --dataset CXR14 --mode pretrain  --backbone resnet50 --batch_size 64 --num_epochs 150 --scheduler COS  --lr 0.0005 \
    --output_func MulSupCon_iwash  --sm_lamdba 1.0  --weight_health 0.7
   
python pretrain.py --dataset CXR14 --mode train  --backbone resnet50 --train_epochs 100   --train_batch_size 32 --train_scheduler RP --train_lr 0.0005 --train_pretrain \
    --loss 'BCE' --train_scheduler_mode validloss \
    --load_pretrain --pretrain_path ws-MulSupCon_resnet50_CXR14_pretrain_0.0005_e150_bs64_COS_iter_MulSupCon_iwash
                                  
python pretrain.py --dataset CXR14 --mode test  --backbone resnet50  \
    --model_path ws-MulSupCon_resnet50_CXR14_train_0.0005_e100_bs32_RP_validloss_iter_lp_pre_3_BCE_ws-MulSupCon_resnet50_CXR14_pretrain_0.0005_e150_bs64_COS_iter_MulSupCon_iwash





