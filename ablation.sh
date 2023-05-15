# ours 
# python train.py \
#     --structure gc_div_joint \
#     --env maze \
#     --reg_beta 0.01 \
#     --mixin_ratio 0.05 \ # 
#     --plan_H 100 \ # 

# no futher rollout.
python train.py \
    --structure gc_div_joint \
    --env maze \
    --reg_beta 0.01 \
    --mixin_ratio 0.05 \
    --plan_H 10 

python hrl_gcid_joint.py \
    -p weights/maze/gc_div_joint/log51_end.bin \
    -rp 5 \
    --q_warmup 5000 \
    --env_name maze \
    --wandb_project_name gc_div_joint

# large ratio 
python train.py \
    --structure gc_div_joint \
    --env maze \
    --reg_beta 0.01 \
    --mixin_ratio 0.2 \ # mixin ratio 
    --plan_H 100 \ # only subtrajectory 

python hrl_gcid_joint.py \
    -p weights/maze/gc_div_joint/log51_end.bin \
    -rp 5 \
    --q_warmup 5000 \
    --env_name maze \
    --wandb_project_name gc_div_joint

# large ratio and no futher rollout
python train.py \
    --structure gc_div_joint \
    --env maze \
    --reg_beta 0.01 \
    --mixin_ratio 0.2 \ # mixin ratio 
    --plan_H 10 \ # only subtrajectory 

python hrl_gcid_joint.py \
    -p weights/maze/gc_div_joint/log51_end.bin \
    -rp 5 \
    --q_warmup 5000 \
    --env_name maze \
    --wandb_project_name gc_div_joint

# no
python train.py \
    --structure gc_div_joint \
    --env maze \
    --reg_beta 0.01 \
    --mixin_ratio 0.00 \
    --plan_H 10 

python hrl_gcid_joint.py \
    -p weights/maze/gc_div_joint/log51_end.bin \
    -rp 5 \
    --q_warmup 5000 \
    --env_name maze \
    --wandb_project_name gc_div_joint