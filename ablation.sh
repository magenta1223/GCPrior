# ours 일단 해서 성공률 좀 안나오는거 알아내야 함,. 
python train.py \
    --structure gc_div_joint \
    --env maze \
    --reg_beta 0.01 \
    --mixin_ratio 0.05 \
    --plan_H 100 \

python hrl_gcid_joint.py \
    -p weights/maze/gc_div_joint/log52_end.bin \
    -rp 5 \
    --q_warmup 5000 \
    --env_name maze \
    --wandb_project_name gc_div_joint \
    --ablation


# # no futher rollout.
# python train.py \
#     --structure gc_div_joint \
#     --env maze \
#     --reg_beta 0.01 \
#     --mixin_ratio 0.05 \
#     --plan_H 10 

# python hrl_gcid_joint.py \
#     -p weights/maze/gc_div_joint/log55_end.bin \
#     -rp 5 \
#     --q_warmup 5000 \
#     --env_name maze \
#     --wandb_project_name gc_div_joint \
#     --ablation

# # large ratio 
# python train.py \
#     --structure gc_div_joint \
#     --env maze \
#     --reg_beta 0.01 \
#     --mixin_ratio 0.2 \
#     --plan_H 100 \

# python hrl_gcid_joint.py \
#     -p weights/maze/gc_div_joint/log56_end.bin \
#     -rp 5 \
#     --q_warmup 5000 \
#     --env_name maze \
#     --wandb_project_name gc_div_joint \
#     --ablation



# # large ratio and no futher rollout
# python train.py \
#     --structure gc_div_joint \
#     --env maze \
#     --reg_beta 0.01 \
#     --mixin_ratio 0.2 \
#     --plan_H 10 \

# python hrl_gcid_joint.py \
#     -p weights/maze/gc_div_joint/log57_end.bin \
#     -rp 5 \
#     --q_warmup 5000 \
#     --env_name maze \
#     --wandb_project_name gc_div_joint \
#     --ablation


# # no
# python train.py \
#     --structure gc_div_joint \
#     --env maze \
#     --reg_beta 0.01 \
#     --mixin_ratio 0.00 \
#     --plan_H 10 

# python hrl_gcid_joint.py \
#     -p weights/maze/gc_div_joint/log58_end.bin \
#     -rp 5 \
#     --q_warmup 5000 \
#     --env_name maze \
#     --wandb_project_name gc_div_joint \
#     --ablation
