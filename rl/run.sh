
# python hrl_gcspirl.py \
#     -p proposed/weights/log35_32.bin


# python hrl_caespirl.py \
#     -p proposed/weights/log34_end.bin


# python hrl_sgspirl.py \
#     -p proposed/weights/log1_45.bin


# python sac_scripts/hrl_gcid.py \
#     -p proposed/weights/log39_30.bin


# SPiRL 
# python sac_scripts/hrl_spirl.py \
#     -p proposed/weights/spirl.bin

# Goal Conditioned Inverse Dynamics
# python sac_scripts/hrl_gcid.py \
#     -p proposed/weights/log276_end.bin

# python sac_scripts/hrl_gcid.py \
#     -p proposed/weights/log278_end.bin

# non-distributional states
# python sac_scripts/hrl_gcid.py \
#     -p proposed/weights/log281_end.bin

# non-dist + beta 0.001
# beta가 넓으면 안된다. reaching이 잘 안됨. 
python sac_scripts/hrl_gcid.py \
    -p proposed/weights/log1_39.bin



# s* 를 항상 마지막 state로 준 버전. 
# python sac_scripts/hrl_gcid.py \
#     -p proposed/weights/log296_end.bin \
#     --wandb