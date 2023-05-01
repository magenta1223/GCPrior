# python LVD/contrib/simpl/reproduce/simpl_meta_train.py \
#     kitchen \
#     -g 0 \
#     -w 0 \
#     -s spirl_pretrained_kitchen.pt \
#     -t 10\
#     -p simpl\
#     -r pretrained 


# python LVD/contrib/simpl/reproduce/spirl_fine_tune.py \
#     maze \
#     -g 0 \
#     -s spirl_pretrained_maze.pt \
#     -p spirl\
#     -r pretrained 

python LVD/contrib/simpl/reproduce/spirl_fine_tune.py \
    maze \
    -g 0 \
    -s weights/maze/sc/log7_end.bin \
    -p spirl \
    -r pretrained 




# python LVD/contrib/simpl/reproduce/simpl_fine_tune.py \
#     kitchen_single \
#     -g 0 \
#     -p simpl-finetune \
#     -m /home/magenta1223/skill-based/SiMPL/pretrained.pt \
#     -s /home/magenta1223/skill-based/SiMPL/spirl_pretrained_kitchen.pt \
