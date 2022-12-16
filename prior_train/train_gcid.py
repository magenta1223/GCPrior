from torch.optim import *
from proposed.prior_train.trainer import *
from proposed.configs.models import *
from proposed.modules.base import *
# from proposed.models.gcspirl import *
from proposed.models.gcid import GCID_SkillPrior

import argparse
import random

from proposed.utils import seed_everything



def main():
    seed_everything()
    parser = argparse.ArgumentParser()


    # GOAL RANGE : TODO last가 아닐 경우 min-max값 사용 
    parser.add_argument("--min", default = 0)
    parser.add_argument("--max", default = -1)
    parser.add_argument("--last", action = "store_true", default= False)

    # TRAINER
    parser.add_argument("--epochs", default = 70, type = int)
    parser.add_argument("--warmup", default = 20, type = int)
    parser.add_argument("--lr", default = 0.001, type = float)

    # LOGGER
    parser.add_argument("--wandb", action = "store_true", default = False)

    # ARCHITECTURE
    parser.add_argument("--nLayers", default= 5, type = int)

    # SPiRL
    parser.add_argument("--reg_beta", default = 0.0005, type = float)

    # Proposed
    parser.add_argument("--norm", default= 'bn')
    parser.add_argument("--use_learned_state", action = "store_true", default= False)
    parser.add_argument("--direct", action = "store_true", default= False)
    parser.add_argument("--subgoal_loss", default = "prior", type = str)
    parser.add_argument("--distributional", action = "store_true", default= False)
    parser.add_argument("--state_reg", default = 0.005, type = float)
    parser.add_argument("--wde", default = 0, type = float, help="weight decay for skill encoder")
    parser.add_argument("--wdd", default = 0, type = float, help="weight decay for skill decoder")
    parser.add_argument("--wdp", default = 0, type = float, help="weight decay for prior_wrapper")

    # penalty 
    parser.add_argument("--L2", action = "store_true", default= False)

    parser.add_argument("--mode", default= "vic")



    args = edict(vars(parser.parse_args()))
    args['goal_range'] = (int(args.min), int(args.max))

    train_conf, train_loader = get_loader("aa", "train", **args)
    test_conf, test_loader = get_loader("aa", "test", **args)
    
    
    print(train_conf)

    m = GCID_SkillPrior(train_conf).cuda()
    trainer = BaseTrainer(m, train_conf)
    trainer.fit(train_loader, test_loader, args.wandb)

if __name__ == "__main__":
    main()