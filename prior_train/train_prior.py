from torch.optim import *
from proposed.prior_train.trainer import *
from proposed.configs.models import *
from proposed.modules.base import *
# from proposed.models.spirl import *
from proposed.models.spirl_easy import *

import argparse



def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--recon_loss", default = "mse", choices = ["nll", "mse"])
    parser.add_argument("--reg_beta", default = 0.0005, type = float)
    parser.add_argument("--epochs", default = 70)
    parser.add_argument("--mode", default= "default")
    parser.add_argument("--ga", action = "store_true")
    parser.add_argument("--min", default = 30)
    parser.add_argument("--max", default = 50)
    parser.add_argument("--last", action = "store_true")
    parser.add_argument("--warmup", default = 20, type = int)



    args = parser.parse_args()  
    args = edict(vars(args))


    args['goal_range'] = (int(args.min), int(args.max))


    train_conf, train_loader = get_loader("aa", "train", **args)
    test_conf, test_loader = get_loader("aa", "test", **args)
    
    
    # train_conf.recon_loss = args.recon_loss
    train_conf.reg_beta = args.reg_beta
    train_conf.project = "spirl"


    m = SkillPrior(train_conf).cuda()
    trainer = BaseTrainer(m, train_conf)
    trainer.fit(train_loader, test_loader, True)

if __name__ == "__main__":
    main()