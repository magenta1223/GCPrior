from torch.optim import *
from proposed.prior_train.trainer import *
from proposed.configs.models import *
from proposed.modules.base import *
# from proposed.models.gcspirl import *
from proposed.models.gcid_ssl import GCID_VICREG_SkillPrior

import argparse




def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--recon_loss", default = "mse", choices = ["nll", "mse"])

    parser.add_argument("--wandb", action = "store_true", default = False)
    parser.add_argument("--reg_beta", default = 0.0005, type = float)
    parser.add_argument("--epochs", default = 70, type = int)
    parser.add_argument("--mode", default= "vic")
    parser.add_argument("--min", default = 0)
    parser.add_argument("--max", default = -1)
    parser.add_argument("--warmup", default = 20, type = int)



    args = edict(vars(parser.parse_args()))
    args['goal_range'] = (int(args.min), int(args.max))

    train_conf, train_loader = get_loader("aa", "train", **args)
    test_conf, test_loader = get_loader("aa", "test", **args)
    
    
    # train_conf.recon_loss = args.recon_loss
    train_conf.reg_beta = args.reg_beta
    train_conf.project = "vic"
    train_conf.goal_range = args['goal_range']

    print(train_conf)

    m = GCID_VICREG_SkillPrior(train_conf).cuda()
    trainer = BaseTrainer(m, train_conf)
    trainer.fit(train_loader, test_loader, args.wandb)

if __name__ == "__main__":
    main()