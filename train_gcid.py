from LVD.utils import seed_everything
seed_everything(777)


from torch.optim import *
import argparse

from LVD.configs.build import *
from LVD.modules.base import *
from LVD.models import *

from LVD.prior_train.trainer import *
from LVD.prior_train.trainer_diversity import *

# from trainer import *
# from ..configs.models import *
# from ..modules.base import *
# from ..models import *
# from .trainer_diversity import *

from LVD.configs.env import ENV_CONFIGS

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", default= "kitchen", type = str, choices= list(ENV_CONFIGS.keys()))
    parser.add_argument("--maze_path", default= "/home/magenta1223/skill-based/SiMPL/proposed/LVD/data/maze/maze_prep", type = str)
    parser.add_argument("--visual_encoder_path", default= "./weights/maze/wae/log21_end.bin", type = str)
    parser.add_argument("--wandb", action = "store_true", default = False)
    parser.add_argument("--structure", default= 'sc', type = str, choices= list(MODELS.keys()))
    parser.add_argument("--reg_beta", default = 0.0005, type = float, help = "Regularization Strength for training Beta-VAE SKILL enc/dec.")

    args = edict(vars(parser.parse_args()))



    


    args['only_weights'] = False

    train_conf, train_loader = get_loader("train", **args)
    _, val_loader = get_loader("val", **args)
    # _, test_loader = get_loader("test", **args) # diversity에 의한 성능 체크 용도. 실제 trainset은 어떻게 되는건지? 

    train_conf.reg_beta = args.reg_beta
    
    print(train_conf)


    m = MODELS[args.structure](train_conf).cuda()
    if "div" in args.structure:
        trainer = DiversityTrainer(m, train_conf)
    else:
        trainer = BaseTrainer(m, train_conf)
    # trainer.fit(train_loader, val_loader, test_loader, args.wandb)
    trainer.fit(train_loader, val_loader, None, args.wandb)


if __name__ == "__main__":
    main()