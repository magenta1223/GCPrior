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

    # -------------------------- DATASET -------------------------- #
    # parser.add_argument("--last", action = "store_true", default= False)
    # parser.add_argument("--mixin_ratio", default = 0.0, type = float)
    # parser.add_argument("--only_proprioceptive", action = "store_true", default= False)


    # -------------------------- TRAINER -------------------------- #
    # parser.add_argument("--epochs", default = 70, type = int)
    # parser.add_argument("--warmup", default = 0, type = int)
    # parser.add_argument("--lr", default = 0.001, type = float)
    # parser.add_argument("--batch_size", default= 1024, type = int)

    # -------------------------- LOGGER -------------------------- #
    parser.add_argument("--wandb", action = "store_true", default = False)
    # parser.add_argument("--save_ckpt", default= 20, type = int)


    # -------------------------- ARCHITECTURES -------------------------- #
    # parser.add_argument("--skill_concat", action = "store_true", default= False)
    parser.add_argument("--structure", default= 'sc', type = str, choices= list(MODELS.keys()))
    # parser.add_argument("--dynamics", action = "store_true", default= False)
    # parser.add_argument("--fe_path", default= '', type = str, help = "pretrained state encoder path")
    # parser.add_argument("--mixin_start", default= 30, type = int)
    # parser.add_argument("--skill_dim", default= 10, type = int)
    # parser.add_argument("--latent_state_dim", default= 256, type = int)
    # parser.add_argument("--nLayers", default= 5, type = int, help = "number of layers of MLP submodules")
    # parser.add_argument("--dropout", default = 0.0, type = float)
    # parser.add_argument("--plan_H", default = 100, type = int)
    # parser.add_argument("--hidden_dim", default = 128, type = int)

    # SPiRL
    # parser.add_argument("--reg_beta", default = 0.0005, type = float, help = "Regularization Strength for training Beta-VAE SKILL enc/dec.")
    # parser.add_argument("--subgoal_loss", default = "prior", type = str, help= """Loss functions for subgoal loss. Choices are 'prior'(negative log likelihood), 'reg'(kl-divergence)""")

    # parser.add_argument("--tanh", action= "store_true", default= False)
    # parser.add_argument("--rollout_method", choices= ['rollout', 'rollout2'], default= "rollout")
    # parser.add_argument("--workers", default = 14, type = int)

    # penalty 
    # parser.add_argument("--mode", default= "sc")

    # diversity controller
    # parser.add_argument("--sample_interval", default= 1, type = int)


    args = edict(vars(parser.parse_args()))

    # if args.env_name == "kitchen":
    #     # args['reg_beta'] = 0.0005
    #     args['epoch_cycles_train'] = 50
    # else:
    #     # args['reg_beta'] = 0.0001
    #     args['epoch_cycles_train'] = 25

    


    args['only_weights'] = False

    train_conf, train_loader = get_loader("train", **args)
    _, val_loader = get_loader("val", **args)
    # _, test_loader = get_loader("test", **args) # diversity에 의한 성능 체크 용도. 실제 trainset은 어떻게 되는건지? 


    
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