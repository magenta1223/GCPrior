# from proposed.utils import seed_everything
# seed_everything(666)


from torch.optim import *
from LVD.prior_train.trainer import *
from LVD.configs.build import *
from LVD.modules.base import *
from LVD.data import *
from LVD.models import *
from torch.utils.data import DataLoader

import argparse
import random





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default= "kitchen", type = str, choices= list(ENV_CONFIGS.keys()))

    # -------------------------- TRAINER -------------------------- #
    # parser.add_argument("--epochs", default = 70, type = int)
    # parser.add_argument("--warmup", default = 0, type = int)
    # parser.add_argument("--lr", default = 0.001, type = float)
    parser.add_argument("--structure", default= "ae", type = str)

    # -------------------------- LOGGER -------------------------- #
    parser.add_argument("--wandb", action = "store_true", default = False)

    # -------------------------- ARCHITECTURES -------------------------- #
    # parser.add_argument("--nLayers", default= 5, type = int, help = "number of layers of MLP submodules")
    # parser.add_argument("--norm", default= 'bn', help = "Normalization layer type. Vaild choices are 'bn' and 'ln'.")
    # parser.add_argument("--mode", default= "gcid")

    # parser.add_argument("--batch_size", default = 512, type = int)
    # parser.add_argument("--latent_dim", default = 128, type = int)
    # parser.add_argument("--hidden_dim", default = 128, type = int)
    # parser.add_argument("--workers", default = 8, type = int)
    # parser.add_argument("--recon_loss", default= "mse", choices= ['mse', 'huber', 'l1'])

    # penalty 
    args = edict(vars(parser.parse_args()))

    train_conf, train_loader = get_loader("train", **args)
    test_conf, val_loader = get_loader("val", **args)

    if args.env_name == "maze":
        train_conf['state_dim'] = 32 * 32
        train_conf['batch_size'] = 10
        train_conf['num_workers'] = 0


    DATASET_CLS = {
        "kitchen" : Kitchen_AEDataset,
        "calvin" : CALVIN_AEDataset,
    }


    # train_dataset = DATASET_CLS[args.env_name](train_conf.data_dir,  train_conf.data, resolution = 64, phase= "train", shuffle= True, n_obj = train_conf.n_obj, n_env = train_conf.n_env)
    # val_dataset = DATASET_CLS[args.env_name](train_conf.data_dir,  train_conf.data, resolution = 64, phase= "val", shuffle= False, n_obj = train_conf.n_obj, n_env = train_conf.n_env)
    

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=train_conf.batch_size,
    #     shuffle=train_dataset.shuffle,
    #     num_workers=args.workers,
    #     drop_last=False,
    #     pin_memory= True, 
    #     worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x)
    #     )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=train_conf.batch_size,
    #     shuffle=val_dataset.shuffle,
    #     num_workers=args.workers,
    #     drop_last=False,
    #     pin_memory= True, 
    #     worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x)
    #     )

    train_conf.scheduler_params['patience'] = 10
    train_conf.structure = "wae"
    train_conf.only_weights = False
    
    m = MODELS["WAE"](train_conf).cuda()
    trainer = BaseTrainer(m, train_conf)
    trainer.fit(train_loader, val_loader, None, use_wandb = False)

if __name__ == "__main__":
    main()