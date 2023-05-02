import os
import time
from datetime import datetime
from glob import glob
import torch
import wandb
from tqdm import tqdm
from math import sqrt
import numpy as np

from easydict import EasyDict as edict
from copy import deepcopy

from ..utils import *
# from ..configs.data.kitchen import KitchenEnvConfig

from ..configs.env import *
from ..configs.model import *


from matplotlib import pyplot as plt
import seaborn as sns




os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



class Logger:
    def __init__(self, log_path, verbose = True):
        self.log_path =log_path
        self.verbose = verbose
        

    def log(self, message):
        if self.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


    def loss_dict_log(self, loss_dict, set_name):

        message = set_name.upper() + " "
        for k, v in loss_dict.items():
            message += f"{k.replace(set_name.upper() + '_', '')} : {v:.5f} "

        return message





class BaseTrainer:

    def __init__(self, model, config):
        self.model = model
        self.prep(config)

    def prep(self, config):
        """
        Prepare for fitting
        """

        self.config = config # for save

        # configuration setting except scheduler
        for k, v in config.items():
            if 'schedule' not in k:
                setattr(self, k, v)
        
            
        
        print(f"--------------------------------{self.model.structure}--------------------------------")


        self.schedulers = {
            k : config.schedulerClass(v['optimizer'], **config.scheduler_params, module_name = k) for k, v in self.model.optimizers.items()
        }

        self.schedulers_metric = {
            k : v['metric'] for k, v in self.model.optimizers.items()
        }

        print(self.schedulers)
        print(self.schedulers_metric)


        self.early_stop = 0
        self.best_summary_loss = 10**5

        # path
        self.model_path = f"{config.save_path}/{self.env_name}/{self.model.structure}" #os.path.join(config.save_path[0], 'models')
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        if not os.path.exists(f"{config.save_path}/{self.env_name}"):
            os.makedirs(f"{config.save_path}/{self.env_name}")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)




        if config.model_id == '':
            self.model_id  = f"{type(self.model).__name__}_{datetime.now().strftime('%m%d')}{str(int(datetime.now().strftime('%H%M')) + 9).zfill(2)}"
        
        else:
            self.model_id = config.model_id
        self.set_logger()


        self.meters = {}

    def meter_initialize(self):
        self.meters = {}


    def set_logger(self):
        

        os.makedirs("logs", exist_ok = True)
        os.makedirs(f"logs/{self.env_name}", exist_ok = True)
        os.makedirs(f"logs/{self.env_name}/{self.model.structure}", exist_ok = True)

        logs = os.listdir(f"logs/{self.env_name}/{self.model.structure}")

        if not logs:
            logs = [-1, 0]
            log_path = f"logs/{self.env_name}/{self.model.structure}/log0.txt"
        else:
            logs = sorted([ int(f.replace(".txt", "").replace("log", "")) for f in logs])
            log_path = f"logs/{self.env_name}/{self.model.structure}/log{max(logs) + 1}.txt"

        self.logger = Logger(log_path, verbose= False)

        config_text = "#" + "-" * 20 + "#"

        for k, v in self.config.items():
            config_text += f"{k} : {v}\n"

        config_text += "#" + "-" * 20 + "#"
        self.logger.log(config_text)
        print(f"Log file : ", log_path)

        self.model_id = f"log{max(logs) + 1}"



    def loop_indicator(self, e, val_loss):

        if self.best_summary_loss > val_loss:
            # record best 
            self.best_summary_loss = val_loss
            self.early_stop = 0
            self.best_e = e
            # self.save(f'{self.model_path}/{self.model_id}_{e}.bin')
            # save
            if e > self.save_ckpt:
                self.save(f'{self.model_path}/{self.model_id}_{e}.bin')
                for path in sorted(glob(f'{self.model_path}/{self.model_id}_*epoch.bin'))[:-1]:
                    os.remove(path)
        else:
            # if e > self.save_ckpt:
            #     self.save(f'{self.model_path}/{self.model_id}_{e}.bin')
            # early self.
            self.early_stop += 1



        if self.early_stop == self.early_stop_rounds:
            return True
        else: 
            return False

    def fit(self, train_loader, valid_loader, test_loader = None, use_wandb = True):




        print("optimizing")
        print(f"weights save path : {self.save_path}")
        
        if use_wandb:
            run = wandb.init(
                # Set the project where this run will be logged
                project= self.model.structure, #$#self.config.name, 
                # Track hyperparameters and run metadata
                config=self.config)

            for e in tqdm(range(self.epochs)):

                start = time.time()

                train_loss_dict  = self.train_one_epoch(train_loader, e)
                valid_loss_dict = self.validate(valid_loader, e)

                message = f'[Epoch {e}]\n'
                message += self.logger.loss_dict_log(train_loss_dict, 'train')
                message += "\n"
                message += self.logger.loss_dict_log(valid_loss_dict, 'valid')
                message += "\n"
                if test_loader is not None:
                    test_loss_dict = self.validate(test_loader, e)
                    message += self.logger.loss_dict_log(test_loss_dict, 'test')
                    message += "\n"
                
                message += f'Time : {time.time() - start:.5f} s \n'

                self.logger.log(message)

                if self.warmup_steps <= e:
                    # if self.scheduler:
                    #     self.scheduler.step(valid_loss_dict['metric'])

                    # if self.schedulers:
                    #     for scheduler in self.schedulers:
                    #         scheduler.step(valid_loss_dict['metric'])

                    if self.schedulers:
                        for module_name, scheduler in self.schedulers.items():
                            scheduler.step(valid_loss_dict[self.schedulers_metric[module_name]])



                    if self.loop_indicator(e, valid_loss_dict['metric']):
                        print("early stop", 'loss',  valid_loss_dict['metric'])
                        break
                
                if e > self.save_ckpt:
                    self.save(f'{self.model_path}/{self.model_id}_{e}.bin')


                train_loss_dict = { "TRAIN " + k : v for k, v in train_loss_dict.items()}
                valid_loss_dict = { "VALID " + k : v for k, v in valid_loss_dict.items()}
                
                
                wandb.log(train_loss_dict, step = e)
                wandb.log(valid_loss_dict, step = e)


            self.save(f'{self.model_path}/{self.model_id}_end.bin')
            run.finish()


            
        else:


            for e in range(self.epochs):
                start = time.time()

                train_loss_dict  = self.train_one_epoch(train_loader, e)
                valid_loss_dict = self.validate(valid_loader, e)

                message = f'[Epoch {e}]\n'
                message += self.logger.loss_dict_log(train_loss_dict, 'train')
                message += "\n"
                message += self.logger.loss_dict_log(valid_loss_dict, 'valid')
                message += "\n"
                if test_loader is not None:
                    test_loss_dict = self.validate(test_loader, e)
                    message += self.logger.loss_dict_log(test_loss_dict, 'test')
                    message += "\n"
                message += f'Time : {time.time() - start:.5f} s \n'

                self.logger.log(message)
                
                # skill enc, dec를 제외한 모듈은 skill에 dependent함
                # 따라서 skill이 충분히 학습된 이후에 step down을 적용해야 함. 
                if "skill_enc_dec" in self.schedulers.keys():
                    skill_scheduler = self.schedulers['skill_enc_dec']
                    skill_scheduler.step(valid_loss_dict[self.schedulers_metric['skill_enc_dec']])

                if e >= self.warmup_steps:
                    # if self.scheduler:
                    #     self.scheduler.step(valid_loss_dict['metric'])

                    for module_name, scheduler in self.schedulers.items():
                        if module_name != "skill_enc_dec":
                            scheduler.step(valid_loss_dict[self.schedulers_metric[module_name]])

                    if self.loop_indicator(e, valid_loss_dict['metric']):
                        print("early stop", 'loss',  valid_loss_dict['metric'])
                        break

                if e > self.save_ckpt:
                    self.save(f'{self.model_path}/{self.model_id}_{e}.bin')

            self.save(f'{self.model_path}/{self.model_id}_end.bin')
            self.logger.log('Finished')           
         


    def train_one_epoch(self, loader, e):

        self.meter_initialize()


        for i, batch in enumerate(loader):
            print("loaded!!!!\n")
            self.model.train()
            loss = self.model.optimize(batch, e)
            
        

            if not len(self.meters):
                # initialize key
                for key in loss.keys():
                    if key in ["rollout_KL", "rollout_KL_main"]:
                        rollout_KL, rollout_KL_main  = [], []
                        rollout_KL_prev = None
                    else:
                        self.meters[key] = AverageMeter()
  


            for k, v in loss.items():
                if k == "rollout_KL":
                    # 이미지로 분포표 저장.
                    v = v.detach().cpu().numpy().tolist()
                    rollout_KL.extend(v)
                elif k == "rollout_KL_main":
                    # 이미지로 분포표 저장.
                    v = v.detach().cpu().numpy().tolist()
                    rollout_KL_main.extend(v)

                else:
                    self.meters[k].update(v, batch['states'].shape[0])

            # if self.warmup_steps != 0:
            #     self.set_lr(1/sqrt(max(self.iteration, self.warmup_steps)))
            del batch
        
        if self.model.structure == "vqvae":
            pass
            # self.model.quantizer.random_restart()
            # self.model.quantizer.reset_usage()
        #     # id counter에서 mean, variance
        #     counter = self.model.id_counter
        #     self.meters['id_count_mean'] =  AverageMeter()
        #     self.meters['id_count_std'] =  AverageMeter()
        #     count = self.meters['loss'].count
            
        #     print(count)

        #     self.meters['id_count_mean'].update(np.mean(list(counter.values())), count)
        #     self.meters['id_count_std'].update(np.std(list(counter.values())), count)
        #     self.model.clear_id_history()

        if "rollout_KL" in loss.keys():
            plt.figure()

            plt.xlim(right = 100)

            rollout_KL = np.array(rollout_KL)
            rollout_KL_main = np.array(rollout_KL_main)

            sns.distplot(rollout_KL, color = "blue")
            sns.distplot(rollout_KL_main, color = "orange")

            q1 = np.quantile(rollout_KL, 0.25)
            q2 = np.quantile(rollout_KL, 0.5)
            q3 = np.quantile(rollout_KL, 0.75)
            plt.vlines(np.array([q1, q2, q3]), 0, 0.1, color = 'red', )
            
            q1 = np.quantile(rollout_KL_main, 0.25)
            q2 = np.quantile(rollout_KL_main, 0.5)
            q3 = np.quantile(rollout_KL_main, 0.75)
            plt.vlines(np.array([q1, q2, q3]), 0, 0.1, color = 'yellow', )

            plt.title(f"Q1 {q1:.2f}, Q2 {q2:.2f}, Q3 {q3:.2f}")
            plt.legend(["rollout_iter_sample", "rollout", "iter_sample_Q", "rollout_Q"])

            plt.savefig(f'imgs/rollout_KL_{e}.png')
            rollout_KL_prev = deepcopy(rollout_KL)
            



        return { k : v.avg for k, v in self.meters.items() }


    def validate(self, loader, e):


        self.model.eval()
        self.meter_initialize()

        for i, batch in enumerate(loader):
            self.model.eval() # ?
            with torch.no_grad():
                loss = self.model.validate(batch, e)

            if not len(self.meters):
                # initialize key
                for key in loss.keys():
                    if key in ["rollout_KL", "rollout_KL_main"]:
                        rollout_KL, rollout_KL_main  = [], []
                        rollout_KL_prev = None
                    else:
                        self.meters[key] = AverageMeter()
  


            for k, v in loss.items():
                if k not in ["rollout_KL", "rollout_KL_main"]:
                    self.meters[k].update(v, batch['states'].shape[0])

            # if self.warmup_steps != 0:
            #     self.set_lr(1/sqrt(max(self.iteration, self.warmup_steps)))
        if self.model.structure == "vqvae":
            self.model.state_encoder.quantizer.random_restart()
            self.model.state_encoder.quantizer.reset_usage()
            
            pass



    
        return { k : v.avg for k, v in self.meters.items()}



    def save(self, path):
        self.model.eval()

        if self.only_weights:
            torch.save({
            "only_weights" : True,
            'model': self.model.state_dict(),
            'optimizer' : self.model.optimizer.state_dict(),
            # 'scheduler' : self.model.scheduler.state_dict(),
            # 'scaler' : self.model.scaler.state_dict(),
            'configuration' : self.config,
            'best_summary_loss': self.best_summary_loss
            }, path)
        else:


            torch.save({
            "only_weights" : False,
            'model': self.model,
            # 'optimizer' : self.model.optimizer,
            # 'optim' : self.model.optim,
            # 'subgoal_optim' : self.model.subgoal_optim,
            # 'scheduler' : self.scheduler,
            # 'scaler' : self.model.scaler,
            'configuration' : self.config,
            'best_summary_loss': self.best_summary_loss
            }, path)


    def load(self, path):
        checkpoint = torch.load(path)
        if checkpoint['only_weights']:
            self.model.load_state_dict(checkpoint['model'])
            self.model.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            # self.model.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.model = checkpoint['model']
            self.model.optimizer = checkpoint['optimizer']
            self.scheduler = checkpoint['scheduler']
            # self.model.scaler = checkpoint['scaler']

        config = checkpoint['configuration']
        self.prep(config)
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.model.eval()

    def set_lr(self, optimizer, lr = False, ratio = False):
        if ratio and not lr:
            self.lr *= ratio
            for param_group in self.model.optimizers[optimizer]['optimizer'].param_groups:
                param_group['lr'] = self.lr
        else:
            self.lr = lr
            for _, param_group in self.model.optimizers[optimizer]['optimizer'].param_groups:
                param_group['lr'] = self.lr

        print(f"{optimizer}'s lr : {self.lr}")

    



def get_loader(
    phase = "train",
    *args,
    **kwargs
    ):


    # if kwargs.get("env_name") == "kitchen":
    #     env_config = KitchenEnvConfig(kwargs.get("structure", False))
    # elif kwargs.get("env_name") == "calvin":
    #     env_config = CALVINEnvConfig(kwargs.get("structure", False))

    env_config = ENV_CONFIGS[kwargs.get("env_name", "kitchen")](kwargs.get("structure", "sc"))
    model_config = MODEL_CONFIGS[kwargs.get("structure", "sc")]



    env_default_conf = {**env_config.attrs}
    conf = edict(
        # general 
        optim_cls = torch.optim.Adam,
        lr = 1e-3,
        init_grad_clip = 5.0, # to prevent gradient explode in early phase of learning
        init_grad_clip_step = 0,
        schedulerClass = Scheduler_Helper,#torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params = dict(      # scheduler params
            mode='min',        
            factor=0.2,
            patience= 6,
            verbose=True, 
            threshold=1e-5,
            threshold_mode='abs',
            # cooldown=6, 
            min_lr=1e-12,
            # eps=1e-08
        ),  
        save_path = "./weights",
        log_path = "./log.txt",
        save_ckpt = 20,
        model_id = "",
        early_stop_rounds = 100,
        # model
        n_rollout_steps=10,
        cond_decode=True,        
        # data
        data = edict(
            dataset_spec = env_config.attrs,
            subseq_len = 10 + 1, # conf.model.n_rollout_steps + 1
            device = "cpu"
        ),

        val_data_size = 5000 
    )

    for k, v in env_default_conf.items():
        conf[k] = v

    for k, v in model_config.items():
        conf[k] = v 

    

    


    # conf = edict(
    #     # general 
    #     data_dir = '.',
    #     epoch_cycles_train = kwargs.get("epoch_cycles_train", 50),
    #     epochs = int(kwargs.get("epochs", False)),

    #     #########################
    #     # BATCH SIZE #
    #     batch_size = kwargs.get("batch_size", 512),
    #     ##########################

    #     optim_cls = torch.optim.Adam,
    #     lr = 1e-3,
    #     init_grad_clip = 5.0, # to prevent gradient explode in early phase of learning
    #     init_grad_clip_step = 0,

    #     schedulerClass = Scheduler_Helper,#torch.optim.lr_scheduler.ReduceLROnPlateau,
    #     scheduler_params = dict(      # scheduler params
    #         mode='min',        
    #         factor=0.2,
    #         patience= 6,
    #         verbose=True, 
    #         threshold=1e-5,
    #         threshold_mode='abs',
    #         # cooldown=6, 
    #         min_lr=1e-12,
    #         # eps=1e-08
    #     ),  

    #     save_path = "./weights",
    #     log_path = "./log.txt",
    #     save_ckpt = kwargs.get("save_ckpt", 20),
    #     warmup_steps = kwargs.get("warmup", False),
    #     warmup_method = None,
    #     model_id = "",
    #     early_stop_rounds = 100,

    #     # model
    #     state_dim=env_config.attrs.state_dim,
    #     action_dim=env_config.attrs.action_dim,
    #     n_rollout_steps=10,
    #     kl_div_weight=5e-4,
    #     hidden_dim=kwargs.get("hidden_dim", 128),
    #     latent_dim=kwargs.get("skill_dim", 10),
    #     n_processing_layers= kwargs.get('nLayers', 5),
    #     cond_decode=True,
    #     # n_obj = 9,
    #     # n_env = 21, 
    #     n_obj = env_config.attrs.n_obj,
    #     n_env = env_config.attrs.n_env, 
    #     n_goal = env_config.attrs.n_goal, 
        
    #     g_agent = kwargs.get("ga", False),
        
    #     # data
    #     data = edict(
    #         dataset_spec = env_config.attrs,
    #         subseq_len = 10 + 1, # conf.model.n_rollout_steps + 1
    #         device = "cpu"
    #     ),

    #     val_data_size = 5000 

    # )


    dataset_params = edict(
        logger_class= None,
        n_repeat= conf.epoch_cycles_train,
        dataset_size=-1,
        # n_obj = conf.n_obj,
        # n_env = conf.n_env,
        last = conf.last,
        mixin_ratio =  conf.mixin_ratio,
        rollout_method = conf.rollout_method,
        plan_H = conf.plan_H,
        only_proprioceptive = conf.only_proprioceptive
    )

    if phase == "train":

        # train params
        dataset = conf.data.dataset_spec.dataset_class(
            conf.data_dir,
            conf.data,
            resolution = 64,
            phase= phase,
            shuffle= phase == "train",
            **dataset_params
        )
    elif phase == "val":
        dataset_params.n_repeat = 1
        dataset_params.dataset_size = conf.val_data_size

        dataset = conf.data.dataset_spec.dataset_class(
            conf.data_dir,
            conf.data,
            resolution = 64,
            phase= "val",
            shuffle= False,
            **dataset_params
        )
    else:
        dataset_params.n_repeat = 1
        dataset_params.dataset_size = conf.val_data_size

        dataset = conf.data.dataset_spec.dataset_class(
            conf.data_dir,
            conf.data,
            resolution = 64,
            phase= "train",
            shuffle= False,
            **dataset_params
        )

    for k, v in kwargs.items():
        conf[k] = v

    return conf, dataset.get_data_loader(conf.batch_size, dataset_params.n_repeat, num_workers = conf.workers)
