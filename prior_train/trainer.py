import os
import time
from datetime import datetime
from glob import glob
import torch
import wandb
from tqdm import tqdm
from math import sqrt


from easydict import EasyDict as edict
from proposed.configs.data.kitchen import KitchenEnvConfig





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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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

        self.scheduler = config.schedulerClass(self.model.optimizer, **config.scheduler_params)

        self.early_stop = 0
        self.best_summary_loss = 10**5

        # path
        self.model_path = config.save_path #os.path.join(config.save_path[0], 'models')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # wandb error issue.. 
        # self.logger = Logger(config.log_path)

        if config.model_id == '':
            self.model_id  = f"{type(self.model).__name__}_{datetime.now().strftime('%m%d')}{str(int(datetime.now().strftime('%H%M')) + 9).zfill(2)}"
        
        else:
            self.model_id = config.model_id
        self.set_logger()


        self.meters = {}

    def meter_initialize(self):
        self.meters = {}


    def set_logger(self):
        

        os.makedirs("./logs", exist_ok = True)

        logs = os.listdir("./logs/")

        if not logs:
            logs = [-1, 0]
            log_path = "./logs/log0.txt"
        else:
            logs = sorted([ int(f.replace(".txt", "").replace("log", "")) for f in logs])
            log_path = f"./logs/log{max(logs) + 1}.txt"

        self.logger = Logger(log_path, verbose= False)

        config_text = "#" + "-" * 20 + "#"

        for k, v in self.config.items():
            config_text += f"{k} : {v}\n"

        config_text += "#" + "-" * 20 + "#"
        self.logger.log(config_text)
        print(log_path)

        self.model_id = f"log{max(logs) + 1}"



    def loop_indicator(self, e, val_loss):

        if self.best_summary_loss > val_loss:
            # record best 
            self.best_summary_loss = val_loss
            self.early_stop = 0
            self.best_e = e
            self.save(f'{self.model_path}/{self.model_id}_{e}.bin')
            # save
            if e > self.save_ckpt:
                self.save(f'{self.model_path}/{self.model_id}_{e}.bin')
                # for path in sorted(glob(f'{self.model_path}/{self.model_id}_*epoch.bin'))[:-1]:
                #     os.remove(path)
        else:
            if e > self.save_ckpt:
                self.save(f'{self.model_path}/{self.model_id}_{e}.bin')
            # early stop
            self.early_stop += 1

        
        if self.early_stop == self.early_stop_rounds:
            return True
        else: 
            return False

    def fit(self, train_loader, valid_loader, use_wandb = True):

        print(self.cond_decode)

        if self.warmup_method == "linear":
            self.set_lr(0.0005 * self.config.batchsize / 256)

        print("optimizing")
        print(f"weights save path : {self.save_path}")
        
        print(self.warmup_steps)
        if use_wandb:
            run = wandb.init(
                settings=wandb.Settings(start_method='thread'),
                # Set the project where this run will be logged
                project= self.project, #$#self.config.name, 
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
                message += f'Time : {time.time() - start:.5f} s \n'

                self.logger.log(message)


                if self.warmup_steps <= e:
                    if self.scheduler:
                        self.scheduler.step(valid_loss_dict['metric'])

                    if self.loop_indicator(e, valid_loss_dict['metric']):
                        break


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
                message += f'Time : {time.time() - start:.5f} s \n'

                self.logger.log(message)

                if self.warmup_steps <= e:
                    if self.scheduler:
                        self.scheduler.step(valid_loss_dict['metric'])

                    if self.loop_indicator(e, valid_loss_dict['metric']):
                        print("early stop", 'loss',  valid_loss_dict['metric'])
                        break
                
            self.save(f'{self.model_path}/{self.model_id}_end.bin')
            self.logger.log('Finished')           
         


    def train_one_epoch(self, loader, e):

        self.model.train()
        self.meter_initialize()


        for i, batch in enumerate(loader):
            loss = self.model.optimize(batch, e)
            if not len(self.meters):
                # initialize key
                for key in loss.keys():
                    self.meters[key] = AverageMeter()

            for k, v in loss.items():
                self.meters[k].update(v, batch['states'].shape[0])
            # if self.warmup_steps != 0:
            #     self.set_lr(1/sqrt(max(self.iteration, self.warmup_steps)))

    
        return { k : v.avg for k, v in self.meters.items() }


    def validate(self, loader, e):

        self.model.eval()
        self.meter_initialize()

        for i, batch in enumerate(loader):
            with torch.no_grad():
                loss = self.model.validate(batch, e)

            if not len(self.meters):
                # initialize key
                for key in loss.keys():
                    self.meters[key] = AverageMeter()

            for k, v in loss.items():
                self.meters[k].update(v, batch['states'].shape[0])
            # if self.warmup_steps != 0:
            #     self.set_lr(1/sqrt(max(self.iteration, self.warmup_steps)))
            
    
        return { k : v.avg for k, v in self.meters.items()}



    def save(self, path, only_weights = False):
        self.model.eval()

        if only_weights:
            torch.save({
            "only_weights" : True,
            'model': self.model.state_dict(),
            'optimizer' : self.model.optimizer.state_dict(),
            'scheduler' : self.scheduler.state_dict(),
            'scaler' : self.model.scaler.state_dict(),
            'configuration' : self.config,
            'best_summary_loss': self.best_summary_loss
            }, path)
        else:
            torch.save({
            "only_weights" : False,
            'model': self.model,
            'optimizer' : self.model.optimizer,
            'scheduler' : self.scheduler,
            'scaler' : self.model.scaler,
            'configuration' : self.config,
            'best_summary_loss': self.best_summary_loss
            }, path)


    def load(self, path):
        checkpoint = torch.load(path)
        if checkpoint['only_weights']:
            self.model.load_state_dict(checkpoint['model'])
            self.model.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.model.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.model = checkpoint['model']
            self.model.optimizer = checkpoint['optimizer']
            self.scheduler = checkpoint['scheduler']
            self.model.scaler = checkpoint['scaler']

        config = checkpoint['configuration']
        self.prep(config)
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.model.eval()

    def set_lr(self,lr = False, ratio = False):
        if ratio and not lr:
            self.lr *= ratio
            self.model.optimizer.param_groups[0]['lr'] = self.lr
        else:
            self.lr = lr
            self.model.optimizer.param_groups[0]['lr'] = self.lr

    



def get_loader(
    env_name,
    phase = "train",
    *args,
    **kwargs
    ):


    if env_name == "kitchen":

        env_config = KitchenEnvConfig(kwargs.get("mode", False))
    else:
        env_config = KitchenEnvConfig(kwargs.get("mode", False))


    env_config.attrs.subseq_len = 11

    conf = edict(

        # general 
        model_cls = None, 
        logger = None,
        data_dir = '.',
        epoch_cycles_train = 50,
        epochs = int(kwargs.get("epochs", False)),
        evaluator = None,
        top_of_n_eval = 100,
        top_comp_metric = "mse",

        #########################
        # BATCH SIZE #
        batch_size = 1024,
        ##########################



        exp_path = None,  # Path to the folder with experiments
        optim_cls = torch.optim.Adam,
        lr = 1e-3,
        gradient_clip = None, # hard gradient clipping factor
        init_grad_clip = 0.001, # to prevent gradient explode in early phase of learning
        init_grad_clip_step = 100,
        momentum = 0,
        adam_beta = 0.9,

        schedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params = dict(      # scheduler params
            mode='min',        
            factor=0.2,
            patience= 4,
            verbose=True, 
            threshold=0.01,
            threshold_mode='abs',
            cooldown=0, 
            min_lr=1e-12,
            eps=1e-08
        ),  

        save_path = "./weights",
        log_path = "./log.txt",
        save_ckpt = 100,
        warmup_steps = kwargs.get("warmup", False),
        warmup_method = None,
        model_id = "",
        early_stop_rounds = 10,


        # model
        state_dim=env_config.attrs.state_dim,
        action_dim=env_config.attrs.n_actions,
        n_rollout_steps=10,
        kl_div_weight=5e-4,
        hidden_dim=128,
        latent_dim=10,
        n_processing_layers=5,
        cond_decode=True,
        n_obj = 9,
        n_env = 21, 
        g_agent = kwargs.get("ga", False),
        

        # data
        data = edict(
            dataset_spec = env_config.attrs,
            subseq_len = 10 + 1, # conf.model.n_rollout_steps + 1
            device = "cpu"
        ),

        val_data_size = 5000 

    )


    if phase == "train":

        # train params
        params = edict(
            logger_class= None,
            model_class= conf.model_cls,
            n_repeat= conf.epoch_cycles_train,
            dataset_size=-1,
            goal_range = kwargs.get("goal_range", False),
            n_obj = conf.n_obj,
            n_env = conf.n_env
            
        )
    else:

        params = edict(
            logger_class= None,
            model_class= conf.model_cls,
            n_repeat=1,
            dataset_size=conf.val_data_size,
            goal_range = kwargs.get("goal_range", False),
            n_obj = conf.n_obj,
            n_env = conf.n_env

        )


    print(conf.data.dataset_spec.dataset_class)

    dataset = conf.data.dataset_spec.dataset_class(
        conf.data_dir,
        conf.data,
        resolution = 64,
        phase= phase,
        shuffle= phase == "train",
        **params
        # g_agent = params.g_agent,
        # n_obj = conf.n_obj,
        # n_env = conf.n_env,
        # goal_range = kwargs['goal_range'],
        # last = kwargs['params'].last,

    )

    return conf, dataset.get_data_loader(conf.batch_size, params.n_repeat, num_workers = 14)


