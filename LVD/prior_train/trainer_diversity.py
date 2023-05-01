import os
from .trainer import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class DiversityTrainer(BaseTrainer):


    def fit(self, train_loader, valid_loader, test_loader = None, use_wandb = True):

        print("Optimization Starts !")
        print(f"Weights save path : {self.save_path}")
        
        print("Warmup Steps : ", self.warmup_steps)
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

                # print(train_loader.dataset.phase, train_loader.dataset.start, train_loader.dataset.end)
                # print(valid_loader.dataset.phase, valid_loader.dataset.start, valid_loader.dataset.end)

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
                skill_scheduler = self.schedulers['skill_enc_dec']
                skill_scheduler.step(valid_loss_dict[self.schedulers_metric['skill_enc_dec']])
                if "state" in self.schedulers.keys():
                    state_scheduler = self.schedulers['state']
                    state_scheduler.step(valid_loss_dict[self.schedulers_metric['state']])

                if e >= self.warmup_steps:
                    for module_name, scheduler in self.schedulers.items():
                        if module_name not in  ["skill_enc_dec", "state"] and self.schedulers_metric[module_name] is not None:
                            scheduler.step(valid_loss_dict[self.schedulers_metric[module_name]])

                    if self.loop_indicator(e, valid_loss_dict['metric']):
                        print("early stop", 'loss',  valid_loss_dict['metric'])
                        break

                if e > self.save_ckpt:
                    self.save(f'{self.model_path}/{self.model_id}_{e}.bin')

                # if e == self.warmup_steps + 10:
                # if e == 1:
                if e == self.mixin_start:
                    train_loader.set_mode("with_buffer")

                train_loader.update_buffer()
                # seed += 1
                # seed_everything(seed)

            self.save(f'{self.model_path}/{self.model_id}_end.bin')
            self.logger.log('Finished')           
         


    def train_one_epoch(self, loader, e):
        print(loader.dataset.mode)

        self.meter_initialize()


        for i, batch in enumerate(loader):
            self.model.train()
            loss = self.model.optimize(batch, e)

            # buffer에 subtrajectory 쌓기
            if 'states_novel' in loss.keys():
                loader.enqueue(loss['states_novel'], loss['actions_novel'])
        
            if not len(self.meters):
                # initialize key
                for key in loss.keys():
                    if key in ["rollout_KL", "rollout_KL_consistent", "KL_with_init", "rollout_r_int_consistent", "rollout_r_int",  "states_novel", "actions_novel"]:
                        rollout_KL, rollout_KL_consistent, rollout_r_int_consistent, rollout_r_int = [], [], [], []
                        KL_with_init = {}
                        rollout_KL_prev = None
                    else:
                        self.meters[key] = AverageMeter()
  
            for k, v in loss.items():
                if k == "rollout_KL":
                    # 이미지로 분포표 저장.
                    v = v.detach().cpu().numpy().tolist()
                    rollout_KL.extend(v)
                elif k == "rollout_KL_consistent":
                    # 이미지로 분포표 저장.
                    v = v.detach().cpu().numpy().tolist()
                    rollout_KL_consistent.extend(v)
                elif k == "rollout_r_int_consistent":
                    v = v.detach().cpu().numpy().tolist()
                    rollout_r_int_consistent.extend(v)
                elif k == "rollout_r_int":
                    v = v.detach().cpu().numpy().tolist()
                    rollout_r_int.extend(v)

                elif k not in ['states_novel', 'actions_novel']:
                    self.meters[k].update(v, batch['states'].shape[0])
                else:
                    pass

                # if k not in ['states_recon', 'actions_recon']:
                #     self.meters[k].update(v, batch['states'].shape[0])

                # self.meters[k].update(v, batch['states'].shape[0])
        
        if "rollout_KL" in loss.keys():
            plt.figure()
            plt.xlim(right = 40)
            rollout_KL = np.array(rollout_KL)
            rollout_KL_consistent = np.array(rollout_KL_consistent)

            sns.distplot(rollout_KL, color = "blue", bins=250)
            sns.distplot(rollout_KL_consistent, color = "orange", bins=250)

            q1 = np.quantile(rollout_KL, 0.25)
            q2 = np.quantile(rollout_KL, 0.5)
            q3 = np.quantile(rollout_KL, 0.75)
            plt.vlines(np.array([q1, q2, q3]), 0, 0.1, color = 'red', )
            
            q1 = np.quantile(rollout_KL_consistent, 0.25)
            q2 = np.quantile(rollout_KL_consistent, 0.5)
            q3 = np.quantile(rollout_KL_consistent, 0.75)
            plt.vlines(np.array([q1, q2, q3]), 0, 0.1, color = 'yellow', )

            plt.title(f"Q1 {q1:.2f}, Q2 {q2:.2f}, Q3 {q3:.2f}")
            plt.legend(["rollout", "consist", "rollout_Q", "consist_Q"])

            plt.savefig(f'imgs/gcid_stitch/rollout_KL/{e}.png')
            
            plt.figure()
            plt.xlim(right = 0.25)
            rollout_r_int_consistent = np.array(rollout_r_int_consistent)
            rollout_r_int = np.array(rollout_r_int)

            sns.distplot(rollout_r_int, color = "blue", bins=250)
            sns.distplot(rollout_r_int_consistent, color = "orange", bins=250)

            q1 = np.quantile(rollout_r_int, 0.25)
            q2 = np.quantile(rollout_r_int, 0.5)
            q3 = np.quantile(rollout_r_int, 0.75)
            plt.vlines(np.array([q1, q2, q3]), 0, 0.1, color = 'red', )

            q1 = np.quantile(rollout_r_int_consistent, 0.25)
            q2 = np.quantile(rollout_r_int_consistent, 0.5)
            q3 = np.quantile(rollout_r_int_consistent, 0.75)
            plt.vlines(np.array([q1, q2, q3]), 0, 0.1, color = 'yellow', )
            
            plt.title(f"Q1 {q1:.2f}, Q2 {q2:.2f}, Q3 {q3:.2f}")
            plt.legend(["rollout", "consist", "rollout_Q", "consist_Q"])
            
            plt.savefig(f'imgs/gcid_stitch/r_int/{e}.png')


        # set KL threshold for next epoch
        # KL_threshold = torch.quantile(torch.tensor(rollout_KL_consistent), 0.995).item()
        # KL_threshold = torch.max(torch.tensor(rollout_KL_single)).item()
        KL_threshold = 0
        self.model.KL_threshold = KL_threshold

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
                    if key in ["rollout_KL", "rollout_KL_main", "states_novel", "actions_novel"]:
                        rollout_KL, rollout_KL_main  = [], []
                        rollout_KL_prev = None
                    else:
                        self.meters[key] = AverageMeter()
  
            # for k, v in loss.items():
            #     if k not in ["rollout_KL", "rollout_KL_main"]:
            #         self.meters[k].update(v, batch['states'].shape[0])

            for k, v in loss.items():
                # if k == "rollout_KL":
                #     # 이미지로 분포표 저장.
                #     v = v.detach().cpu().numpy().tolist()
                #     rollout_KL.extend(v)
                # elif k == "rollout_KL_consistent":
                #     # 이미지로 분포표 저장.
                #     v = v.detach().cpu().numpy().tolist()
                #     rollout_KL_consistent.extend(v)
                # elif k == "rollout_r_int_consistent":
                #     v = v.detach().cpu().numpy().tolist()
                #     rollout_r_int_consistent.extend(v)
                # elif k == "rollout_r_int":
                #     v = v.detach().cpu().numpy().tolist()
                #     rollout_r_int.extend(v)

                if k not in ['states_novel', 'actions_novel']:
                    self.meters[k].update(v, batch['states'].shape[0])
                else:
                    pass


            # if self.warmup_steps != 0:
            #     self.set_lr(1/sqrt(max(self.iteration, self.warmup_steps)))
    
        return { k : v.avg for k, v in self.meters.items()}