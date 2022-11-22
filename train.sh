
# # spirl
# python prior_train/train_prior.py

# didspirl 
# python prior_train/train_did.py --min 0 --max -1
# python prior_train/train_gcid.py 


python prior_train/train_vic.py --epochs 35


# gcspirl
# goal range에 대한 효과. recon이 잘 안되는 것 확인해야 함. 
# python train_gc_spirl.py --gc True --ga True --min 0 --max -1 --epochs 20 --reg_beta 0.0001
# python train_gc_spirl.py --gc --min 0 --max -1 --epochs 50 --last
# python train_gc_spirl.py --gc --min 100 --max -1 --epochs 50 --reg_beta 0.00001

# python train_gc_spirl.py --gc --min 0 --max 20 --epochs 50 --reg_beta 0.00001
# python train_caes.py --gc --min 0 --max 20 --epochs 50 --reg_beta 0.0005 --warmup 10

# python train_sgspirl.py --mode sg --min 0 --max -1 --epochs 50 --reg_beta 0.0005 --warmup 10