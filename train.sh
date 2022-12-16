
# # spirl
# python prior_train/train_prior.py

# baseline 
# python prior_train/train_gcid.py --epochs 70 --reg_beta 0.0005 --last --direct




# 현재 최적
# python prior_train/train_gcid.py --epochs 50 --reg_beta 0.0005 --last --direct --wde 2.5e-5 --wdd 2.5e-5 --wdp 1e-6 --norm bn
# python prior_train/train_gcid.py --epochs 50 --last --direct --wde 2.5e-5 --wdd 2.5e-5 --wdp 1e-6
python prior_train/train_gcid.py --epochs 50 --last --direct --wde 2.5e-5 --wdd 2.5e-5 --wdp 1e-6




