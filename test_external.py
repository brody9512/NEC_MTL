import os
import shutil
import torch
import pandas as pd

import config
import utils

args = config.get_args_test()
path_=args.path
layers=args.layers
gpu=args.gpu
optim=args.optim
EPOCHS = args.epoch
ver = args.ver
st = args.st
de = args.de
clipLimit_=args.clahe
train_batch=args.batch
min_side_=args.size
lr_=args.lr_
lr__=args.lr__
lr_p=args.lr___
seg_weight_= args.seg_weight
feature=args.feature
infer=args.infer
external=args.external
weight_=args.weight
cbam_ = args.cbam
thr = args.thr
half= args.half

change_epoch = [0, 15, 28, 40, 55, 68, 75]
ratio = [[5, 5], [5, 5],[5, 5],[1,9] ,[1, 9], [1,9], [1, 9]]

if len(ratio[0]) == 3:
    consist_ = True
else:
    consist_ = False

if external:
    name=f'{weight_}_ex_{feature}'
else:
    name=f'{weight_}_'
    
save_dir = f"/workspace/203changhyun/nec_ch/v1_pneumoperiT_code/result/{name}"

if os.path.exists(save_dir): 
    shutil.rmtree(save_dir) 
os.mkdir(save_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = gpu#args.gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

utils.my_seed_everywhere(args.seed)

df = pd.read_csv(args.path)

# Modify only the paths that start with '/home/brody9512/workspace/'
df['img_dcm'] = df['img_dcm'].apply(lambda x: x.replace('/home/brody9512', '') if x.startswith('/home/brody9512') else x)



if external:
    test_df=df
else:
    train_df = df[df['Mode_1'] == 'train']
    val_df = df[df['Mode_1'] == 'validation']
    test_df = df[df['Mode_1'] == 'test']

if half:
    # Sample half of each DataFrame (assuming each has more than 1 row)
    train_df = train_df.sample(n=len(train_df_full) // 2, random_state=42)
    val_df = val_df.sample(n=len(val_df_full) // 2, random_state=42)
    test_df = test_df.sample(n=len(test_df_full) // 2, random_state=42)