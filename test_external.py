from config import get_args_test

args = get_args_test()
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
seed_=args.seed