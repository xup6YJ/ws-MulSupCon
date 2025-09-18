from fastai.vision.all import *
from loss import get_criterion

SEED = 85
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

def find_lr(args):
    train_df = pd.read_csv(f'{args.train_df}.csv')
    valid_df = pd.read_csv(f'{args.valid_df}.csv')

    item_transforms = [
        Resize((args.resize, args.resize)),
    ]
    batch_transforms = [
        Flip(),
        Rotate(),
        Normalize.from_stats(*imagenet_stats),
    ]

    def get_x(row):
        return args.base_path + row['img_path']
    def get_y(row):
        labels = row[args.features].tolist()
        return labels

    dblock = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock(encoded=True,vocab=args.features)),
                    splitter=RandomSplitter(valid_pct=0.125, seed=SEED),
                    get_x=get_x,
                    get_y=get_y,
                    item_tfms=item_transforms,
                    batch_tfms=batch_transforms,
                    )
    # dls = dblock.dataloaders(train_val_df, bs=args.batch_size)
    train_dl = dblock.dataloaders(train_df, bs=args.batch_size, shuffle=True).train
    valid_dl = dblock.dataloaders(valid_df, bs=args.batch_size, shuffle=False).train

    dls = DataLoaders(train_dl, valid_dl)


    cbs=[
        SaveModelCallback(monitor='roc_auc_score', min_delta=0.0001, with_opt=True),
        EarlyStoppingCallback(monitor='roc_auc_score', min_delta=0.001, patience=5),
        ShowGraphCallback()
        ]


    loss_func = get_criterion(args)
    learn = vision_learner(dls, models.densenet121, metrics=[accuracy_multi, F1ScoreMulti(), RocAucMulti()], cbs=cbs, wd=0.001, loss_func=loss_func)
    # learn.model = torch.nn.DataParallel(learn.model)

    lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
    print(lrs.valley)
    return lrs.valley


def get_current_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    return (sum(lrs)/len(lrs))

