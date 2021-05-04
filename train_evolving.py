import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
import os

import numpy as np
import time
import datetime
import path
import shutil

import config


args = config.parse()


# gpu, seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)


use_norm = 'use-norm' if args.use_norm else 'no-norm'
add_self_loop = 'add-self-loop' if args.add_self_loop else 'no-self-loop'


#### configure output directory

dataname = f'{args.data}_{args.dataset}'
model_name = args.model_name
nlayer = args.nlayer
dirname = f'{datetime.datetime.now()}'.replace(' ', '_').replace(':', '.')
out_dir = path.Path( f'./{args.out_dir}/{model_name}_{nlayer}_{dataname}/seed_{args.seed}' )


if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.makedirs_p()

 

### configure logger 
from logger import get_logger

baselogger = get_logger('base logger', f'{out_dir}/logging.log', not args.nostdout)
resultlogger = get_logger('result logger', f'{out_dir}/result.log', not args.nostdout)
baselogger.info(args)



# load data
from prepare import *


test_seens, test_unseens = [], []
best_test_seens, best_test_unseens = [], []

resultlogger.info(args)


def get_split(Y, p=0.2):
    from random import sample, shuffle
    Y = Y.tolist()
    N, nclass = len(Y),  len(set(Y))
    D = [[] for _ in range(nclass)]
    for i, y in enumerate(Y):
        D[y].append(i)
    k = int(N * p / nclass)
    train_idx = torch.cat([torch.LongTensor(sample(idxs, k)) for idxs in D]).tolist()
    test_idx = list(set(range(N)) - set(train_idx))
    shuffle(train_idx)
    shuffle(test_idx)
    seen_len = len(test_idx) // 2
    test_idx_seen, test_idx_unseen = test_idx[:seen_len], test_idx[seen_len:]

    return train_idx, test_idx_seen, test_idx_unseen

# load data
X, Y, G = fetch_data(args)



for run in range(1, args.n_runs+1):
    run_dir = out_dir / f'{run}'
    run_dir.makedirs_p()
    
    train_idx,  test_idx_seen, test_idx_unseen = get_split(Y, 0.2)

    from collections import Counter
    counter = Counter(Y[train_idx].tolist())
    print(counter)


    Xseen = X.clone()
    Xseen[test_idx_unseen] = 0

    # model 
    model, optimizer = initialise(Xseen, Y, G, args, test_idx_unseen)


    baselogger.info(f'Run {run}/{args.n_runs}, Total Epochs: {args.epochs}')
    baselogger.info(model)
    baselogger.info( f'total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}'  )

    tic_run = time.time()

    best_test_seen, best_test_unseen, test_seen, test_unseen,  Z = 0, 0, 0, 0,  None
    for epoch in range(args.epochs):
        # train
        tic_epoch = time.time()
        model.train()

        optimizer.zero_grad()
        Z = model(Xseen)
        loss = F.nll_loss(Z[train_idx], Y[train_idx])

        loss.backward()
        optimizer.step()

        train_time = time.time() - tic_epoch 
        
        
        # eval
        model.eval()
        Z = model(X)
        
        train_acc= accuracy(Z[train_idx], Y[train_idx])
        test_seen = accuracy(Z[test_idx_seen], Y[test_idx_seen])
        test_unseen = accuracy(Z[test_idx_unseen], Y[test_idx_unseen])

        best_test_seen = max(best_test_seen, test_seen)
        best_test_unseen = max(best_test_unseen, test_unseen)
        baselogger.info(f'epoch:{epoch} | loss:{loss:.4f} | train acc:{train_acc:.2f} | best_seen: {best_test_seen:.2f} | seen:{test_seen:.2f} | best_unseen: {best_test_unseen:.2f} | unseen:{test_unseen:.2f} | time:{train_time*1000:.1f}ms')

    resultlogger.info(f"Run {run}/{args.n_runs}, best_seen: {best_test_seen:.2f}, seen(last): {test_seen:.2f}, best_unseen: {best_test_unseen:.2f} , unseen:{test_unseen:.2f},  total time: {time.time()-tic_run:.2f}s")
    test_seens.append(test_seen)
    test_unseens.append(test_unseen)
    best_test_seens.append(best_test_seen)
    best_test_unseens.append(best_test_unseen)


resultlogger.info(f"Average final seen: {np.mean(test_seens)} ± {np.std(test_seens)}")
resultlogger.info(f"Average best seen: {np.mean(best_test_seens)} ± {np.std(best_test_seens)}")
resultlogger.info(f"Average final unseen: {np.mean(test_unseens)} ± {np.std(test_unseens)}")
resultlogger.info(f"Average best unseen: {np.mean(best_test_unseens)} ± {np.std(best_test_unseens)}")
