from model import *
import torch, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F


def accuracy(Z, Y):
    
    return 100 * Z.argmax(1).eq(Y).float().mean().item()


import torch_sparse

def fetch_data(args):
    from data import data 
    dataset, _, _ = data.load(args)
    args.dataset_dict = dataset 

    X, Y, G = dataset['features'], dataset['labels'], dataset['hypergraph']
   
    # node features in sparse representation
    X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)
    X = torch.FloatTensor(np.array(X.todense()))
    
    # labels
    Y = np.array(Y)
    Y = torch.LongTensor(np.where(Y)[1])

    X, Y = X.cuda(), Y.cuda()
    return X, Y, G

def initialise(X, Y, G, args, unseen=None):
    """
    initialises model, optimiser, normalises graph, and features
    
    arguments:
    X, Y, G: the entire dataset (with graph, features, labels)
    args: arguments
    unseen: if not None, remove these nodes from hypergraphs

    returns:
    a tuple with model details (UniGNN, optimiser)    
    """
    
    G = G.copy()
    
    if unseen is not None:
        unseen = set(unseen)
        # remove unseen nodes
        for e, vs in G.items():
            G[e] =  list(set(vs) - unseen)

    if args.add_self_loop:
        Vs = set(range(X.shape[0]))

        # only add self-loop to those are orginally un-self-looped
        # TODO:maybe we should remove some repeated self-loops?
        for edge, nodes in G.items():
            if len(nodes) == 1 and nodes[0] in Vs:
                Vs.remove(nodes[0])

        for v in Vs:
            G[f'self-loop-{v}'] = [v]



    N, M = X.shape[0], len(G)
    indptr, indices, data = [0], [], []
    for e, vs in G.items():
        indices += vs 
        data += [1] * len(vs)
        indptr.append(len(indices))
    H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=int).tocsr() # V x E
    degV = torch.from_numpy(H.sum(1)).view(-1, 1).float()
    degE2 = torch.from_numpy(H.sum(0)).view(-1, 1).float()

    (row, col), value = torch_sparse.from_scipy(H)
    V, E = row, col
    from torch_scatter import scatter
    assert args.first_aggregate in ('mean', 'sum'), 'use `mean` or `sum` for first-stage aggregation'
    degE = scatter(degV[V], E, dim=0, reduce=args.first_aggregate)
    degE = degE.pow(-0.5)
    degV = degV.pow(-0.5)
    degV[degV.isinf()] = 1 # when not added self-loop, some nodes might not be connected with any edge


    V, E = V.cuda(), E.cuda()
    args.degV = degV.cuda()
    args.degE = degE.cuda()
    args.degE2 = degE2.pow(-1.).cuda()


    


    nfeat, nclass = X.shape[1], len(Y.unique())
    nlayer = args.nlayer
    nhid = args.nhid
    nhead = args.nhead

    # UniGNN and optimiser
    if args.model_name == 'UniGCNII':
        model = UniGCNII(args, nfeat, nhid, nclass, nlayer, nhead, V, E)
        optimiser = torch.optim.Adam([
            dict(params=model.reg_params, weight_decay=0.01),
            dict(params=model.non_reg_params, weight_decay=5e-4)
        ], lr=0.01)
    elif args.model_name == 'HyperGCN':
        args.fast = True
        dataset = args.dataset_dict
        model = HyperGCN(args, nfeat, nhid, nclass, nlayer, dataset['n'], dataset['hypergraph'], dataset['features'])
        optimiser = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    else:
        model = UniGNN(args, nfeat, nhid, nclass, nlayer, nhead, V, E)
        optimiser = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


    model.cuda()
   
    return model, optimiser



def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M  
    where D is the diagonal node-degree matrix 
    """
    
    d = np.array(M.sum(1))
    
    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)    # D inverse i.e. D^{-1}
    
    return DI.dot(M)
