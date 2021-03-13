# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: fed_dual_att.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-07-25 (YYYY-MM-DD)
-----------------------------------------------
"""
import numpy as np
import h5py
import tqdm
import copy
import torch
import pandas as pd
import random
from collections import defaultdict
from torch.utils.data import DataLoader
import os
import sys
from sklearn import metrics
from scipy.spatial.distance import pdist

sys.path.append('../')
from DualFedAtt.utils.misc import args_parser, avg_dual_att
from DualFedAtt.utils.misc import get_data, process_isolated, get_warm_up_data
from DualFedAtt.utils.misc import get_cluster_label, jfi, cv
from DualFedAtt.utils.models import LSTM
from DualFedAtt.utils.fed_update import LocalUpdate, test_inference

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_stat_mean(d):
    d = d.iloc[:-24 * args.test_days, :]
    df_avg = d.groupby([d.index.week, d.index.hour]).mean().reset_index().iloc[:, 2:]
    df_avg = (df_avg - df_avg.mean()) / df_avg.std()
    return copy.deepcopy(df_avg.T)


def get_warm_model(g_index, save_name):
    # warm-up model using the statistical mean model
    warm_xc, warm_xp, warm_y = [], [], []
    # print(g_index)

    model = LSTM(args).to(device)
    for i in g_index:
        cell_xc, cell_xp, cell_y = get_warm_up_data(args, df.loc[i][:-1])
        if args.phi > 0:
            n_transfer = int(np.floor(len(cell_xc) * args.phi))
            idx = [a for a in np.random.randint(0, len(cell_xc), n_transfer)]
            warm_xc.append(cell_xc[idx])
            warm_xp.append(cell_xp[idx])
            warm_y.append(cell_y[idx])
        else:
            warm_xc.append(cell_xc)
            warm_xp.append(cell_xp)
            warm_y.append(cell_y)
    warm_xc_arr = np.concatenate(warm_xc, axis=0)[:, :, np.newaxis]
    if args.period_size > 0:
        warm_xp_arr = np.concatenate(warm_xp, axis=0)[:, :, np.newaxis]
    else:
        warm_xp_arr = warm_xc_arr
    warm_y_arr = np.concatenate(warm_y, axis=0)

    warm_data = list(zip(*[warm_xc_arr, warm_xp_arr, warm_y_arr]))
    warm_loader = DataLoader(warm_data, shuffle=False, batch_size=args.batch_size)
    warm_criterion = torch.nn.MSELoss().to(device)
    if args.opt == 'adam':
        warm_opt = torch.optim.Adam(model.parameters(), lr=args.w_lr)
    elif args.opt == 'sgd':
        warm_opt = torch.optim.SGD(model.parameters(), lr=args.w_lr, momentum=args.momentum)

    warm_scheduler = torch.optim.lr_scheduler.MultiStepLR(warm_opt, milestones=[0.5 * args.w_epoch,
                                                                                0.75 * args.w_epoch],
                                                          gamma=0.1)

    for epoch in range(args.w_epoch):
        warm_epoch_loss = []
        model.train()
        for batch_idx, (xc, xp, y) in enumerate(warm_loader):
            xc, xp, y = xc.float().to(device), xp.float().to(device), y.float().to(device)
            model.zero_grad()
            pred = model(xc, xp)
            loss = warm_criterion(y, pred)
            warm_epoch_loss.append(loss)
            loss.backward()
            warm_opt.step()

        warm_scheduler.step()

    return model.state_dict()


if __name__ == '__main__':
    args = args_parser()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir(args.directory):
        os.mkdir(args.directory)

    data, df_ori, selected_cells, mean, std, lng, lat = get_data(args)
    # print(selected_cells)
    device = 'cuda' if args.gpu else 'cpu'

    parameter_list = 'FedDualAtt-data-{:}-type-{:}-'.format(args.file, args.type)
    parameter_list += 'rho-{:.3f}-cluster-{:}-lr-{:.4f}-'.format(args.rho, args.cluster, args.lr)
    parameter_list += '-frac-{:.2f}-le-{:}-lb-{:}-epsilon-{:.2f}-seed-{:}-'.format(args.frac, args.local_epoch,
                                                                                   args.local_bs, args.epsilon,
                                                                                   args.seed)
    parameter_list += 'warm_up:{:}'.format(args.warm_up)
    log_id = args.directory + parameter_list
    # print(args)

    train, val, test = process_isolated(args, data)
    # get the statistical mean of the traffic data
    df = get_stat_mean(data)
    # print(df_mean.head())
    data_dist = pdist(df.values)
    data_jfi = jfi(np.array(data_dist))
    data_cv = cv(np.array(data_dist))
    # print('jfi: {:.4f}, cv: {:.4f}'.format(data_jfi, data_cv))

    # dual-stage iterative clustering
    df['label'] = get_cluster_label(args, df, lng, lat)

    global_model = LSTM(args).to(device)
    # use this warm-up model as initialization
    cluster_weights = defaultdict()

    if args.warm_up:
        warm_weights = copy.deepcopy(get_warm_model(selected_cells, log_id))
        global_weights = copy.deepcopy(warm_weights)
        global_model.load_state_dict(global_weights)
        for label in df['label'].unique():
            cluster_weights[label] = copy.deepcopy(warm_weights)
    else:
        warm_weights = copy.deepcopy(global_model.state_dict())
        for label in df['label'].unique():
            cluster_weights[label] = copy.deepcopy(warm_weights)

    # training
    best_val_loss = None
    val_loss = []
    val_acc = []

    for epoch in tqdm.tqdm(range(args.epochs)):
        local_weights, local_losses = defaultdict(list), defaultdict(list)
        # print(f'\n Global Training Round: {epoch+1}|\n')

        m = max(int(args.frac * args.bs), 1)
        cell_idx = random.sample(selected_cells, m)

        avg_loss = 0
        for cell in cell_idx:
            group_id = df.loc[cell]['label']
            # print('Group ID:', group_id)
            global_model.load_state_dict(global_weights)

            cell_train, cell_test = train[cell], test[cell]
            local_model = LocalUpdate(args, cell_train, cell_test)
            w, loss, epoch_loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                             global_round=epoch)
            avg_loss += loss

            local_weights[group_id].append(copy.deepcopy(w))
            local_losses[group_id].append(copy.deepcopy(loss))

        # Update global model
        local_cluster = defaultdict()
        for group_id in local_weights.keys():
            local_cluster[group_id] = avg_dual_att(local_weights[group_id], cluster_weights[group_id],
                                                      warm_weights,
                                                      args.epsilon, args.rho)
        cw = []
        for c_key, c_weights in local_cluster.items():
            cw.append(c_weights)
        global_weights = avg_dual_att(cw, global_weights, warm_weights, args.epsilon, args.rho)
        # global_weights = average_weights(cw)
        global_model.load_state_dict(global_weights)

    pred, truth = defaultdict(), defaultdict()
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    global_model.load_state_dict(global_weights)

    with torch.no_grad():
        for cell in selected_cells:
            cell_test = test[cell]
            group_id = int(df.loc[cell]['label'])

            test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, global_model, cell_test)
            # print(f'Cell: {cell} MSE: {test_mse:.4f}')
            nrmse += test_nrmse

            test_loss_list.append(test_loss)
            test_mse_list.append(test_mse)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    print(
        'FedDualAtt File: {:}, Type: {:}, BS: {:}, frac: {:.2f}, Cluster: {:}, rho: '
        '{:.2f}, epsilon: {:.2f}, seed: {:}, lb: {:}, le: {:}, close: {:}, period: {:}, hidden: {:}, layers: {:},'
        ' lr: {:.4f}, w_lr: {:.4f}, '
        'MSE: {:.4f}, MAE: {:.4f}, NRMSE: {:.4f}'.format(
            args.file, args.type, args.bs, args.frac,
            args.cluster, args.rho, args.epsilon,
            args.seed, args.local_bs, args.local_epoch, args.close_size, args.period_size,
            args.hidden_dim, args.phi, args.lr, args.w_lr,
            mse, mae, nrmse))
