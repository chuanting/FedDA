# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: fed_att_algo.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-01-20 (YYYY-MM-DD)
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
import os
import sys
from sklearn import metrics

sys.path.append('../')
from DualFedAtt.utils.misc import args_parser, average_weights_att
from DualFedAtt.utils.misc import get_data, process_isolated
from DualFedAtt.utils.models import LSTM
from DualFedAtt.utils.fed_update import LocalUpdate, test_inference

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = args_parser()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir(args.directory):
        os.mkdir(args.directory)

    data, df_ori, selected_cells, mean, std, lng, lat = get_data(args)
    device = 'cuda' if args.gpu else 'cpu'
    # print(selected_cells)

    parameter_list = 'FedAtt-Data-{:}-type-{:}-'.format(args.file, args.type)
    parameter_list += 'rho-{:.3f}-cluster-{:}-lr-{:.4f}-'.format(args.rho, args.cluster, args.lr)
    parameter_list += '-frac-{:.2f}-le-{:}-lb-{:}-epsilon-{:.2f}-seed-{:}'.format(args.frac, args.local_epoch,
                                                                                  args.local_bs, args.epsilon,
                                                                                  args.seed)
    log_id = args.directory + parameter_list
    # print(args)

    train, val, test = process_isolated(args, data)

    global_model = LSTM(args).to(device)
    global_weights = global_model.state_dict()
    # training
    best_val_loss = None
    val_loss = []
    val_acc = []
    cell_loss = []
    loss_hist = []
    for epoch in tqdm.tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        # print(f'\n Global Training Round: {epoch+1}|\n')

        m = max(int(args.frac * args.bs), 1)
        cell_idx = random.sample(selected_cells, m)
        # print(cell_idx)

        for cell in cell_idx:
            cell_train, cell_test = train[cell], test[cell]
            local_model = LocalUpdate(args, cell_train, cell_test)
            w, loss, epoch_loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                             global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            cell_loss.append(loss)

        loss_hist.append(sum(cell_loss)/len(cell_loss))

        # Update global model
        global_weights = average_weights_att(local_weights, global_weights, args.epsilon)
        global_model.load_state_dict(global_weights)

    # test
    pred, truth = defaultdict(), defaultdict()
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0
    global_model.load_state_dict(global_weights)
    with torch.no_grad():
        for cell in selected_cells:
            cell_test = test[cell]

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
        'FedAtt File: {:} Type: {:} epsilon: {:.2f} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type,
                                                                                                   args.epsilon,
                                                                                                   mse, mae, nrmse))
