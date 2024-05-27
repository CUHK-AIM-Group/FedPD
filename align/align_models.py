from pyexpat import model
import _init_paths
import ot
import torch
import numpy as np
from align.ground_metric import GroundMetric
import math
import sys
import os
import align.parameters
import sys 
import torch.nn as nn
from models import *
from models.digit import DigitModel
import time

def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c

def get_histogram(args, idx, cardinality, layer_name, activations=None, return_numpy = True, float64=False):
    if activations is None:
        # returns a uniform measure
        if not args.unbalanced:
            # print("returns a uniform measure of cardinality: ", cardinality)
            return np.ones(cardinality)/cardinality
        else:
            return np.ones(cardinality)

def get_wassersteinized_layers_modularized(args, networks, activations=None, eps=1e-7, test_loader=None):

    avg_aligned_layers = []
    T_last = None
    # cumulative_T_var = None
    T_var = None
    # print(list(networks[0].parameters()))
    previous_layer_shape = None
    ground_metric_object = GroundMetric(args)

    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
            enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):

        # print(layer0_name, fc_layer0_weight.size())
        # if 'cf2' not in layer0_name and 'fc3' not in layer0_name:

        assert fc_layer0_weight.shape == fc_layer1_weight.shape
        # print("Previous layer shape is ", previous_layer_shape)
        previous_layer_shape = fc_layer1_weight.shape
        print(fc_layer0_weight.shape)
        # print(fc_layer0_weight.shape)
        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        layer_shape = fc_layer0_weight.shape
        if 'clf2.bias' in layer0_name:
            avg_aligned_layers.append(fc_layer0_weight)
            continue
        if len(layer_shape) > 2:
            is_conv = True
            is_bn_bias = False
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
        elif len(layer_shape) == 1:
            is_bn_bias = True
            is_conv = False
            if 'bn5' in layer0_name:
                T_last = T_var
        else:
            is_conv = False
            is_bn_bias = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data

        if idx == 0:
            if is_conv:
                M = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                                fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
            else:
                M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)

            aligned_wt = fc_layer0_weight_data
        else:
            if is_conv:
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)

                M = ground_metric_object.process(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                )
            elif is_bn_bias:
                # print(fc_layer0_weight.data.shape, T_var.shape)

                aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)

                if args.ensemble_step != 0.5:
                    avg_aligned_layers.append((1 - args.ensemble_step) * aligned_wt +
                                        args.ensemble_step * fc_layer1_weight)
                else:
                    avg_aligned_layers.append((aligned_wt + fc_layer1_weight)/2)
                continue
            else:
                if 'clf2' in layer0_name:
                    T_var = T_last
                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    # Handles the switch from convolutional layers to fc layers
                    fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened,
                        T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                else:
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                M = ground_metric_object.process(aligned_wt, fc_layer1_weight)
                if 'clf2' in layer0_name:
                    if args.ensemble_step != 0.5:
                        avg_aligned_layers.append((1 - args.ensemble_step) * aligned_wt +
                                            args.ensemble_step * fc_layer1_weight)
                    else:
                        avg_aligned_layers.append((aligned_wt + fc_layer1_weight)/2)
                    continue
                # print("ground metric is ", M)
            if args.skip_last_layer and idx == (num_layers - 1):
                # print("Simple averaging of last layer weights. NO transport map needs to be computed")
                if args.ensemble_step != 0.5:
                    avg_aligned_layers.append((1 - args.ensemble_step) * aligned_wt +
                                        args.ensemble_step * fc_layer1_weight)
                else:
                    avg_aligned_layers.append((aligned_wt + fc_layer1_weight)/2)
                return avg_aligned_layers
            
        if not is_bn_bias:
            if args.importance is None or (idx == num_layers -1):
                mu = get_histogram(args, 0, mu_cardinality, layer0_name)
                nu = get_histogram(args, 1, nu_cardinality, layer1_name)

            cpuM = M.data.cpu().numpy()
            if args.exact:
                T = ot.emd(mu, nu, cpuM)
            else:
                T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)
            # T = ot.emd(mu, nu, log_cpuM)

            T_var = torch.from_numpy(T).float()

        if args.correction:
            if not args.proper_marginals:
                # think of it as m x 1, scaling weights for m linear combinations of points in X
                # T.t().shape[1] = T.shape[0]
                marginals = torch.ones(T_var.shape[0]) / T_var.shape[0]
                marginals = torch.diag(1.0/(marginals + eps))  # take inverse
                T_var = torch.matmul(T_var, marginals)
            else:
                marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype)

                marginals = (1 / (marginals_beta + eps))
                # print("shape of inverse marginals beta is ", marginals_beta.shape)
                # print("inverse marginals beta is ", marginals_beta)

                T_var = T_var * marginals
        # if is_bn_bias:
        #     T_bn_bias = T_var
        setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

        if args.past_correction:
            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))

        # Average the weights of aligned first layers
        if args.ensemble_step != 0.5:
            geometric_fc = ((1-args.ensemble_step) * t_fc0_model +
                            args.ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
        else:
            geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))/2
        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
            
        avg_aligned_layers.append(geometric_fc)
        # print(T_var.shape)
    avg_model = assign(networks[0], avg_aligned_layers)

    return avg_model

def assign(model, weights):
    idx = 0
    for name, param in model.named_parameters():
        # print(name)
        param.data = weights[idx].data
        # print(param, weights[idx])
        idx += 1
    return model
        
if __name__ == '__main__':
    args = parameters.get_parameters()
    start_time = time.time()


    model1 = DigitModel(num_classes=6)
    model1.clf2=nn.Linear(512, 1)
    model1.load_state_dict(torch.load('/data/cyang/Code/otfusion/fed_models/0.checkpoint'))
    model2 = DigitModel(num_classes=6)
    model2.clf2=nn.Linear(512, 1)
    model2.load_state_dict(torch.load('/data/cyang/Code/otfusion/fed_models/1.checkpoint'))

    aligned_layers = get_wassersteinized_layers_modularized(args, [model1, model2])
    # assign(aligned_layers)


    # for i in aligned_layers:
    #     print(i.shape)

    end_time = time.time()
    execution_time = end_time - start_time

    # 打印执行时间
    print("Running Time: ", execution_time, "S")





