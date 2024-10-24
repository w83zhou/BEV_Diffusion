'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-10-24 15:22:33
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import os
import os.path as osp
import numpy as np
import pickle
import torch
import torch
from copy import deepcopy
from collections import OrderedDict


def append_prefix(state_dict, prefix=None):
    """Append the prefix in the front of the keys of the state_dict

    Args:
        state_dict (OrderedDict): the state dictionary
        prefix (str, optional): _description_. Defaults to None.

    Returns:
        OrderedDcit: the new state dictionary
    """
    if prefix is None:
        return deepcopy(state_dict)
    
    if not prefix.endswith('.'):
        prefix += '.'  # append '.' to the end of prefix if not exist
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = prefix + k
        new_state_dict[name] = v
    return new_state_dict


def main_load_bevdet_pretrained(
        bevdet_model_path, 
        prefix='bev_extractor',
        save_path=None):
    ## Load the pretrained model and save it with the prefix
    checkpoint = torch.load(bevdet_model_path, map_location="cpu")
    new_state_dict = append_prefix(checkpoint['state_dict'],
                                       prefix=prefix)
    
    # start saving
    checkpoint['state_dict'] = new_state_dict
    torch.save(checkpoint, save_path)


if __name__ == "__main__":
    main_load_bevdet_pretrained(bevdet_model_path='./BEVDet/bevdet-r50.pth', save_path='./ckpts/bevdet-r50-bev-extractor.pth')