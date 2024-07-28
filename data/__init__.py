import os
import torch
from data.siat import SIATData
from data.cc359 import CC359Data
from data.fastMRI import FastMRIData
from torch.utils.data import DistributedSampler, DataLoader


def build_dataloader(args, sample_rate, sample_pattern, mode='train'):
    siat_dataset = SIATData(args, sample_rate['siat'], sample_pattern['siat'], mode=mode)
    cc359_dataset = CC359Data(args, sample_rate['cc359'], sample_pattern['cc359'], mode=mode)
    fastmri_dataset = FastMRIData(args, sample_rate['fastmri'], sample_pattern['fastmri'], mode=mode)

    datasets = [siat_dataset, cc359_dataset, fastmri_dataset]

    data_loader = []
    dataset_len = []
    for dataset in datasets:
        dataset_len.append(len(dataset))
        if mode == 'train':
            sampler = DistributedSampler(dataset)
            data_loader.append(DataLoader(dataset, batch_size=args.BATCH_SIZE, sampler=sampler,
                                          num_workers=args.NUM_WORKERS, pin_memory=True))
        elif mode == 'val':
            sampler = DistributedSampler(dataset, shuffle=False)
            data_loader.append(DataLoader(dataset, batch_size=args.BATCH_SIZE, sampler=sampler,
                                          num_workers=args.NUM_WORKERS, pin_memory=True))

    return data_loader, dataset_len
