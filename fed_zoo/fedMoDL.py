import torch
import torch.distributed
import time
import os
import datetime
import random
import copy
import logging
import numpy as np
import argparse
import torch.nn as nn
from util.metric import save_csv
from models.MoDL import MoDL
from data import build_dataloader
from client import client_train, client_eval
from center_server import communication, validation
from torch.utils.tensorboard import SummaryWriter
from util.maths import softmax_loss
import matplotlib.pyplot as plt
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
CLIENTS = ['siat', 'cc359', 'fastMRI']


parser = argparse.ArgumentParser(description='ModFed')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--init-method', default='tcp://localhost:1836', help='initialization method')
parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--gpus', type=int, default=torch.cuda.device_count(), help='number of gpus per node')
parser.add_argument('--world-size', type=int, default=None, help='world_size = nodes * gpus')
parser.add_argument('--seed', type=int, default=24, help='random seed number')
parser.add_argument('--init_weight', type=bool, default=False)
parser.add_argument('--client_num', type=int, default=3)
parser.add_argument('--epochs', type=int, default=200)  # 200
parser.add_argument('--client_epoch', type=int, default=2)  # 2
parser.add_argument('--BATCH_SIZE', type=int, default=24)

parser.add_argument('--NUM_WORKERS', type=int, default=8)
parser.add_argument('--mask_path', type=str, default='../mask/1D_Random/mask_1DRandom_4x_acs24_256x256.mat')
parser.add_argument('--cc359_data_path', type=str, default='/media/sswang/2033127b-b5be-41bb-a893-3022d6f8b72a/data/CC359/Raw-data/Single-channel/')
parser.add_argument('--fastmri_data_path', type=str, default='/media/sswang/2033127b-b5be-41bb-a893-3022d6f8b72a/data/fastMRI/T1/')
parser.add_argument('--siat_data_path', type=str, default='/media/sswang/2033127b-b5be-41bb-a893-3022d6f8b72a/data/in_house_brain_data/')
parser.add_argument('--checkpoint', type=str, default='../checkpoint/')
parser.add_argument('--loss-curve-path', type=str, default='../runs/loss_curve/', help='save path of loss curve in tensorboard')

# args = parser.parse_args()


def create_logger():
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s:\t%(message)s')
    stream_formatter = logging.Formatter('%(levelname)s:\t%(message)s')

    file_handler = logging.FileHandler(filename='logger.txt', mode='a+', encoding='utf-8')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def init_weights(net, init_type='xavier', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialization method {} is not implemented.'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def create_models(args, device):
    client_num = args.client_num
    server_model = MoDL(2, 2, 5, device).to(device)
    clients = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    # distributed
    for model in clients:
        for key in server_model.state_dict().keys():
            model.state_dict()[key].data.copy_(server_model.state_dict()[key])

    r_models = []
    server_model = torch.nn.parallel.DistributedDataParallel(server_model)
    for model in clients:
        r_models.append(torch.nn.parallel.DistributedDataParallel(model))

    return server_model, r_models


def train(rank, nprocs, args):
    logger = create_logger()
    logger.info('New job assigned {}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H: %M')))
    logger.info('Running distributed data parallel on {} gpus.'.format(torch.cuda.device_count()))

    torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size,
                                        rank=rank)
    torch.cuda.set_device(rank)
    # create model
    server_model, clients = create_models(args, rank)

    train_sample_rate = {'siat': 1, 'cc359': 1, 'fastmri': 0.2}
    val_sample_rate = {'siat': 1, 'cc359': 1, 'fastmri': 0.2}

    start_epoch = 0
    server_model.to(rank)
    for client in clients:
        client.to(rank)
    criterion = nn.MSELoss().to(rank)

    # show params for server/clients
    n_parameters = sum(p.numel() for p in server_model.parameters() if p.requires_grad)

    logger.info('the parameter number of server_model is : {:.2f} M'.format(n_parameters/1e6))
    for idx, model in enumerate(clients):
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('the parameter number of : {:<8s} is : {:.2f} M'.format(CLIENTS[idx], n_parameters/1e6))

    # build optimizer
    optimizers = [torch.optim.AdamW(params=clients[idx].parameters(), lr=1e-4) for idx in range(args.client_num)]
    lr_schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[idx], mode='max', factor=0.3, patience=10)
                     for idx in range(args.client_num)]

    # load data  train, val, test
    sample_pattern = {
        'siat': '../mask/2D_Random/mask_2DRandom_4x_acs24_256x256.mat',
        'cc359': '../mask/1D_Random/mask_1DRandom_4x_acs24_256x256.mat',
        'fastmri': '../mask/1D_Equispaced/mask_1DEquispaced_4x_acs24_256x256.mat'
    }
    train_dataloaders, train_len = build_dataloader(args, train_sample_rate, sample_pattern, mode='train')
    valid_dataloaders, valid_len = build_dataloader(args, val_sample_rate, sample_pattern, mode='val')

    logger.info('training data : {}'.format(train_len))
    logger.info('validation data: {}'.format(valid_len))

    if args.init_weight:
        init_weights(server_model)
        for idx in range(args.client_num):
            init_weights(clients[idx])

    if args.pretrained:
        checkpoint = torch.load(args.checkpoint)
        server_model_checkpoint = checkpoint['server_model']

        logger.info('resume from %s' % args.checkpoint)
        server_model.load_state_dict(server_model_checkpoint, strict=True)
        for idx, client_name in enumerate(CLIENTS):
            model_checkpoint = checkpoint['model_{}'.format(client_name)]
            clients[idx].load_state_dict(model_checkpoint)
            optimizers[idx].load_state_dict(checkpoint['optimizer_{}'.format(client_name)])
            lr_schedulers[idx].load_state_dict(checkpoint['lr_scheduler_{}'.format(client_name)])
        start_epoch = checkpoint['epoch'] + 1

    start_time = time.time()

    best_status = [{'loss': 0., 'psnr': 0., 'ssim': 0.} for i in range(args.client_num)]
    best_checkpoint = [{} for i in range(args.client_num)]
    best_server = {'loss': 0., 'psnr': 0., 'ssim': 0.}
    server_checkpoint = {}
    server_dict = [
        {'loss_epoch': [], 'psnr_epoch': [], 'ssim_epoch': []} for i in range(args.client_num)]
    train_dict = [
        {'loss_epoch': [], 'psnr_epoch': [], 'ssim_epoch': []} for i in range(args.client_num)]
    val_dict = [
        {'loss_epoch': [], 'psnr_epoch': [], 'ssim_epoch': []} for i in range(args.client_num)]
    # start training
    writer = SummaryWriter(args.loss_curve_path)
    client_regular = [0. for _ in range(args.client_num)]
    # aggregate_weights = [data / sum(train_len) for _, data in enumerate(train_len)]
    aggregate_weights = [1 / len(train_len) for _, data in enumerate(train_len)]
    for epoch in range(start_epoch, args.epochs):

        logger.info('----------------------Global Epoch {:<3d}---------------'.format(epoch+1))
        for client_idx in range(args.client_num):
            loss, psnr, ssim = 0., 0., 0.
            for iter in range(args.client_epoch):
                # model, optimizer, dataloader, criterion
                train_status = client_train(clients[client_idx], optimizers[client_idx], train_dataloaders[client_idx],
                                                   criterion, rank, client_regular[client_idx])
                logger.info('{:<11s} | epoch: {:<3d}| train loss: {:.4f} | psnr: {:.4f} | ssim:{:.4f}'.format(
                CLIENTS[client_idx], iter+1, train_status['loss'], train_status['psnr'], train_status['ssim']))

                loss += train_status['loss']
                psnr += train_status['psnr']
                ssim += train_status['ssim']

            writer.add_scalar('/train_loss/{}'.format(CLIENTS[client_idx]), loss / args.client_epoch, epoch)
            writer.add_scalar('/train_psnr/{}'.format(CLIENTS[client_idx]), psnr / args.client_epoch, epoch)
            writer.add_scalar('/train_ssim/{}'.format(CLIENTS[client_idx]), ssim / args.client_epoch, epoch)

            train_dict[client_idx]['loss_epoch'].append(loss / args.client_epoch)
            train_dict[client_idx]['psnr_epoch'].append(psnr / args.client_epoch)
            train_dict[client_idx]['ssim_epoch'].append(ssim / args.client_epoch)

        # communication
        server_model, clients = communication(server_model, clients, aggregate_weights)

        # valid server
        status_server = validation(server_model, valid_dataloaders, criterion, rank)

        writer.add_scalar('/server_loss', np.mean(status_server['loss']), epoch)
        writer.add_scalar('/server_psnr', np.mean(status_server['psnr']), epoch)
        writer.add_scalar('/server_ssim', np.mean(status_server['ssim']), epoch)
        for client_idx in range(args.client_num):
            server_dict[client_idx]['loss_epoch'].append(status_server['loss'][client_idx])
            server_dict[client_idx]['psnr_epoch'].append(status_server['loss'][client_idx])
            server_dict[client_idx]['ssim_epoch'].append(status_server['loss'][client_idx])

        logger.info('server valid: | loss: {:.4f} | psnr: {:.4f} | ssim: {:.4f}'.format(
            np.mean(status_server['loss']), np.mean(status_server['psnr']), np.mean(status_server['ssim'])))

        if np.mean(status_server['psnr']) > best_server['psnr']:
            best_server['loss'] = np.mean(status_server['loss'])
            best_server['psnr'] = np.mean(status_server['psnr'])
            best_server['ssim'] = np.mean(status_server['ssim'])
            server_checkpoint = {
                'server_model': server_model.state_dict(),
                'loss': np.mean(status_server['loss']),
                'psnr': np.mean(status_server['psnr']),
                'ssim': np.mean(status_server['ssim']),
                'epoch': epoch + 1,
            }

        # updata regular parameter
        for i in range(args.client_num):
            client_regular[i] = status_server['loss'][i]

        # valid client
        for client_idx in range(args.client_num):
            model, valid_dataloader = clients[client_idx], valid_dataloaders[client_idx]
            eval_status = client_eval(model, valid_dataloader, criterion, rank)
            val_dict[client_idx]['loss_epoch'].append(eval_status['loss'])
            val_dict[client_idx]['psnr_epoch'].append(eval_status['psnr'])
            val_dict[client_idx]['ssim_epoch'].append(eval_status['ssim'])

            writer.add_scalar('/val_loss/{}'.format(CLIENTS[client_idx]), eval_status['loss'], epoch)
            writer.add_scalar('/val_psnr/{}'.format(CLIENTS[client_idx]), eval_status['psnr'], epoch)
            writer.add_scalar('/val_ssim/{}'.format(CLIENTS[client_idx]), eval_status['ssim'], epoch)

            # update aggregation weight
            aggregate_weights[client_idx] = eval_status['loss']

            logger.info('client valid: {:<11s} | epoch: {:<3d}| val loss: {:.4f} | psnr: {:.4f} | '
                        'ssim:{:.4f}'.format(CLIENTS[client_idx], epoch + 1, eval_status['loss'],
                                             eval_status['psnr'], eval_status['ssim']))

            if eval_status['psnr'] > best_status[client_idx]['psnr']:
                best_status[client_idx] = eval_status
                best_checkpoint[client_idx] = {
                    'model_{}'.format(CLIENTS[client_idx]): clients[client_idx].state_dict(),
                    'optimizer_{}'.format(CLIENTS[client_idx]): optimizers[client_idx].state_dict(),
                    'lr_scheduler_{}'.format(CLIENTS[client_idx]): lr_schedulers[client_idx].state_dict(),
                    'epoch': epoch + 1,
                    'args': args,
                    'loss': eval_status['loss'],
                    'psnr': eval_status['psnr'],
                    'ssim': eval_status['ssim'],
                    'lr': optimizers[client_idx].param_groups[0]['lr']
                }
            lr_schedulers[client_idx].step(eval_status['psnr'])
        aggregate_weights = softmax_loss(aggregate_weights)

        # save server model
        server_checkpoint_path = os.path.join(args.checkpoint, 'server_checkpoint_4x.pth')
        torch.save(server_checkpoint, server_checkpoint_path)

        # save client model
        for idx, client_name in enumerate(CLIENTS):
            checkpoint_path = os.path.join(args.checkpoint, 'client_{}_4x.pth'.format(client_name))
            torch.save(best_checkpoint[idx], checkpoint_path)

    # print server
    logger.info('the best epoch for server is {}'.format(server_checkpoint['epoch']))
    logger.info('loss: {:.4f}'.format(server_checkpoint['loss']))
    logger.info('psnr: {:.4f}'.format(server_checkpoint['psnr']))
    logger.info('ssim: {:.4f}'.format(server_checkpoint['ssim']))

    # print client
    for idx, client_name in enumerate(CLIENTS):
        logger.info('the best epoch for client {:<8s} is {}'.format(client_name, best_checkpoint[idx]['epoch']))
        logger.info('loss: {:.4f}'.format(best_checkpoint[idx]['loss']))
        logger.info('psnr: {:.4f}'.format(best_checkpoint[idx]['psnr']))
        logger.info('ssim: {:.4f}'.format(best_checkpoint[idx]['ssim']))
        logger.info('lr: {:.4f}'.format(best_checkpoint[idx]['lr']))

    total_time = time.time() - start_time
    logger.info('Training time: {} h \n'.format(total_time/3600.))

    # save metric result
    columns = ['loss_epoch', 'psnr_epoch', 'ssim_epoch']
    train_csv_url = '../fig/Client_train.csv'
    val_csv_url = '../fig/Client_val.csv'
    server_csv_url = '../fig/server_val.csv'
    save_csv(train_dict, columns, train_csv_url)
    save_csv(val_dict, columns, val_csv_url)
    save_csv(server_dict, columns, server_csv_url)

    # plot
    plt.ion()
    x = range(0, args.epochs)
    plt.subplot(1, 3, 1)
    plt.plot(x, val_dict[0]['psnr_epoch'])
    plt.xlabel('epoch')
    plt.ylabel('psnr')
    plt.title('siat')
    plt.subplot(1, 3, 2)
    plt.plot(x, val_dict[1]['psnr_epoch'])
    plt.xlabel('epoch')
    plt.ylabel('psnr')
    plt.title('cc359')
    plt.subplot(1, 3, 3)
    plt.plot(x, val_dict[2]['psnr_epoch'])
    plt.xlabel('epoch')
    plt.ylabel('psnr')
    plt.title('fastmri')
    plt.show()
    name = "../fig/" + str(time.asctime(time.localtime(time.time()))) + '_4x.jpg'
    plt.savefig(name)


def main():
    args = parser.parse_args()
    args.world_size = args.nodes * args.gpus  # 1*2
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.multiprocessing.spawn(train, nprocs=args.gpus, args=(args.gpus, args))


if __name__ == '__main__':
    main()
