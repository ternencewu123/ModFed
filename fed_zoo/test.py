# System / Python
import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# PyTorch
import torch
from torch.utils.data.dataloader import DataLoader
# Custom
from data.siat import SIATData
from data.cc359 import CC359Data
from data.fastMRI import FastMRIData
from models.MoDL import MoDL
from util.metric import remove_module

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, default='ModFed', help='name of experiment')
# parameters related to model
parser.add_argument('--num-layers', type=int, default=5, help='number of iterations')
parser.add_argument('--in-channels', type=int, default=2, help='number of model input channels')
parser.add_argument('--out-channels', type=int, default=2, help='number of model output channels')
# batch size, num workers
parser.add_argument('--batch-size', type=int, default=24, help='batch size of single gpu')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
# parameters related to test data
parser.add_argument('--mask_path', type=str, default='../mask/1D_Random/mask_1DRandom_4x_acs24_256x256.mat')
parser.add_argument('--cc359_data_path', type=str, default='/media/sswang/2033127b-b5be-41bb-a893-3022d6f8b72a/data/CC359/Raw-data/Single-channel/')
parser.add_argument('--fastmri_data_path', type=str, default='/media/sswang/2033127b-b5be-41bb-a893-3022d6f8b72a/data/fastMRI/T1/')
parser.add_argument('--siat_data_path', type=str, default='/media/sswang/2033127b-b5be-41bb-a893-3022d6f8b72a/data/in_house_brain_data/')

parser.add_argument('--test_sample_rate', '-tesr', type=float, default=1.0, help='sampling rate of test data')
# others
parser.add_argument('--model_save_path', type=str, default='../checkpoint/', help='save path of trained model')
parser.add_argument('--results_save_path', type=str, default='../result/',
                    help='save path of reconstruction results')
parser.add_argument('--save_results', '-sr', type=bool, default=False, help='whether save results')


def save_results(name, under_slice, output_slice, label_slice, psnr_zerof, ssim_zerof, psnr, ssim):
    import matplotlib.pyplot as plt
    from matplotlib import colors

    diff_img = label_slice - output_slice
    norm = colors.Normalize(vmin=0.0, vmax=1.0)

    plt.ion()
    # plt.figure(10)
    plt.subplot(221)
    plt.imshow(label_slice, cmap='gray')
    plt.axis('off')
    plt.title('full_img')
    plt.subplot(222)
    plt.imshow(under_slice, cmap='gray')
    plt.axis('off')
    plt.title('under_img, psnr:{:.5f}, ssim:{:.5f}'.format(psnr_zerof, ssim_zerof))
    plt.subplot(223)
    plt.imshow(output_slice, cmap='gray')
    plt.axis('off')
    plt.title('infer_img, psnr:{:.5f}, ssim:{:.5f}'.format(psnr, ssim))
    plt.subplot(224)
    plt.imshow(diff_img, norm=norm, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.title('diff_img')
    plt.savefig(name)
    # plt.pause(0.1)
    plt.close()


def validate(args):
    torch.cuda.set_device(0)
    sample_pattern = {
        'siat': '../mask/2D_Random/mask_2DRandom_4x_acs24_256x256.mat',
        'cc359': '../mask/1D_Random/mask_1DRandom_4x_acs24_256x256.mat',
        'fastmri': '../mask/1D_Equispaced/mask_1DEquispaced_4x_acs24_256x256.mat',
    }
    test_set = SIATData(args, args.test_sample_rate, sample_pattern['siat'], mode='test')
    # test_set = CC359Data(args, args.test_sample_rate, sample_pattern['cc359'], mode='test')
    # test_set = FastMRIData(args, args.test_sample_rate, sample_pattern['fastmri'], mode='test')
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    model = MoDL(in_channels=args.in_channels, out_channels=args.out_channels, num_layers=args.num_layers, device=0)
    # load checkpoint

    model_path = os.path.join(args.model_save_path, 'client_siat_4x.pth')
    # model_path = os.path.join(args.model_save_path, 'client_cc359_4x.pth')
    # model_path = os.path.join(args.model_save_path, 'client_fastMRI_4x.pth')

    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path, map_location='cuda:{}'.format(0))
    # CLIENTS = ['siat', 'cc359', 'fastMRI']model_fastMRI  server_model
    new_state_dict = remove_module(checkpoint['model_siat'])  # revove 'module'
    model.load_state_dict(new_state_dict)  # ####
    print('The model is loaded.')
    model = model.cuda(0)

    print('Now testing {}.'.format(args.exp_name))
    model.eval()
    with torch.no_grad():
        mean_psnr, mean_ssim, mean_psnr_zerof, mean_ssim_zerof, average_time, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        t = tqdm(test_loader, desc='testing', total=int(len(test_loader)))
        batch_psnr, batch_ssim, batch_psnr_zerof, batch_ssim_zerof = [], [], [], []
        for iter_num, data in enumerate(t):
            image = data[0].to(device)
            label = data[1].to(device)
            mask = data[2].to(device)
            fname = data[3]
            slice_id = data[4]

            input = torch.view_as_real(image).permute(0, 3, 1, 2).contiguous()
            # inference
            start_time = time.time()
            output = model(input, mask)
            output = torch.view_as_complex(output.permute(0, 2, 3, 1).contiguous())
            infer_time = time.time() - start_time
            average_time += infer_time
            # calculate and print test information
            under_img, output, label = image.detach().cpu().numpy(), output.detach().cpu().numpy(), label.detach().cpu().numpy()
            total_num += under_img.shape[0]

            for i in range(under_img.shape[0]):
                name = args.results_save_path + fname[i] + '_' + str(slice_id[i].item()) + '_1Drandom_4x.png'
                under_slice, output_slice, label_slice = np.abs(under_img[i]), np.abs(output[i]), np.abs(label[i])
                psnr = peak_signal_noise_ratio(label_slice, output_slice, data_range=label_slice.max())
                ssim = structural_similarity(label_slice, output_slice, data_range=label_slice.max())

                psnr_zerof = peak_signal_noise_ratio(label_slice, under_slice, data_range=label_slice.max())
                ssim_zerof = structural_similarity(label_slice, under_slice, data_range=label_slice.max())
                if args.save_results:
                    if not os.path.exists(args.results_save_path):
                        os.makedirs(args.results_save_path)
                    save_results(name, under_slice, output_slice, label_slice, psnr_zerof, ssim_zerof, psnr, ssim)
                batch_psnr.append(psnr)
                batch_ssim.append(ssim)
                batch_psnr_zerof.append(psnr_zerof)
                batch_ssim_zerof.append(ssim_zerof)

        mean_psnr = np.mean(batch_psnr)
        mean_ssim = np.mean(batch_ssim)
        mean_psnr_zerof = np.mean(batch_psnr_zerof)
        mean_ssim_zerof = np.mean(batch_ssim_zerof)
        average_time /= total_num
    print(
        'average_time:{:.5f}s\tmean_zerof_psnr:{:.5f}\tmean_zerof_ssim:{:.5f}\tmean_test_psnr:{:.5f}'
        '\tmean_test_ssim:{:.5f}'.format(average_time, mean_psnr_zerof, mean_ssim_zerof, mean_psnr, mean_ssim))


if __name__ == '__main__':
    args_ = parser.parse_args()
    validate(args_)