import copy
import torch
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm
from util.metric import psnr_slice, ssim_slice
# distributed
shared_weight = ['layers.conv1.weight', 'layers.conv1.bias', 'layers.conv2.weight', 'layers.conv2.bias',
                 'layers.conv3.weight', 'layers.conv3.bias', 'layers.conv4.weight', 'layers.conv4.bias',
                 'layers.conv5.weight', 'layers.conv5.bias', 'layers.conv6.weight', 'layers.conv6.bias', ]


# aggregation
def communication(server_model, clients, aggregation_weights):
    update_state = OrderedDict()
    with torch.no_grad():
        for k, client in enumerate(clients):
            local_state = client.state_dict()
            for key in server_model.state_dict().keys():
                if k == 0:
                    update_state[key] = local_state[key] * aggregation_weights[k]
                else:
                    update_state[key] += local_state[key] * aggregation_weights[k]
        server_model.load_state_dict(update_state)

        for key in server_model.state_dict().keys():
            if key in shared_weight:
                for k, client in enumerate(clients):
                    client.state_dict()[key].data.copy_(server_model.state_dict()[key])

    return server_model, clients


def validation(server_model, dataloaders, criterion, device):

    server_model.eval()
    loss, psnr, ssim = 0., 0., 0.
    avg_loss, avg_psnr, avg_ssim = [], [], []
    with torch.no_grad():
        for idx, dataloader in enumerate(dataloaders):
            t = tqdm(dataloader, desc='server_eval', total=int(len(dataloader)))
            for i, data in enumerate(t):
                image = data[0].to(device)
                label = data[1].to(device)
                mask = data[2].to(device)

                input = torch.view_as_real(image).permute(0, 3, 1, 2).contiguous()
                output = server_model(input, mask)

                output = torch.view_as_complex(output.permute(0, 2, 3, 1).contiguous())
                batch_loss = criterion(torch.abs(output), torch.abs(label))

                # metric
                loss += batch_loss.item()
                psnr += psnr_slice(label, output)
                ssim += ssim_slice(label, output)
            loss /= len(dataloader)
            psnr /= len(dataloader)
            ssim /= len(dataloader)

            avg_loss.append(loss)
            avg_psnr.append(psnr)
            avg_ssim.append(ssim)

        # avg_loss /= len(dataloaders)
        # avg_psnr /= len(dataloaders)
        # avg_ssim /= len(dataloaders)

    return {'loss': avg_loss, 'psnr': avg_psnr, 'ssim': avg_ssim}