import torch
from tqdm import tqdm
from util.metric import psnr_slice, ssim_slice
GAMMA = 0.1


def client_train(model, optimizer, dataloader, criterion, device, regularizer):
    model.train()
    loss, psnr, ssim = 0., 0., 0.
    t = tqdm(dataloader, desc='client_train', total=int(len(dataloader)))
    for _, data in enumerate(t):
        image = data[0].to(device)
        label = data[1].to(device)
        mask = data[2].to(device)

        input = torch.view_as_real(image).permute(0, 3, 1, 2).contiguous()
        output = model(input, mask)

        output = torch.view_as_complex(output.permute(0, 2, 3, 1).contiguous())
        batch_loss = criterion(torch.abs(output), torch.abs(label)) + GAMMA*regularizer

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        psnr += psnr_slice(label, output)
        ssim += ssim_slice(label, output)

    loss /= len(dataloader)
    psnr /= len(dataloader)
    ssim /= len(dataloader)

    return {'loss': loss, 'psnr': psnr, 'ssim': ssim}


def client_eval(model, dataloader, criterion, device):
    model.eval()
    criterion.eval()
    criterion.to(device)
    loss, psnr, ssim = 0., 0., 0.
    t = tqdm(dataloader, desc='client_eval', total=int(len(dataloader)))
    for _, data in enumerate(t):
        image = data[0].to(device)
        label = data[1].to(device)
        mask = data[2].to(device)

        input = torch.view_as_real(image).permute(0, 3, 1, 2).contiguous()
        output = model(input, mask)

        output = torch.view_as_complex(output.permute(0, 2, 3, 1).contiguous())
        batch_loss = criterion(torch.abs(output), torch.abs(label))

        loss += batch_loss.item()
        psnr += psnr_slice(label, output)
        ssim += ssim_slice(label, output)

    loss /= len(dataloader)
    psnr /= len(dataloader)
    ssim /= len(dataloader)

    return {'loss': loss, 'psnr': psnr, 'ssim': ssim}