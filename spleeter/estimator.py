import math

import torch
import torch.nn.functional as F
from torch import nn
# from torchaudio.functional import istft

from .unet import UNet
#I pre-converted the models so that there's no tensorflow required.
#from .util import tf2pytorch


def load_ckpt(model, ckpt):
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict:
            target_shape = state_dict[k].shape
            assert target_shape == v.shape
            state_dict.update({k: torch.from_numpy(v)})
        else:
            print('Ignore ', k)

    model.load_state_dict(state_dict)
    return model


def pad_and_partition(tensor, T):
    """
    pads zero and partition tensor into segments of length T

    Args:
        tensor(Tensor): BxCxFxL

    Returns:
        tensor of size (B*[L/T] x C x F x T)
    """
    print("in pad and partition, sizing tensor")
    old_size = tensor.size(3)
    print("getting new size")
    new_size = math.ceil(old_size/T) * T
    print("setting tensor to a F pad")
    tensor = F.pad(tensor, [0, new_size - old_size])
    print("getting tensor shape")
    [b, c, t, f] = tensor.shape
    print("setting split")
    split = new_size // T
    print("returning torch.cat(torch.split")
    return torch.cat(torch.split(tensor, T, dim=3), dim=0)

#The answer isn't to have tensorflow somewhere on the side to convert the models, it's to pre-convert
#the models and have the models fully loaded into torch
class Estimator(nn.Module):
    def __init__(self, num_instrumments, checkpoint_path):
        super(Estimator, self).__init__()
        self.num_instruments=num_instrumments
        # stft config
        self.F = 1024
        self.T = 512
        self.win_length = 4096
        self.hop_length = 1024
        self.win = nn.Parameter(
            torch.hann_window(self.win_length),
            requires_grad=False
        )

        self.ckpts = torch.load(checkpoint_path)#, num_instrumments)

        # filter
        #this loads both net models into memory... I wonder if we can do this one at a time?
        #cpu expensive, but possible 
        #self.instruments = nn.ModuleList()
        #for i in range(num_instrumments):
        #    print('Loading model for instrumment {}'.format(i))
        #    net = UNet(2)
        #    ckpt = ckpts[i]
        #    net = load_ckpt(net, ckpt)
        #    net.eval()  # change mode to eval
        #    self.instruments.append(net)

    def compute_stft(self, wav):
        """
        Computes stft feature from wav

        Args:
            wav (Tensor): B x L
        """
        print("Now getting stft in compute stft")
        stft = torch.stft(wav, n_fft=self.win_length, hop_length=self.hop_length, window=self.win,
                          center=True, return_complex=False, pad_mode='constant')

        # only keep freqs smaller than self.F
        print("first stft calc")
        stft = stft[:, :self.F, :, :]
        print("second stft calc")
        real = stft[:, :, :, 0]
        print("third stft calc")
        im = stft[:, :, :, 1]
        print("fourth stft calc")
        mag = torch.sqrt(real ** 2 + im ** 2)
        print("mag calc")

        return stft, mag

    def inverse_stft(self, stft):
        """Inverses stft to wave form"""
        print("in inverse stft")
        print("setting pad")
        pad = self.win_length // 2 + 1 - stft.size(1)
        print("setting stft")
        stft = F.pad(stft, (0, 0, 0, 0, 0, pad))
        print("making wav")
        wav = torch.istft(stft, self.win_length, hop_length=self.hop_length, center=True,
                    window=self.win)
        print("wav.detach")
        return wav.detach()

    def separate(self, wav):
        """
        Separates stereo wav into different tracks corresponding to different instruments

        Args:
            wav (tensor): 2 x L
        """

        # stft - 2 X F x L x 2
        # stft_mag - 2 X F x L
        print("In separate, now getting stft and stfg_mag")
        stft, stft_mag = self.compute_stft(wav)
        print("In separate, now getting stft size")
        L = stft.size(2)

        # 1 x 2 x F x T
        print("in separate, unsqueezing")
        stft_mag = stft_mag.unsqueeze(-1).permute([3, 0, 1, 2])
        print("in separate, pad and partitioning...")
        stft_mag = pad_and_partition(stft_mag, self.T)  # B x 2 x F x T
        print("now transposing...")
        stft_mag = stft_mag.transpose(2, 3)  # B x 2 x T x F
        print("getting the shape...")
        B = stft_mag.shape[0]

        # compute instruments' mask
        masks = []
        #so it fails here from massive memory consumption on a docker container, but not on bare metal.
        #interesting.
        #self.instruments = nn.ModuleList()
        #for i in range(num_instrumments):
        #    print('Loading model for instrumment {}'.format(i))
        #    net = UNet(2)
        #    ckpt = ckpts[i]
        #    net = load_ckpt(net, ckpt)
        #    net.eval()  # change mode to eval
        #    self.instruments.append(net)
        for i in range(self.num_instruments):
            print("loading model for instrument "+str(i))
            net=UNet(2)
            ckpt=self.ckpts[i]
            net=load_ckpt(net,ckpt)
            print("netting the mask...")
            mask=net(stft_mag)
            print("appending the mask")
            masks.append(mask)
        #for net in self.instruments:
        #    print("netting a mask...")
        #    mask = net(stft_mag)
        #    masks.append(mask)

        # compute denominator
        print("summing masks")
        mask_sum = sum([m ** 2 for m in masks])
        print("adding magic number to mask sum")
        mask_sum += 1e-10

        wavs = []
        for mask in masks:
            print("making mask again")
            mask = (mask ** 2 + 1e-10/2)/(mask_sum)
            print("transposing mask")
            mask = mask.transpose(2, 3)  # B x 2 X F x T

            print("performing torch cat on mask")
            mask = torch.cat(
                torch.split(mask, 1, dim=0), dim=3)
            print("squeezing mask")
            mask = mask.squeeze(0)[:,:,:L].unsqueeze(-1) # 2 x F x L x 1
            print("multiplying stft * mask")
            stft_masked = stft *  mask
            print("doing inverse stft on wav_masked")
            wav_masked = self.inverse_stft(stft_masked)
            print("appending to wavs")
            wavs.append(wav_masked)

        return wavs
