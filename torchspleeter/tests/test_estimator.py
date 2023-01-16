import numpy as np
import librosa
import soundfile
import torch
import pydub
import os
from torchspleeter.estimator import Estimator
dirname = os.path.dirname(__file__)
testfilename = os.path.join(dirname, 'checkpoints/2stems/audio_example.mp3')

if __name__ == '__main__':
    sr = 44100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #es = Estimator(2, './checkpoints/2stems/model').to(device)
    #es = Estimator(2, ['./checkpoints/2stems/testcheckpoint0.ckpt','./checkpoints/2stems/testcheckpoint1.ckpt']).to(device)
    es=Estimator()
    es.eval()

    # load wav audio
    testaudiofile=testfilename
    filestats=pydub.AudioSegment.from_file(testaudiofile)
    channels=filestats.channels
    mono_selection=False
    if channels==1:
        mono_selection=True
        multichannel=pydub.AudioSegment.from_mono_audiosegments(filestats,filestats)
        testaudiofile=testaudiofile.split('.')[0]+"-stereo."+testaudiofile.split('.')[-1]
        multichannel.export(out_f=testaudiofile,format=testaudiofile.split('.')[-1])
    print(mono_selection)
    print(channels)
    wav, _ = librosa.load(testaudiofile, mono=False, res_type='kaiser_fast',sr=sr)
    wav = torch.Tensor(wav).to(device)
    if mono_selection:
        os.remove(testaudiofile)


    # normalize audio
    # wav_torch = wav / (wav.max() + 1e-8)

    wavs = es.separate(wav)
    for i in range(len(wavs)):
        fname = 'output/out_{}.wav'.format(i)
        print('Writing ',fname)
        soundfile.write(fname, wavs[i].cpu().detach().numpy().T, sr, "PCM_16")
        # write_wav(fname, np.asfortranarray(wavs[i].squeeze().numpy()), sr)
