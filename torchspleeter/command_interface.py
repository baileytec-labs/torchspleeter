"""

This provides an interface to interact with the spleeter system on


"""

import os
from torchspleeter.estimator import Estimator
import argparse
import uuid
import numpy as np
import librosa
import soundfile
import torch
import pydub
import os
import shutil

def split_to_parts(inputaudiofile,outputdir,instruments=2,models=[]):
    filedata=pydub.AudioSegment.from_file(inputaudiofile)
    sr=filedata.frame_rate
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #es = Estimator(2, './checkpoints/2stems/model').to(device)
    #es = Estimator(2, ['./checkpoints/2stems/testcheckpoint0.ckpt','./checkpoints/2stems/testcheckpoint1.ckpt']).to(device)
    es=Estimator()
    es.eval()

    # load wav audio
    testaudiofile=inputaudiofile
    channels=filedata.channels
    mono_selection=False
    if channels==1:
        mono_selection=True
        multichannel=pydub.AudioSegment.from_mono_audiosegments(filedata,filedata)
        testaudiofile=os.path.join(outputdir,"/tmp/"+str(uuid.uuid4())+"."+testaudiofile.split('.')[-1])
        #testaudiofile=testaudiofile.split('.')[0]+"-stereo."+testaudiofile.split('.')[-1]
        multichannel.export(out_f=testaudiofile,format=testaudiofile.split('.')[-1])
    print(mono_selection)
    print(channels)
    wav, _ = librosa.load(testaudiofile, mono=False, res_type='kaiser_fast',sr=sr)
    wav = torch.Tensor(wav).to(device)
    if mono_selection:
        shutil.rmtree(os.path.join(outputdir,"/tmp"))
        #os.remove(testaudiofile)
    wavs = es.separate(wav)
    outputname=str(uuid.uuid4())
    returnarray=[]
    for i in range(len(wavs)):
        finaloutput=os.path.join(outputdir,outputname)
        fname = '-out_{}.wav'.format(i)
        fname=finaloutput+fname
        print('Writing ',fname)
        soundfile.write(fname, wavs[i].cpu().detach().numpy().T, sr, "PCM_16")
        returnarray.append(fname)
        # write_wav(fname, np.asfortranarray(wavs[i].squeeze().numpy()), sr)
    return returnarray

def get_file_list(dirname):
    outputfilelist=[]
    for subdir,dirs,files in os.walk(dirname):
        for file in files:
            outputfilelist.append(os.path.join(subdir,file))
    
    return outputfilelist


def main():
    parser = argparse.ArgumentParser(description='torchspleeter allows you to separate instrumentals from audio (vocals, instruments, background noise, etc) in a simple, cross platform manner')
    parser.add_argument('-i', '--inputfile', help='Input Audio File to split into instrumentals', required=True)
    parser.add_argument('-o', '--output', help='Output directory to deposit split audio', required=True)
    parser.add_argument('-n','--number',help="Number of instruments in the model (default 2)",required=False,default=2,type=int)
    parser.add_argument('-m','--modeldir',help="directory containing number of pre-converted torch compatible model components",required=False)
    args = vars(parser.parse_args())
    print(args)
    if args['modeldir'] is not None:
        modelfiles=get_file_list(args['modeldir'])
        if len(modelfiles) != args['number']:
            raise ValueError("You must have the same number of models as you do number of instruments!")
    else:
        args['modeldir']=[]
    outputfiles=split_to_parts(args['inputfile'],args['output'],args['number'],args['modeldir'])
    print("Your output files are:")
    for item in outputfiles:
        print(item)

if __name__ == "__main__":
    main()