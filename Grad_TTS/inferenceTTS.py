# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import datetime as dt
import os
import numpy as np
from scipy.io.wavfile import write
import torch
from tqdm import tqdm
import sys

sys.path.append('./Grad_TTS/')
from  Grad_TTS import params 
from model import GradTTS
# from .model.tts import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

sys.path.append('./Grad_TTS/hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN


HIFIGAN_CONFIG = './Grad_TTS/checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './Grad_TTS/checkpts/hifigan.pt'

class InferenceTTS:
    def __init__(
            self,
            # transcriptions,
            # timesteps,
            # speaker_id  # It's included in the transcription data with file ids
    ) -> None:
        # self.transcriptions = transcriptions
        # self.timesteps = timesteps
        self.checkpoint = './Grad_TTS/checkpts/grad-tts-libri-tts.pt'
        # self.speaker_id = speaker_id
        # if not isinstance(speaker_id, type(None)):
        #     assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
        #     self.spk = torch.LongTensor([speaker_id]).cuda()
        # else:
        #     self.spk = None
        print('Initializing Grad-TTS...')
        self.generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                            params.n_enc_channels, params.filter_channels,
                            params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                            params.enc_kernel, params.enc_dropout, params.window_size,
                            params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
        self.generator.load_state_dict(torch.load(self.checkpoint, map_location=lambda loc, storage: loc))
        _ = self.generator.cuda().eval()
        print(f'Number of parameters: {self.generator.nparams}')
        
        print('Initializing HiFi-GAN...')
        with open(HIFIGAN_CONFIG) as f:
            h = AttrDict(json.load(f))
        self.vocoder = HiFiGAN(h)
        self.vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
        _ = self.vocoder.cuda().eval()
        self.vocoder.remove_weight_norm()
    

    # generate training data with original audios then train then synthecise using that trained model!
    def synthesize_audios(self, transcriptions, timesteps):
        # with open(args.file, 'r', encoding='utf-8') as f:
        #     texts = [line.strip() for line in f.readlines()]
        cmu = cmudict.CMUDict('./Grad_TTS/resources/cmu_dictionary')
        
        with torch.no_grad():
            for file_id, text, speaker_id in tqdm(transcriptions, desc="Synthesizing Audios", unit="audio"):
                print(f"Synthesizing: File ID: {file_id}, Speaker ID: {speaker_id}, Transcription: {text}")
                if text is None or len(text) == 0:
                    continue

                # for i, text in enumerate(texts):
                spk = torch.LongTensor([int(speaker_id)]).cuda()

                # print(f'Synthesizing {i} text...', end=' ')
                x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                
                t = dt.datetime.now()
                y_enc, y_dec, attn = self.generator.forward(x, x_lengths, n_timesteps=timesteps, temperature=1.5,
                                                    stoc=False, spk=spk, length_scale=0.91)
                t = (dt.datetime.now() - t).total_seconds()
                print(f'Grad-TTS RTF: {t * 16000 / (y_dec.shape[-1] * 256)}')   # 22050, 16000

                audio = (self.vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

                write(file_id, 16000, audio)   # 22050, 16000

        print('Done. Check out `out` folder for samples.')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')
    parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None, help='speaker id for multispeaker model')
    args = parser.parse_args()
    
    if not isinstance(args.speaker_id, type(None)):
        assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
        spk = torch.LongTensor([args.speaker_id]).cuda()
    else:
        spk = None
    
    print('Initializing Grad-TTS...')
    generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    _ = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')
    
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    
    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'Synthesizing {i} text...', end=' ')
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
            
            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                   stoc=False, spk=spk, length_scale=0.91)
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')

            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            
            write(f'./out/sample_{i}.wav', 22050, audio)

    print('Done. Check out `out` folder for samples.')
