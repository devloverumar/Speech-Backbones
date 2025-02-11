# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch


def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def latest_checkpoint_path(dir_path, regex="grad_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def load_checkpoint(logdir, model, num=None):
    if num is None:
        model_path = latest_checkpoint_path(logdir, regex="grad_*.pt")
    else:
        model_path = os.path.join(logdir, f"grad_{num}.pt")
    print(f'Loading checkpoint {model_path}...')
    model_dict = torch.load(model_path, map_location=lambda loc, storage: loc)
    model.load_state_dict(model_dict, strict=False)
    return model


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_tensor(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return


# Function to transcribe audio to text using Wav2Vec 2.0
def transcribe_audio(audio, processor, model,device):

    input_values = processor(audio,sampling_rate=16000, return_tensors="pt").input_values

    input_values = input_values.to(device)
    input_values = input_values.squeeze(0)
    input_values = input_values.squeeze(1)
    input_values = input_values.half()  # Convert input to FP16

    # Get the predicted logits from the model
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the predicted logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcriptions = processor.batch_decode(predicted_ids)
    return transcriptions


# Function to transcribe audio to text using Wav2Vec 2.0
def transcribe_s2t(audio, processor, model,device):

    audio = audio.squeeze(1).detach().cpu().numpy()
    input_features = processor(
        audio,
        sampling_rate=16_000,
        return_tensors="pt"
    ).input_features

    # Ensure input features match the model's dtype and are on the correct device
    input_features = input_features.to(device).to(model.dtype)

    # Generate transcription
    generated_ids = model.generate(input_features=input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return transcription

