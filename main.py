import os
import torch
from torch.utils.data import Dataset, DataLoader
from compose_models import get_speech_to_text_model, get_wav2vec2_model
from data_loader import DATAReader
from tqdm import tqdm
from Grad_TTS.inferenceTTS import InferenceTTS

class AudioDeepfake:
    def __init__(
            self,
            device,
            data_loader,
            t_processor_1,
            t_model_1,
            # t_processor_2,
            # t_model_2
    ) -> None:
        self.device = device
        self.data_loader = data_loader
        self.t_processor_1 = t_processor_1
        self.t_model_1 = t_model_1.to(device)
        # self.t_processor_2 = t_processor_2
        # self.t_model_2 = t_model_2
    


    # Function to transcribe audio to text using Wav2Vec 2.0
    def transcribe_audios(self, transcription_file_path, save_dir_path):

        progress_bar = tqdm(self.data_loader, unit="batch", leave=True)
        for index, audio_batch in enumerate(progress_bar):
            audios_paths = audio_batch[0]
            audios = audio_batch[1].unsqueeze(1).to(self.device, dtype=torch.float)
            labels = audio_batch[2]
            # forged = train_sample[2].unsqueeze(1).to(self.device, dtype=torch.float)
            input_values = self.t_processor_1(audios,sampling_rate=16000, return_tensors="pt").input_values
            input_values = input_values.to(self.device)
            input_values = input_values.squeeze(0)
            input_values = input_values.squeeze(1)
            # input_values = input_values.half()  # Convert input to FP16

            # Get the predicted logits from the model
            with torch.no_grad():
                logits = self.t_model_1(input_values).logits
            # Decode the predicted logits to text
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = self.t_processor_1.batch_decode(predicted_ids)
            # store transcriptions with file and speaker ids
            with open(transcription_file_path, "a") as file:
                for transcription, file_path, file_id, speaker_id in zip(transcriptions, audios_paths, labels[0],labels[1]):
                    # file.write(f"{file_id:<20}|{speaker_id:<20}|{transcription:<100}\n")
                    # Construct absolute file path
                    audio_file_path = os.path.join(save_dir_path, 'audios', f"{file_id}.wav")  # New File Path, where we need to synthesize audio , file_path for the original file path for training from scratch
                    file.write(f"{audio_file_path}|{transcription}|{int(speaker_id[-3:])}\n")  # Need to convert speaker id to numeric!
            
        return transcription_file_path

    def read_transcriptions(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line using '|' as the delimiter
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    file_id = parts[0].strip()
                    speaker_id = parts[1].strip()
                    transcription = parts[2].strip()
                    # Append the parsed data as a tuple
                    data.append((file_id, speaker_id, transcription))
        return data    

        

# a method to convert audios to text in a file with corresponding speaker Id
# a method to convert that text to Audio Deepfake with corresponding speaker Id






def load_models(device):
    # G = DDP(G, device_ids=[local_rank])
    # D = DDP(D, device_ids=[local_rank])
    # Load classification models
    t_processor_1, t_model_1 = get_wav2vec2_model(device)
    # t_processor_2, t_model_2 = get_speech_to_text_model(device)

    return t_processor_1, t_model_1#,t_processor_2,t_model_2



def prepare_dataloader(batch_size, n_workers, data_split):
    train_dataset = DATAReader(split = data_split)
    # sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, #sampler=sampler,
        num_workers=n_workers, pin_memory=True, drop_last=True
    )
    return train_loader



def main(batch_size, num_workers, device_id):
    # read partial labels
    # cut audio sample as per partial segment (specify partial segment  min lenght)
    # convert audio segment to text
    # generate audio with text with diffusion model    
    # import datetime
    model_name = "Grad_tts"
    dataset = "ASVspoof"
    data_split = "TEST"
    # time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir_path = f"{model_name}_{dataset}_{data_split}"
    audios_path = os.path.join(save_dir_path,"audios")
    device = torch.device('cuda', device_id)
    data_loader = prepare_dataloader(batch_size, num_workers, data_split)
    t_processor_1, t_model_1 = load_models(device)
    transcription_file_path = os.path.join(save_dir_path, "transcription.txt")

    audio_deepfake = AudioDeepfake(device,data_loader, t_processor_1,t_model_1)

    os.makedirs(save_dir_path, exist_ok=True)  # Ensure the directory exists
    os.makedirs(audios_path, exist_ok=True)
    if not os.path.exists(transcription_file_path):
        transcription_file_path = audio_deepfake.transcribe_audios(transcription_file_path, save_dir_path)
    else:
        print("Reading the existing trascritions...")
    transcription_data = audio_deepfake.read_transcriptions(transcription_file_path)
    gradtts_inference = InferenceTTS()
    gradtts_inference.synthesize_audios(transcription_data,50)
    # Print the extracted data
    # for file_id, speaker_id, transcription in transcription_data:
    #     print(f"File ID: {file_id}, Speaker ID: {speaker_id}, Transcription: {transcription}")
   



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=24, help='Batch size')
    parser.add_argument('--num_workers', '-w', type=int, default=16, help='Number of data loader workers')
    args = parser.parse_args()
    args.save_output = "yes"
    args.device_id = 0

    main(args.batch_size, args.num_workers, args.device_id)
