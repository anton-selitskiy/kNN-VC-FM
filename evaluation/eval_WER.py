import os
import numpy as np
import scipy
import pandas as pd
import random
import torch
import torchaudio
import torch.nn as nn
from tqdm import tqdm
#from torchvggish import vggish, vggish_input
import whisper
from jiwer import wer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Whisper model
model = whisper.load_model("base")  # or "small", "medium", etc.
model.to(device)

dir_name = "c://Users/laure/Downloads/archive_Libri/eval/Phoneme_Hallucinator/unexpanded" #our_1" #"got_256_5_0" #"dot_4_1"



dataset_path = r"C:\Users\laure\Downloads\archive_Libri\LibriSpeech\test-clean" #train-clean-100"



# Add a new column
csv_path = "c://Users/laure/Downloads/archive_Libri/librispeech_test_pairs_text_w.csv"
df = pd.read_csv(csv_path)
df["wer_PhH_unexp"] = None


total_wer = 0.0
count = 0

#with open("wer_results_libri.txt", "w", encoding="utf-8") as f:

for idx, row in tqdm(df.iterrows(), total=len(df)):

    src_stem = os.path.splitext(os.path.basename(row["source_file"]))[0]
    tgt_stem = os.path.splitext(os.path.basename(row["target_file"]))[0]

    #wav_name = f"{src_stem}__to__{tgt_stem}.wav"
    wav_name = f"{src_stem}_to_{tgt_stem}.wav"
    wav_path = os.path.join(dir_name, wav_name)

    if not os.path.exists(wav_path):
        print(f"Missing: {wav_name}")
        continue

    ####################################################
    # Load audio
    ####################################################

    waveform, sr = torchaudio.load(wav_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    audio = waveform.squeeze().numpy()

    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)

    ####################################################
    # Whisper transcription
    ####################################################

    result = model.decode(mel)
    hypothesis = result.text.lower().strip()

    ####################################################
    # Compute WER
    ####################################################

    reference = row["text"].lower().strip()

    score = wer(reference, hypothesis)

    df.at[idx, "wer_PhH_unexp"] = score

    total_wer += score
    count += 1

    #f.write(f"{wav_name}\t{score:.4f}\n")

print(f"Average WER: {total_wer / count:.4f}")

df.to_csv("c://Users/laure/Downloads/archive_Libri/librispeech_test_pairs_text_w.csv", index=False)

#          
#SB
#1 step of integration WER: 0.2326 std  0.158685
#5 steps of integration WER: 0.2306 std   0.160663
#50 steps of integration WER: 0.2282 std 0.159731
#our
#1 step of integration WER: 0.2342 std 0.160281
#5 steps of integration WER: 0.2293 std 0.159800
#50 step of integration WER: 0.2289 std 0.163339
#tube
#1 step of integration Average WER: 0.2316 std: 0.158902
#5 steps of integration Average WER: 0.2299 std 0.160470
#50 steps of integration 0.2288 std 0.159085

#MeanVC:  0.5003
#freevc: 0.2435
#PhH_exp: 0.2528
#PhH_unexp: 0.4885