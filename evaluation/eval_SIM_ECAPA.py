import os
import sys
from glob import glob
import json
import pandas as pd

import torch
import torchaudio
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
#import pyloudnorm as pyln

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


device = 'cuda' if torch.cuda.is_available() else 'cpu'


from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="C:/speechbrain_models/ecapa",
    run_opts={"device": "cuda"}
)



def load_audio(path):
    wav, sr = torchaudio.load(path)

    # mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)

    else:
        wav = wav.squeeze(0)

    # resample
    if sr != 16000:
        wav = torchaudio.functional.resample(
            wav,
            sr,
            16000
        )
    return wav

def extract_spk_emb(path):
    # Load waveform (1D tensor)
    wav = load_audio(path)

    # Add batch dimension: [1, T]
    wav = wav.unsqueeze(0).to(device)

    # Relative length (full utterance)
    rel_length = torch.tensor([1.0], device=device)

    with torch.no_grad():
        emb = classifier.encode_batch(wav, rel_length)

    # Remove batch dimensions
    emb = emb.squeeze(0).squeeze(0)

    # Normalize for cosine similarity
    #emb = F.normalize(emb, dim=0)

    return emb.cpu()

def speaker_similarity(wav1, wav2):
    emb1 = extract_spk_emb(wav1) # get_embedding(wav1)
    emb2 = extract_spk_emb(wav2) # get_embedding(wav2)
    
    return F.cosine_similarity(emb1, emb2, dim=0).item()


def evaluate_dataset(pairs):
    scores = []

    for conv_path, tgt_path in pairs:
        sim = speaker_similarity(conv_path, tgt_path)
        scores.append(sim)

    return sum(scores) / len(scores)


dataset_path = "c://Users/laure/Downloads/archive_Libri/LibriSpeech/test-clean"
converted_path = "c://Users/laure/Downloads/archive_Libri/eval/Phoneme_Hallucinator/unexpanded"

csv_path = "c://Users/laure/Downloads/archive_Libri/librispeech_test_pairs_text_w.csv"
df = pd.read_csv(csv_path)
#df["sim_our_1"] = None

scores = []

for idx, row in tqdm(df.iterrows(), total=len(df)):

    src_stem = os.path.splitext(os.path.basename(row["source_file"]))[0]
    tgt_stem = os.path.splitext(os.path.basename(row["target_file"]))[0]

    wav_name = f"{src_stem}_to_{tgt_stem}.wav"
    converted_wav = os.path.join(converted_path, wav_name)

    if not os.path.exists(converted_wav):
        print(f"Missing: {wav_name}")
        continue

    # Target (reference speaker)
    #converted_wav = os.path.join(dataset_path, row["source_file"])
    target_wav = os.path.join(dataset_path, row["source_file"])
    #target_wav = os.path.join(dataset_path, row["target_file"])

    # Converted filename
    # src_name = Path(row["source_file"]).stem
    # tgt_name = Path(row["target_file"]).stem

    # converted_wav = converted_path / f"{src_name}__to__{tgt_name}.wav"

    # if not converted_wav.exists():
    #     print(f"Missing: {converted_wav.name}")
    #     scores.append(None)
    #     continue

    sim = speaker_similarity(converted_wav, target_wav)
    scores.append(sim)

# df["sim_e_our"] = scores

# print(df["sim_e_our"].mean(), df["sim_e_our"].std())

# df.to_csv("c://Users/laure/Downloads/archive_Libri/librispeech_test_pairs_text_w.csv", index=False)

import numpy as np
print(scores, np.mean(scores))

#SB
#
#
#0.28445440844014175 0.11018182232972626
#our
#
#
#0.27440009403028276 0.10937088885638895
#tube
#
#
#0.27313398101281744 0.11080351
#MeanVC 0.15827182351206323 0.09059735275133236
#freevc 0.331849630890798 0.1084979109134567
#PhH_exp 0.42366318758719296 0.1265964837657654
#PhH_unexp 0.6141650286453371 0.11499198307178775