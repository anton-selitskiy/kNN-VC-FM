import os
import sys
from glob import glob
import torch
import torchaudio
import torch.nn.functional as F

import pandas as pd
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'



# wavlm = hubconf.wavlm_large(device=device)
# hifigan, hifigan_cfg = hubconf.hifigan_wavlm(pretrained=True, device=device)
# worker = matcher.KNeighborsVC(wavlm, hifigan, hifigan_cfg, device=device)



from transformers import WavLMModel, AutoFeatureExtractor

model_name = "microsoft/wavlm-base-plus"

processor = AutoFeatureExtractor.from_pretrained(model_name)

wavlm = WavLMModel.from_pretrained(model_name).to(device)

wavlm.eval()


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

    wav = load_audio(path)

    inputs = processor(
        wav,
        sampling_rate=16000,
        return_tensors="pt"
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():

        outputs = wavlm(input_values)

        # [B, T, D]
        hidden = outputs.last_hidden_state

    # mean pooling over time
    emb = hidden.mean(dim=1)

    # normalize
    emb = F.normalize(emb, dim=-1)

    return emb.squeeze(0).cpu()


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
    target_wav = os.path.join(dataset_path, row["target_file"])

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

df["sim_PhH_unexp"] = scores

print(df["sim_PhH_unexp"].mean(), df["sim_PhH_unexp"].std())

df.to_csv("c://Users/laure/Downloads/archive_Libri/librispeech_test_pairs_text_w.csv", index=False)
# dir_name = r"C:\Users\laure\Downloads\archive_Libri\outputs"
# files = glob(os.path.join(dir_name,  "*.wav"))
# #print(files)

# dataset_path = r"C:\Users\laure\Downloads\archive_Libri\LibriSpeech\test-clean" #train-clean-100"

# pairs = []
# for item in files:
#     file_name = os.path.basename(item)
#     f_name = file_name.split("__")[-1]
#     spk_name, chapter, _ = f_name.split("-")
#     #print(file_name, f_name, spk_name, chapter)
#     pairs.append((os.path.join(dataset_path,spk_name,chapter,f_name.split('.')[0]+".flac"),item))

# print(pairs[:3])
# print(evaluate_dataset(pairs))

#SB
#0.8837671432786  0.047733
#0.8826651750630095 0.04776240565880
#0.8821441869699318 0.047783950477
#our
#0.8836248  0.047495
#0.88249172836  0.047718
#0.881958   0.047739
#tube
#0.8836100766676983 0.0475547344090
#0.8825920041280848 0.0476646263
#0.8819199024720956 0.047804021281596894

# SB
# Mean	Std	95% CI
# 0.8837671433	0.0477330	(0.881179, 0.886355)
# 0.8826651751	0.0477624	(0.880075, 0.885255)
# 0.8821441870	0.0477840	(0.879553, 0.884735)
# Our
# Mean	Std	95% CI
# 0.8836248000	0.047495	(0.881050, 0.886199)
# 0.8824917284	0.047718	(0.879904, 0.885080)
# 0.8819580000	0.047739	(0.879370, 0.884546)
# Tube
# Mean	Std	95% CI
# 0.8836100767	0.0475547	(0.881032, 0.886188)
# 0.8825920041	0.0476646	(0.880010, 0.885174)
# 0.8819199025	0.0478040	(0.879327, 0.884513)
#MeanVC 0.8396086518546098 0.05015309286643498
#freevc 0.8743253407132534 0.049182002497896976
#PhH_exp 0.8803759088043038 0.04831407260925165
#PhH_unexp