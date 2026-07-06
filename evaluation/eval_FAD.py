import os
import numpy as np
import scipy
import pandas as pd
import torch
import torchaudio
import torch.nn as nn
from tqdm import tqdm
from torchvggish import vggish, vggish_input


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#mu_r, cov_r = torch.load('mu_cov_real.pt')
#
#fad_dict = {}

model = vggish()
model.eval()
model.postprocess = False
model.embeddings = nn.Sequential(*list(model.embeddings.children())[:-1])
model.to(device)


# mu_r, cov_r = torch.load('mu_cov_real(1).pt')
# dir_name = "got_2_1"
# dir_name1 = "got_2"


dataset_path = "c://Users/laure/Downloads/archive_Libri/LibriSpeech/test-clean"
converted_path = "c://Users/laure/Downloads/archive_Libri/eval/Phoneme_Hallucinator/unexpanded"

csv_path = "c://Users/laure/Downloads/archive_Libri/librispeech_test_pairs_text_w.csv"
df = pd.read_csv(csv_path)
#df["sim_our_1"] = None

#scores = []
audio_source = []
audio_converted = []

for idx, row in tqdm(df.iterrows(), total=len(df)):

    src_stem = os.path.splitext(os.path.basename(row["source_file"]))[0]
    tgt_stem = os.path.splitext(os.path.basename(row["target_file"]))[0]

    wav_name = f"{src_stem}_to_{tgt_stem}.wav"
    converted_wav = os.path.join(converted_path, wav_name)

    if not os.path.exists(converted_wav):
        print(f"Missing: {wav_name}")
        continue

    # Target (reference speaker)
    source_wav = os.path.join(dataset_path, row["source_file"])
    audio_source.append(source_wav)
    target_wav = os.path.join(dataset_path, row["target_file"])
    #audio_source.append(target_wav)

    audio_converted.append(converted_wav)
    


# data = []
# for fname in os.listdir(dir_name):
#         if fname.endswith(".wav"):
#                  data.append({"fname": fname.split('.')[0], "to": int(fname.split('_')[-1][:-4])})
# df = pd.DataFrame(data)
# df.to_csv('combinations.csv', index=False)    

source_embeddings = []
with torch.no_grad():
    for a in tqdm(audio_source):
        aa, _ = torchaudio.load(a)
        source_embeddings.append(model(vggish_input.waveform_to_examples(aa[0].numpy(),16000).to(device)).detach().cpu().to(dtype=torch.float16))
source_embeddings = torch.vstack(source_embeddings)
source_embeddings = source_embeddings.to(dtype=torch.float32)
mu_r = torch.mean(source_embeddings, axis=0)
cov_r = torch.cov(source_embeddings.T)

converted_embeddings = []
with torch.no_grad():
    for a in tqdm(audio_converted):
        aa, _ = torchaudio.load(a)
        converted_embeddings.append(model(vggish_input.waveform_to_examples(aa[0].numpy(),16000).to(device)).detach().cpu().to(dtype=torch.float16))
converted_embeddings = torch.vstack(converted_embeddings)
converted_embeddings = converted_embeddings.to(dtype=torch.float32)
mu_f = torch.mean(converted_embeddings, axis=0)
cov_f = torch.cov(converted_embeddings.T)


# fake_audio = []
# for fname in os.listdir(dir_name):
#         if fname.endswith(".wav"):
#             fake_audio.append(os.path.join(dir_name, fname))
# for fname in os.listdir(dir_name1):
#         if fname.endswith(".wav"):
#             fake_audio.append(os.path.join(dir_name1, fname))
# fake_embeddings = []
# with torch.no_grad():
#     for a in tqdm(fake_audio):
#         aa, _ = torchaudio.load(a)
#         fake_embeddings.append(model(vggish_input.waveform_to_examples(aa[0].numpy(),16000).to(device)).detach().cpu().to(dtype=torch.float16))
# fake_embeddings = torch.vstack(fake_embeddings)
# fake_embeddings = fake_embeddings.to(dtype=torch.float32)
# mu_f = torch.mean(fake_embeddings, axis=0)
# cov_f = torch.cov(fake_embeddings.T)
mu_diff = mu_r-mu_f
diffnorm_sq = mu_diff@mu_diff
# Calculate trace term
cov_prod_np = cov_r.cpu().numpy().dot(cov_f.cpu().numpy())
covmean_sqrtm, _ = scipy.linalg.sqrtm(cov_prod_np, disp=False)
if np.iscomplexobj(covmean_sqrtm):
    covmean_sqrtm = covmean_sqrtm.real  # Ensure real values
tr_covmean = torch.tensor(np.trace(covmean_sqrtm), dtype=torch.float32) #, device=device)
fad = diffnorm_sq + torch.trace(cov_r) + torch.trace(cov_f) - 2*tr_covmean
print(f"FAD for {converted_path}: {fad}")

#our_no_reg: 0.7535915374755859 our_no_reg: 0.8253574371337891 our_no_reg: 0.9224834442138672
#our_no_reg_1: 0.91314697265625 our_no_reg_1: 0.9657707214355469 our_no_reg_1: 1.0443687438964844
#our_no_reg_5: 0.7962932586669922 our_no_reg_5: 0.8672828674316406 our_no_reg_5: 0.9635753631591797
#our: 0.7809429168701172 our: 0.8533668518066406 our: 0.9509201049804688
#our_1: 0.9297580718994141 our_1: 0.9866695404052734 our_1: 1.0694541931152344
#our_5: 0.8075523376464844 our_5: 0.8802623748779297 our_5: 0.9778690338134766
#our_SB: 0.7502689361572266 our_SB: 0.8290424346923828 our_SB: 0.93255615234375
#our_SB_1: 0.8537654876708984 our_SB_1: 0.9277534484863281 our_SB_1: 1.0265846252441406
#our_SB_5: 0.7780189514160156 our_SB_5: 0.8567066192626953 our_SB_5: 0.9600620269775391
# MeanVC: 4.483150482177734 4.6953535079956055
# freevc: 2.579212188720703 2.800403594970703
# PhH_exp: 1.6509895324707031 1.7943038940429688
# PhH_unexp: 2.0311050415039062 2.1758995056152344