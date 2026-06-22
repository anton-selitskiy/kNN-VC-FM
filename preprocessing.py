from pathlib import Path
import sys
import torch
import torchaudio
import torch.nn as nn
from tqdm import tqdm

sys.path.append('knn-vc')
import matcher
import knnvc_utils
import hubconf

device = 'cuda' if torch.cuda.is_available() else 'cpu'

wavlm = hubconf.wavlm_large(device=device)
hifigan, hifigan_cfg = hubconf.hifigan_wavlm(pretrained=True, device=device)
worker = matcher.KNeighborsVC(wavlm, hifigan, hifigan_cfg, device=device)

audio_root = Path("LibriSpeech/train-clean-100")
flac_files = list(audio_root.rglob("*.flac"))

output_root = Path("LibriSpeech_wavlm/train-clean-100")

for flac_path in tqdm(flac_files):
    try:
        aa, sr = torchaudio.load(flac_path)
        if sr != 16000:
            aa = torchaudio.functional.resample(aa, sr, 16000)
            print(flac_path, 'sr=', sr)
        if aa.shape[0] > 1:
            aa = aa.mean(dim=0, keepdim=True)
            print(flac_path, "stereo")
        aa = aa.to(device)
        with torch.no_grad():
            embeddings = worker.get_features(aa).detach().cpu().half()
        relative_path = flac_path.relative_to(audio_root)
        pt_path = (output_root / relative_path).with_suffix(".pt")
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, pt_path)
    except Exception as e:
        print("Error processing", flac_path, e)
    
