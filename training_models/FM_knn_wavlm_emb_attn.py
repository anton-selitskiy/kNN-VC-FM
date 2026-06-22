import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.distributions import Beta
import math
import random
from tqdm import tqdm



class FlowMatchingDataset(Dataset):

    def __init__(self, emb_root, speakers,
                 k=4, reg=0.1,
                 src_pool_size=8,
                 tgt_pool_size=10):

        self.emb_root = Path(emb_root)
        self.speakers = speakers
        self.k = k
        self.reg = reg
        self.src_pool_size = src_pool_size
        self.tgt_pool_size = tgt_pool_size
        self.max_spk_len = 200

        self.spk_to_embs = {}
        for spk in speakers:
            files = list((self.emb_root / spk).rglob("*.pt"))
            if len(files) > 0:
                self.spk_to_embs[spk] = files

        self.all_files = []
        for spk, files in self.spk_to_embs.items():
            for f in files:
                self.all_files.append((spk, f))

        print("Total files:", len(self.all_files))

    def __len__(self):
        return len(self.all_files)

    def sample_embeddings(self, spk, pool_size):
        files = self.spk_to_embs[spk]
        selected = random.sample(files, min(pool_size, len(files)))

        embs = [torch.load(p) for p in selected]  # [T, D]
        lengths = [e.shape[0] for e in embs]

        embs = torch.cat(embs, dim=0)  # [N, D]
        return embs.float(), lengths

    def __getitem__(self, idx):

        # --- source speaker ---
        src_spk, _ = self.all_files[idx]
        z0, src_lengths = self.sample_embeddings(src_spk, self.src_pool_size)

        # --- target speaker ---
        tgt_spk = random.choice(self.speakers)
        while tgt_spk == src_spk:
            tgt_spk = random.choice(self.speakers)

        tgt_embeddings, _ = self.sample_embeddings(tgt_spk, self.tgt_pool_size)

        # --- OT / NN mapping ---
        z0_norm = F.normalize(z0, dim=-1, eps=1e-6)
        tgt_norm = F.normalize(tgt_embeddings, dim=-1, eps=1e-6)

        sim = z0_norm @ tgt_norm.T
        top_k = sim.topk(self.k, dim=1).indices
        z1 = tgt_embeddings[top_k].mean(dim=1)
         
        # --- split back to utterances ---
        z0_split = torch.split(z0, src_lengths)
        z1_split = torch.split(z1, src_lengths)

        idx_chunk = random.randint(0, len(z0_split) - 1)

        z0 = z0_split[idx_chunk]
        z1 = z1_split[idx_chunk]

        # --- single-file speaker conditioning (IMPORTANT) ---
        spk_file = random.choice(self.spk_to_embs[tgt_spk])
        spk_seq = torch.load(spk_file).float()  # [T_spk, D]
        # spk_seq = F.normalize(spk_seq, dim=-1, eps=1e-6)
        seq_T = spk_seq.shape[0]
        if seq_T > self.max_spk_len:
            start = random.randint(0, seq_T-self.max_spk_len)
            spk_seq = spk_seq[start:start+self.max_spk_len]

        return z0, z1, spk_seq, len(z0), len(spk_seq)

    
    
def fm_collate(batch):
    z0_list, z1_list, spk_list, z0_lens, spk_lens = zip(*batch)

    z0 = nn.utils.rnn.pad_sequence(z0_list, batch_first=True)
    z1 = nn.utils.rnn.pad_sequence(z1_list, batch_first=True)
    spk = nn.utils.rnn.pad_sequence(spk_list, batch_first=True)

    z0_lens = torch.tensor(z0_lens)
    spk_lens = torch.tensor(spk_lens)

    return z0, z1, spk, z0_lens, spk_lens


def make_mask(lengths, max_len):
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]

import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x, context, context_mask=None):
        # key_padding_mask: True = ignore
        if context_mask is not None:
            key_padding_mask = ~context_mask
        else:
            key_padding_mask = None

        out, _ = self.attn(
            query=x,
            key=context,
            value=context,
            key_padding_mask=key_padding_mask
        )
        return out

class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):

        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)

        emb = torch.exp(
            torch.arange(half_dim, device=x.device) * -emb
        )

        emb = x[:, None] * emb[None]

        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class ResidualBlock(nn.Module):

    def __init__(self, dim, cond_dim):

        super().__init__()

        self.fc = nn.Linear(dim, dim)
        self.act = nn.SiLU()

        self.film = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, cond):

        scale, shift = self.film(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)  # [B, 1, D]
        shift = shift.unsqueeze(1)  # [B, 1, D]


        h = self.fc(x)

        h = h * (1 + scale) + shift

        h = self.act(h)

        return x + h


# class AttentiveStatsPooling(nn.Module):
#     def __init__(self, in_dim, hidden_dim=128):
#         super().__init__()

#         self.attention = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, x):
#         """
#         x: (T, D)
#         """

#         # compute attention scores
#         attn = self.attention(x)          # (T, 1)
#         attn = torch.softmax(attn, dim=0) # (T, 1)

#         # weighted mean
#         mean = torch.sum(attn * x, dim=0)

#         # weighted std (optional but powerful)
#         var = torch.sum(attn * (x - mean)**2, dim=0)
#         std = torch.sqrt(var + 1e-5)

#         # concatenate mean + std
#         out = torch.cat([mean, std], dim=-1)  # (2D)

#         return out

def get_spk_embedding(x, mask):
    x = x * mask.unsqueeze(-1)  # zero padded
    return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)

class SpeakerFM(nn.Module):

    def __init__(self, emb_dim=1024, hidden=1024):

        super().__init__()

        self.norm = nn.LayerNorm(emb_dim)
        self.time_emb = SinusoidalPosEmb(128)

        # --- speaker global embedding ---
        self.spk_proj = nn.Sequential(
            nn.Linear(emb_dim, 512), nn.SiLU(),
            nn.Linear(512, 128)
        )

        # --- cross-attention ---
        self.cross_attn = CrossAttention(emb_dim, heads=4)

        cond_dim = 256
        self.cond_proj = nn.Linear(128 + 128, cond_dim)

        self.input = nn.Linear(emb_dim, hidden)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden, cond_dim)
            for _ in range(4)
        ])

        self.out = nn.Linear(hidden, emb_dim)

    def forward(self, z, t, spk_seq, z_mask=None, spk_mask=None):

        z = self.norm(z)

        # --- time embedding ---
        t_emb = self.time_emb(t)

        # --- global speaker embedding ---
        spk_mean = get_spk_embedding(spk_seq, spk_mask) #spk_seq.mean(dim=1)
        s_emb = self.spk_proj(spk_mean)

        # --- conditioning ---
        cond = torch.cat([t_emb, s_emb], dim=-1)
        cond = self.cond_proj(cond)

        # --- cross-attention ---
        z = self.cross_attn(z, spk_seq, spk_mask) # z +

        h = self.input(z)

        for block in self.blocks:
            h = block(h, cond)

        return self.out(h)


def train_fm(model, loader, optimizer, device, epochs=20, contin=False):
    start_epoch = 0
    best_loss = float("inf")
    if contin:
        checkpoint = torch.load("best_fm_knn_wavlm_emb_model_attn.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["loss"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler = torch.cuda.amp.GradScaler()
    beta_dist = Beta(2, 2)

    sigma = 0.03

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for z0, z1, spk, z0_lens, spk_lens in tqdm(loader):
            z0 = z0.to(device)
            z1 = z1.to(device)
            spk = spk.to(device)
            z0_lens = z0_lens.to(device)
            spk_lens = spk_lens.to(device)

            # --- masks ---
            z_mask = make_mask(z0_lens, z0.size(1))
            spk_mask = make_mask(spk_lens, spk.size(1))


            N = z0.shape[0]

            # sample time
            t = beta_dist.sample((N,)).to(device)

            with torch.cuda.amp.autocast(dtype=torch.float16):

                # interpolation
                zt = (1 - t[:, None, None]) * z0 + t[:, None, None] * z1

                # target perturbation
                noise_scale = sigma * torch.sqrt(t * (1 - t))
                zt = zt + noise_scale[:, None, None] * torch.randn_like(zt)

                target_v = z1 - z0

                pred_v = model(zt, t, spk, z_mask, spk_mask)

                #loss = F.mse_loss(pred_v.float(), target_v.float())
                pred_spk = get_spk_embedding(zt+(1-t).view(z0.shape[0],1,1)*pred_v, z_mask)
                tgt_spk  = get_spk_embedding(spk, spk_mask)
                loss_spk = 1 - F.cosine_similarity(pred_spk, tgt_spk, dim=-1)
                loss_spk = loss_spk.mean()
                loss_fm = ((pred_v - target_v) ** 2)
                loss_fm = loss_fm[z_mask].mean()
                loss = loss_fm + 0.1 * loss_spk

            optimizer.zero_grad(set_to_none=True)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch+1+start_epoch} | Loss: {avg_loss:.4f}")

         # Save best model every 5 epochs
        if (epoch + 1) % 1 == 0 and avg_loss < best_loss:

            best_loss = avg_loss

            torch.save(
                {
                    "epoch": epoch + 1 + start_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss
                },
                "best_fm_knn_wavlm_emb_model_attn.pt"
            )

            # print(f"Saved new best model at epoch {epoch+1} (loss {avg_loss:.4f})")

if __name__=='__main__':
    TGT_SPEAKERS = os.listdir("/scratch/aselitsk/LibriSpeech_wavlm/train-clean-100")

    dataset = FlowMatchingDataset(
        "/scratch/aselitsk/LibriSpeech_wavlm/train-clean-100",
        TGT_SPEAKERS
        )
    
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=28, #22
        collate_fn=fm_collate,
        pin_memory=True,
        persistent_workers=True
        )

    #print(dataset[0][0].shape, dataset[0][1].shape, dataset[0][2].shape, dataset[0][3], dataset[0][4])
    # for batch in loader:
    #     print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3], batch[4])
    #     break
    # torch.save(dataset.speaker_to_id, "speaker_to_id.pt")
    device = "cuda"

    model = SpeakerFM().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    train_fm(model, loader, optimizer, device, epochs=1000, contin=True)

