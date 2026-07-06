from pathlib import Path
from collections import defaultdict
import random

import csv
import soundfile as sf

import os

seed = 42
# Python RNG
random.seed(seed)


# files_by_speaker = defaultdict(list)

# data_root = Path("c://Users/laure/Downloads/archive_Libri/LibriSpeech/test-clean")

# for f in data_root.rglob(f"*.flac"):
#     speaker = f.parent.parent.name   # speaker/chapter/file
#     files_by_speaker[speaker].append(f)

# all_files = list(data_root.rglob(f"*.flac"))

# speakers = list(files_by_speaker.keys())

# src_files = all_files # random.sample(all_files, k)

# pairs = []

# for src in src_files:
#     src_spk = src.parent.parent.name

#     tgt_spk = random.choice([s for s in speakers if s != src_spk])
#     tgt = random.choice(files_by_speaker[tgt_spk])

#     pairs.append((src, tgt))

#print(len(pairs))

csv_path = "c://Users/laure/Downloads/archive_Libri/librispeech_test_pairs_text_w.csv"

# with open(csv_path, "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow([
#         "source_file",
#         "target_file",
#         "target_duration_sec"
#     ])

#     for src, tgt in pairs:
#         info = sf.info(str(tgt))
#         duration = info.frames / info.samplerate

#         writer.writerow([
#             src.relative_to(data_root).as_posix(), #src.name, #str(src),
#             tgt.relative_to(data_root).as_posix(), #tgt.name, #str(tgt),
#             round(duration, 3)
#         ])

# print(f"Saved {len(pairs)} pairs to {csv_path}")

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV
df = pd.read_csv(csv_path)
#df = df[df.index%2==0]

# # Function to read transcription
# def get_text(row):

#     speaker, chapter, file_name = row["source_file"].split('/')
#     #chapter = row["chapter_id"]
#     #file_name = row["file_name"]

#     base = file_name.replace(".flac", "")

#     txt_file = f"{speaker}-{chapter}.trans.txt"

#     txt_path = os.path.join(
#         data_root,
#         str(speaker),
#         str(chapter),
#         txt_file
#     )

#     with open(txt_path, "r") as f:
#         for line in f:
#             if line.startswith(base):
#                 return line.split(" ", 1)[1].strip()

#     return None

# df["text"] = df.apply(get_text, axis=1)

# df.to_csv("c://Users/laure/Downloads/archive_Libri/librispeech_test_pairs_text.csv", index=False)
# #print(df.head())

# Basic information
print(df.head())
print(df[['wer_our_SB_5', 'wer_our_SB']].describe()) #["target_duration_sec"]

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(df["target_duration_sec"], bins=30)
plt.xlabel("Target duration (seconds)")
plt.ylabel("Number of files")
plt.title("Distribution of target utterance durations")
plt.grid(alpha=0.3)

plt.show()

