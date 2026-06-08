## Remove all names and mentions from the data, using the mapping in author_anon_map

import os
import re
import json
import pandas as pd
import unicodedata

# Load the author_anon_map
with open("VideosComments/youtube/author_anon_map.json", "r") as f:
    author_anon_map = json.load(f)
author_anon_map = dict(sorted(author_anon_map.items(), key=lambda item: len(item[0]), reverse=True))

INPUT_DIRECTORY = "VideosComments/youtube/comments/{newspaper}"
OUTPUT_DIRECTORY = "VideosComments/youtube/comments_anonymized/{newspaper}"


newspapers = ["corriere_della_sera", "il_gazzettino", "ilmessaggero", "lastampa", "repubblica"]


def anonymize_text(text):
    if pd.isna(text):
        return text

    # Replace known usernames
    for author_name, anon_name in author_anon_map.items():
        text = text.replace(author_name, f"@{anon_name}")

    # Replace only remaining unknown mentions (should not happen but just to be safe)
    text = re.sub(r'@(?!author_\d+\b)[^\s]+', '@user', text)

    return text

def preprocess_text(text):
    if pd.isna(text):
        return text

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove zero-width/invisible chars
    text = re.sub(r'[\u200b-\u200d\uFEFF]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text

# Loop through all newspapers and video files
for newspaper in newspapers:
    print(f"Processing newspaper: {newspaper}")
    input_dir = INPUT_DIRECTORY.format(newspaper=newspaper)
    output_dir = OUTPUT_DIRECTORY.format(newspaper=newspaper)

    os.makedirs(output_dir, exist_ok=True)

    # Loop through all video files in the input directory
    for idx_video, video_file in enumerate(os.listdir(input_dir)):
        if video_file.endswith(".csv"):
            video_id = video_file[:-4]  # Remove the .csv extension
            input_file_path = os.path.join(input_dir, video_file)
            output_file_path = os.path.join(output_dir, video_file)

            df = pd.read_csv(input_file_path)

            # Anonymize & cleanup
            df["author"] = df["author"].apply(lambda x: author_anon_map.get(x, x))
            df["text"] = df["text"].apply(anonymize_text) 
            df["text"] = df["text"].apply(preprocess_text)

            df.to_csv(output_file_path, index=False)
            print(f"Processed video {idx_video + 1}/{len(os.listdir(input_dir))}: {video_id}")
