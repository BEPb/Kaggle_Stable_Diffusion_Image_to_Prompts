"""
Python 3.10 программа для ..
Название файла subprocess.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2023-02-21
"""
import inspect
import importlib

from blip.models import blip
from clip_interrogator import clip_interrogator
import os
import sys
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import open_clip
from sentence_transformers import SentenceTransformer, models

comp_path = Path('.\\data')

# replace tokenizer path to prevent downloading
blip_path = inspect.getfile(blip)
fin = open(blip_path, "rt")
data = fin.read()
data = data.replace(
    "BertTokenizer.from_pretrained('bert-base-uncased')",
    "BertTokenizer.from_pretrained('/kaggle/input/clip-interrogator-models-x/bert-base-uncased')"
)
fin.close()
fin = open(blip_path, "wt")
fin.write(data)
fin.close()
# reload module
importlib.reload(blip)


# fix clip_interrogator bug
clip_interrogator_path = inspect.getfile(clip_interrogator.Interrogator)
fin = open(clip_interrogator_path, "rt")
data = fin.read()
data = data.replace(
    'open_clip.get_tokenizer(clip_model_name)',
    'open_clip.get_tokenizer(config.clip_model_name.split("/", 2)[0])'
)
fin.close()
fin = open(clip_interrogator_path, "wt")
fin.write(data)
fin.close()
# reload module
importlib.reload(clip_interrogator)

# set config
class CFG:
    device = "cuda"
    seed = 42
    embedding_length = 384
    # sentence_model_path = "/kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2"
    # blip_model_path = "/kaggle/input/clip-interrogator-models-x/model_large_caption.pth"
    ci_clip_model_name = "ViT-H-14/laion2b_s32b_b79k"
    clip_model_name = "ViT-H-14"
    # clip_model_path = "/kaggle/input/clip-interrogator-models-x/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
    # cache_path = "/kaggle/input/clip-interrogator-models-x"

# Load the Sample Submission
df_submission = pd.read_csv(comp_path / 'sample_submission.csv', index_col='imgId_eId')
df_submission.head()

# Build index from images
images = os.listdir(comp_path / 'images')
imgIds = [i.split('.')[0] for i in images]
eIds = list(range(CFG.embedding_length))
imgId_eId = [
    '_'.join(map(str, i)) for i in zip(
        np.repeat(imgIds, CFG.embedding_length),
        np.tile(range(CFG.embedding_length), len(imgIds))
    )
]
assert sorted(imgId_eId) == sorted(df_submission.index)

# Load the embedding model
st_model = SentenceTransformer()

################################## Prepare CLIP interrogator tool #####################################################

# Define CLIP interrogator config
model_config = clip_interrogator.Config(clip_model_name=CFG.ci_clip_model_name)
model_config.cache_path = CFG.cache_path

# Define BLIP model
configs_path = os.path.join(os.path.dirname(os.path.dirname(blip_path)), 'configs')
med_config = os.path.join(configs_path, 'med_config.json')
blip_model = blip.blip_decoder(
    pretrained=CFG.blip_model_path,
    image_size=model_config.blip_image_eval_size,
    vit=model_config.blip_model_type,
    med_config=med_config
)
blip_model.eval()
blip_model = blip_model.to(model_config.device)
model_config.blip_model = blip_model

# Define CLIP model
clip_model = open_clip.create_model(CFG.clip_model_name, precision='fp16' if model_config.device == 'cuda' else 'fp32')
open_clip.load_checkpoint(clip_model, CFG.clip_model_path)
clip_model.to(model_config.device).eval()
model_config.clip_model = clip_model
clip_preprocess = open_clip.image_transform(
    clip_model.visual.image_size,
    is_train = False,
    mean = getattr(clip_model.visual, 'image_mean', None),
    std = getattr(clip_model.visual, 'image_std', None),
)
model_config.clip_preprocess = clip_preprocess

# Create CLIP interrogator object
ci = clip_interrogator.Interrogator(model_config)

########################################## Define interrogate function #################################################

# Get labels embeddings
'''Original CLIP Interrogator uses image_features and text_embeds matrix multiplication to fine the similarity 
between the corresponding image and text label. But I found that using cosine similarity is much faster and the 
resulting score is almost identical. So take that into account.'''

cos = torch.nn.CosineSimilarity(dim=1)

mediums_features_array = torch.stack([torch.from_numpy(t) for t in ci.mediums.embeds]).to(ci.device)
movements_features_array = torch.stack([torch.from_numpy(t) for t in ci.movements.embeds]).to(ci.device)
flavors_features_array = torch.stack([torch.from_numpy(t) for t in ci.flavors.embeds]).to(ci.device)

# Create main interrogation function (It's modified version of the original interrogate_classic method.)
def interrogate(image: Image) -> str:
    caption = ci.generate_caption(image)
    image_features = ci.image_to_features(image)

    medium = [ci.mediums.labels[i] for i in cos(image_features, mediums_features_array).topk(1).indices][0]
    movement = [ci.movements.labels[i] for i in cos(image_features, movements_features_array).topk(1).indices][0]
    flaves = ", ".join([ci.flavors.labels[i] for i in cos(image_features, flavors_features_array).topk(3).indices])

    if caption.startswith(medium):
        prompt = f"{caption}, {movement}, {flaves}"
    else:
        prompt = f"{caption}, {medium}, {movement}, {flaves}"

    return clip_interrogator._truncate_to_fit(prompt, ci.tokenize)

# Extract promt from images
prompts = []

images_path = "../input/stable-diffusion-image-to-prompts/images/"
for image_name in images:
    img = Image.open(images_path + image_name).convert("RGB")

    generated = interrogate(img)

    prompts.append(generated)

# Check the result
def add_text_limiters(text: str) -> str:
    return " ".join([
        word + "\n" if i % 20 == 0 else word
        for i, word in enumerate(text.split(" "), start=1)
    ])

def plot_image(image: np.ndarray, original_prompt: str, generated_prompt: str) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.annotate(
        "Original prompt:\n" + add_text_limiters(original_prompt) + "\n\nGenerated prompt:\n" + add_text_limiters(generated_prompt),
        xy=(1.05, 0.5), xycoords='axes fraction', ha='left', va='center',
        fontsize=16, rotation=0, color="#00786b"
    )

# DO NOT FORGET TO COMMENT OUT THIS CELL DURING SUBMISSION
original_prompts_df = pd.read_csv("../data/prompts.csv")

for image_name, prompt in zip(images, prompts):
    img = Image.open(images_path + image_name).convert("RGB")
    original_prompt = original_prompts_df[
        original_prompts_df.imgId == image_name.split(".")[0]
    ].prompt.iloc[0]
    plot_image(img, original_prompt, prompt)

# Create a sample submission with a constant prompt prediction
# Encode prompts
prompt_embeddings = st_model.encode(prompts).flatten()

# Create submission DataFrame and save it as a .csv file
submission = pd.DataFrame(
    index=imgId_eId,
    data=prompt_embeddings,
    columns=['val']
).rename_axis('imgId_eId')

submission.to_csv('submission.csv')