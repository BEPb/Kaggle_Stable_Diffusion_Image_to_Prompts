"""
Python 3.10 программа для распознования изображения и написания краткого коммента
Название 00.lb-0-45836-blip-clip.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2023-02-25
"""
# подключаем библиотеки
import inspect
'''Модуль inspect предоставляет несколько полезных функций, помогающих получить информацию о живых
объектах, таких как модули, классы, методы, функции, обратные трассировки, объекты фреймов и объекты кода.
Например, он может помочь вам изучить содержимое класса, получить исходный код метода, извлечь и отформатировать
список аргументов для функции или получить всю информацию, необходимую для отображения подробной обратной
трассировки.
'''
import importlib
'''Цель пакета importlib тройная.
Один из них — предоставить реализацию инструкции import(и, соответственно, функции __import__()) в исходном коде
Python. Это обеспечивает реализацию, import переносимую на любой интерпретатор Python. Это также обеспечивает
реализацию, которую легче понять, чем реализация, реализованная на языке программирования, отличном от Python.
Во-вторых, компоненты, которые необходимо реализовать, import представлены в этом пакете, что упрощает пользователям
создание собственных настраиваемых объектов (обычно известных как средство импорта ) для участия в процессе импорта.
В-третьих, пакет содержит модули, предоставляющие дополнительные функции для управления аспектами пакетов Python:
      importlib.metadata представляет доступ к метаданным из сторонних дистрибутивов.
      importlib.resources предоставляет подпрограммы для доступа к не кодовым «ресурсам» из пакетов Python.
'''
from blip.models import blip
'''BLIP — это модель, способная выполнять различные мультимодальные задачи, в том числе
1. Визуальный ответ на вопрос
2. Поиск изображения и текста (сопоставление изображения и текста)
3. Подпись к изображению
'''
from clip_interrogator import clip_interrogator
'''Хотите выяснить, что может быть хорошей подсказкой для создания новых изображений, подобных существующему?
CLIP Interrogator готов дать вам ответы!
https://huggingface.co/spaces/pharma/CLIP-Interrogator
'''
import os
'''Модуль os предоставляет множество функций для работы с операционной системой'''
from PIL import Image
'''PIL, известная как библиотека Python Imaging Library, может быть использована для работы с изображениями 
достаточно легким способом '''
from pathlib import Path
'''Этот модуль предлагает классы, представляющие пути файловой системы с семантикой, подходящей для разных 
операционных систем '''
# import matplotlib.pyplot as plt
import numpy as np
'''библиотека языка Python, добавляющая поддержку больших многомерных массивов и матриц, вместе с большой библиотекой
 высокоуровневых (и очень быстрых) математических функций для операций с этими массивами.'''
import pandas as pd  # библиотека анализа данных
import torch
'''PyTorch — это пакет Python, который предоставляет две функции высокого уровня:
Тензорные вычисления (например, NumPy) с сильным ускорением графического процессора
Глубокие нейронные сети, построенные на ленточной системе автоградации
'''
import open_clip  # (предварительное обучение контрастному языку и изображению) с открытым исходным кодом.
from sentence_transformers import SentenceTransformer, models
'''Эта структура предоставляет простой метод для вычисления плотных векторных представлений предложений, 
абзацев и изображений. Модели основаны на трансформаторных сетях, таких как BERT / RoBERTa / XLM-RoBERTa и т. д., 
и обеспечивают самые современные характеристики в различных задачах. Текст встраивается в векторное пространство 
таким образом, что похожий текст близок и может быть эффективно найден с помощью косинусного сходства.   '''

comp_path = Path('.\\data')

# replace tokenizer path to prevent downloading
blip_path = inspect.getfile(blip)
fin = open(blip_path, "rt")
data = fin.read()
data = data.replace(
    "BertTokenizer.from_pretrained('bert-base-uncased')",
    "BertTokenizer.from_pretrained('./clip-interrogator-models-x/bert-base-uncased')"
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
    sentence_model_path = "./clip-interrogator-models-X/all-MiniLM-L6-v2"
    blip_model_path = ".\clip-interrogator-models-x\model_large_caption.pth"
    ci_clip_model_name = "ViT-H-14/laion2b_s32b_b79k"
    clip_model_name = "ViT-H-14"
    clip_model_path = "./clip-interrogator-models-x/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
    cache_path = "./clip-interrogator-models-x"

# Load the Sample Submission
df_submission = pd.read_csv(comp_path / 'sample_submission.csv', index_col='imgId_eId')
# df_submission.head()

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
st_model = SentenceTransformer(CFG.sentence_model_path)

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
'''Оригинальный CLIP Interrogator использует умножение матриц image_features и text_embeds для уточнения подобия
между соответствующим изображением и текстовой меткой. Но я обнаружил, что использование сходства косинусов намного быстрее, и
результирующий балл почти идентичен. Так что примите это во внимание.'''

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

images_path = "./data/images/"
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

# def plot_image(image: np.ndarray, original_prompt: str, generated_prompt: str) -> None:
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     plt.annotate(
#         "Original prompt:\n" + add_text_limiters(original_prompt) + "\n\nGenerated prompt:\n" + add_text_limiters(generated_prompt),
#         xy=(1.05, 0.5), xycoords='axes fraction', ha='left', va='center',
#         fontsize=16, rotation=0, color="#00786b"
#     )

# DO NOT FORGET TO COMMENT OUT THIS CELL DURING SUBMISSION
# original_prompts_df = pd.read_csv("./data/prompts.csv")
#
# for image_name, prompt in zip(images, prompts):
#     img = Image.open(images_path + image_name).convert("RGB")
#     original_prompt = original_prompts_df[
#         original_prompts_df.imgId == image_name.split(".")[0]
#     ].prompt.iloc[0]
#     plot_image(img, original_prompt, prompt)

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