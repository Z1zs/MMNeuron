import os
from datasets import load_dataset
import numpy as np
import torch
import json
import pandas as pd
import random
from PIL import Image
from types import MethodType
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlamaForCausalLM


def load_ad(split_flag,simple_flag):
    image_root = f"benchs/ad/{split_flag}"

    df = pd.read_parquet(f"benchs/ad/{split_flag}/{split_flag}.parquet").sample(frac=1, random_state=42)
    # img_set = set()
    for i, (index, row) in enumerate(df.iterrows()):
        img_names = row['images']
        question = row['question']
        answer = row['answer']
        # if img_names[0] in img_set:
        #     continue
        # img_set.add(img_names[0])
        images = []
        for img_name in img_names:
            if os.path.exists(f"{image_root}/{img_name}"):
                img = Image.open(f"{image_root}/{img_name}")
                images.append(img)
            else:
                print(img_name + " not found!")
                continue
        # convert images into one image
        img_array = np.concatenate([np.array(img) for img in images], axis=1)
        image = Image.fromarray(img_array)
        prompt = '<image>' + '\n' + f"Role: You are an advanced AI assistant installed on the Ego vehicle, equipped with conversational analysis capabilities for discussing autonomous driving scenarios. The perspective presented is from the point-of-view of the Ego vehicle, where the camera is mounted. It's important to note that the Ego vehicle itself is not visible in the images provided. Question: {question}"
        if simple_flag:
            prompt=f'<image>\n{question}'
        yield prompt, image, answer, i


def load_med(split_flag,simple_flag):
    img_set = set()
    image_root = ['benchs/med/figures',
                  'benchs/med/images']

    df1 = pd.read_csv(f"benchs/med/{split_flag}.csv")
    df2 = pd.read_csv(f"benchs/med/{split_flag}.csv")
    df = pd.concat([df1, df2], axis="index")
    df = df.sample(frac=1, random_state=42)

    for index, row in df.iterrows():
        img_name = row['Figure_path']
        answer = row['Answer_label']
        if answer not in ['A', 'B', 'C', 'D']:
            answer = row['Answer']
        if img_name in img_set:
            continue
        img_set.add(img_name)
        if os.path.exists(f"{image_root[0]}/{img_name}"):
            image = Image.open(f"{image_root[0]}/{img_name}")
        elif os.path.exists(f"{image_root[1]}/{img_name}"):
            image = Image.open(f"{image_root[1]}/{img_name}")
        else:
            print(img_name + "not found!")
            continue

        question = row['Question']
        context = "N/A"
        choice_txt = [row[f'Choice {i}'] for i in ['A', 'B', 'C', 'D']]
        prompt = ('<image>' + '\n' + f"Question: {question}\nContext: {context}\nOptions: {choice_txt}\n"
                  + "\nAnswer with the option's letter from the given choices directly.")
        if simple_flag:
            prompt=f'<image>\n{question}'
        yield prompt, image, answer, index


def load_rs(split_flag,simple_flag):
    img_root = "benchs/rs/Data"
    with open(f"benchs/rs/USGS_split_{split_flag}_questions.json", "r") as read_file:
        data = json.load(read_file)
    data = data['questions']

    imgs = [str(tmp['img_id']) for tmp in data if tmp['active']]
    ques = [tmp['question'] for tmp in data if tmp['active']]
    ans_ids = [tmp['answers_ids'] for tmp in data if tmp['active']]

    with open(f"benchs/rs/USGS_split_{split_flag}_answers.json", "r") as read_file:
        ans_data = json.load(read_file)
    ans_data = ans_data['answers']
    ans = [ans_data[tmp[0]] for tmp in ans_ids]
    img_set = set()

    zipped = list(zip(imgs, ques, ans))
    random.Random(4).shuffle(zipped)
    imgs, ques, ans = zip(*zipped)

    for i, tmp in enumerate(zip(imgs, ques, ans)):
        img_name = tmp[0]
        question = f"{tmp[1]}"
        answer = tmp[2]
        if img_name in img_set:
            continue
        img_set.add(img_name)
        image_path = f"{img_root}/{img_name}.png"

        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            continue

        prompt = '<image>' + '\n' + f"Question: {question}" + "\nAnswer the question using a single word or phrase."

        if simple_flag:
            prompt=f'<image>\n{question}'
        yield prompt, image, answer, i

def load_com(split_flag,simple_flag):
    img_root = f"benchs/vqav2/{split_flag}2014"
    with open(f"benchs/vqav2/v2_OpenEnded_mscoco_{split_flag}2014_questions.json",
              "r") as read_file:
        data = json.load(read_file)
    with open(f"/benchs/vqav2/v2_mscoco_{split_flag}2014_ansdict.json", "r") as read_file:
        qid2ans = json.load(read_file)

    data = data['questions']
    imgs = [str(tmp['image_id']) for tmp in data]
    ques = [tmp['question'] for tmp in data]
    que_ids = [tmp['question_id'] for tmp in data]
    ans = [qid2ans[str(qid)] for qid in que_ids]
    img_set = set()

    zipped = list(zip(imgs, ques, ans))
    random.Random(4).shuffle(zipped)
    imgs, ques, ans = zip(*zipped)

    for i, tmp in enumerate(zip(imgs, ques, ans)):
        img_id = tmp[0]
        img_name = 'COCO_' + f"{split_flag}2014" + '_' + str(img_id).zfill(12) + '.jpg'
        question = f"{tmp[1]}"
        answer = tmp[2]
        if img_id in img_set:
            continue
        img_set.add(img_id)
        image_path = f"{img_root}/{img_name}"

        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            continue

        prompt = '<image>' + '\n' + f"Question: {question}" + "\nAnswer the question using a single word or phrase."
        if simple_flag:
            prompt=f'<image>\n{question}'
        yield prompt, image, answer, i


def load_doc(split_flag,simple_flag):
    split_flag="train" if split_flag=="train" else "validation"
    dataset = load_dataset("pixparse/docvqa-single-page-questions", split=f"{split_flag}",
                           cache_dir="benchs/doc").shuffle(seed=42)
    img_set = set()
    for i, tmp in enumerate(dataset):
        question = f"{tmp['question']}"
        img_name = tmp['other_metadata']['image']
        if img_name in img_set:
            continue
        img_set.add(img_name)
        image = tmp['image']
        answer = tmp['answers']
        if image is None:
            continue

        prompt = '<image>' + '\n' + f"Question: {question}" + "\nAnswer the question using a single word or phrase."
        if simple_flag:
            prompt=f'<image>\n{question}'
        yield prompt, image, answer, i


def load_pair(dom,split_flag="train",simple_flag=False):
    if dom == "ad":
        return load_ad(split_flag,simple_flag)

    elif dom == "med":
        return load_med(split_flag,simple_flag)

    elif dom == "rs":
        return load_rs(split_flag,simple_flag)

    elif dom == "com":
        return load_com(split_flag,simple_flag)
    
    elif dom == "doc":
        return load_doc(split_flag,simple_flag)
    
    else:
        return None, None, None, None


if __name__ == "__main__":
    dom = "doc"
    index = 12
    df = load_pair(dom)
    for j, tp in enumerate(df):
        if j == index:
            prompt, image, answer, idx = tp
            image.save(f"test.png")
            break
