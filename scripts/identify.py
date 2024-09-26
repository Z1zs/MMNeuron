import os.path
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

token_nums, logs = [], []
ds2idx = {"ad": 1, "com": 1, "med": 1, "rs": 1, "doc": 1}
pres = ['ad', 'med', 'rs', 'com', 'doc']

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="specific models from ['llava','blip']")
parser.add_argument("-d", "--module", help="specific modules from ['lang','vision','encoder','mmproj','qformer','query']")
args = parser.parse_args()
md_flag=args.model
mod=args.module


save_name = f"{md_flag}-{mod}"

domains = pres
for prefix in pres:
    data = torch.load(
        f'data/{md_flag}_activation/{prefix}/{prefix}_{ds2idx[prefix]}.log')
    token_nums.append(data[f'{mod}_len'])
    logs.append(data[f'{mod}_log'])

token_nums = torch.tensor(token_nums)
logs = torch.stack(logs, dim=-1)

num_layers, intermediate_size, lang_num = logs.size()


def activation():
    top_rate = 0.01
    filter_rate = 0.95
    activation_bar_ratio = 0.95
    activation_probs = logs / token_nums  # layer x inter x lang_num
    normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    largest = False

    if torch.isnan(entropy).sum():
        print(torch.isnan(entropy).sum())
        raise ValueError

    flattened_probs = activation_probs.flatten()
    top_prob_value = flattened_probs.kthvalue(round(len(flattened_probs) * filter_rate)).values.item()
    print(top_prob_value)
    # dismiss the neruon if no domain has an activation value over top 90%
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy[top_position == 0] = -torch.inf if largest else torch.inf

    flattened_entropy = entropy.flatten()
    top_entropy_value = round(len(flattened_entropy) * top_rate)
    _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)
    selected_probs = activation_probs[row_index, col_index]  # n x domain

    print(selected_probs.shape, torch.bincount(selected_probs.argmax(dim=-1)))
    selected_probs = selected_probs.transpose(0, 1)
    activation_bar = flattened_probs.kthvalue(round(len(flattened_probs) * activation_bar_ratio)).values.item()
    print((selected_probs > activation_bar).sum(dim=1).tolist())
    print(activation_bar)
    lang, indice = torch.where(selected_probs > activation_bar)

    merged_index = torch.stack((row_index, col_index), dim=-1)
    final_indice = []
    for _, index in enumerate(indice.split(torch.bincount(lang).tolist())):
        lang_index = [tuple(row.tolist()) for row in merged_index[index]]
        lang_index.sort()
        layer_index = [[] for _ in range(num_layers)]
        for l, h in lang_index:
            layer_index[l].append(h)
        for l, h in enumerate(layer_index):
            layer_index[l] = torch.tensor(h).long()
        final_indice.append(layer_index)
    save_root = f"data/{md_flag}_mask/"
    os.makedirs(save_root, exist_ok=True)
    torch.save(final_indice, f"{save_root}{save_name}")

    return final_indice


def get_insect(df, do1, do2):
    st1_df = df[df[do1] == 1]
    st2_df = df[df[do2] == 1]
    both_df = st1_df[st1_df[do2] == 1]
    return len(both_df) / len(st2_df)


def my_count(data):
    df = {}
    for i, do in enumerate(domains):  # i=0-5
        st = data[i]
        for j, layer in enumerate(st):  # j=0-23
            for index in layer:
                loc = (j, int(index))
                if loc in df:
                    df[loc][i] = 1
                else:
                    df[loc] = [0] * len(domains)
                    df[loc][i] = 1

    df = pd.DataFrame.from_dict(df, orient='index', columns=domains)
    print(len(df))
    print(df.sum(axis='columns').value_counts())
    print(df.sum(axis='index'))


data = activation()
my_count(data)
