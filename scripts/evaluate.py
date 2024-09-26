# Import necessary libraries
import os.path
from anls import anls_score
import pandas as pd
from transformers import pipeline
from load_bench import load_pair
import torch
import argparse


def judge(question_list, answer_list, prediction_list):
    # Define the model name to be used in the pipeline
    model_name = 'wayveai/Lingo-Judge'
    # Initialize the pipeline with the specified model, device, and other parameters
    pipe = pipeline("text-classification", model=model_name)

    res = []
    for question, answer, prediction in zip(question_list, answer_list, prediction_list):
        # Format the input string with the question, answer, and prediction
        input = f"[CLS]\nQuestion: {question}\nAnswer: {answer}\nStudent: {prediction}"

        # Pass the input through the pipeline to get the result
        result = pipe(input)

        # Print the result and score
        score = result[0]['score']
        res.append(score)
    return res


def eval_ad(filename, save_name):
    if not os.path.exists(filename):
        return 0

    index2qid = {}
    df = pd.read_parquet("benchs/ad/val/val.parquet").sample(frac=1, random_state=42)
    for i, (index, row) in enumerate(df.iterrows()):
        index2qid[i] = row['question_id']

    dt = load_pair("ad")
    index2que = {}
    for j, tp in enumerate(dt):
        # preprocessing
        prompt, image, answer, index = tp

        loc = prompt.find("Question: ")
        question = prompt[loc + len("Question: "):]
        index2que[index] = question

    question_list, answer_list, prediction_list, qid_list = [], [], [], []
    df = torch.load(filename)
    for row in df:
        question_list.append(index2que[row['index']])
        answer_list.append(row['ground_truth'])
        prediction_list.append(row['answer'])
        qid_list.append(index2qid[row['index']])

    if filename.find("llava") != 1:
        flag = "llava"
    else:
        flag = "blip"

    res=torch.load(f"results/{flag}/ad/{save_name}_score.pt")
    qids = set(index2qid.values())
    ev = {q: 0 for q in qids}
    for r, q in zip(res, qid_list):
        if r > 0.5:
            ev[q] = 1

    return sum(ev.values()) / len(ev)


def eval_med(filename):
    df = torch.load(filename)
    med_df = load_pair("med")
    options = ['A', 'B', 'C', 'D', 'E']

    count = 0
    for tmp in med_df:
        lb = tmp['ground_truth']
        if isinstance(tmp['answer'], list):
            pre = tmp['answer'][0]
        else:
            pre = tmp['answer']
        for i in range(len(pre)):
            if pre[i] in options and pre[i] == lb:
                count += 1

    return count / len(med_df)


def eval_doc(filename):
    df = torch.load(filename)

    sm = 0
    for tmp in df:
        lb = [t.lower() for t in tmp['ground_truth']]
        if isinstance(tmp['answer'], list):
            pre = tmp['answer'][0]
        else:
            pre = tmp['answer']
        pre = pre.lower()
        asc = anls_score(prediction=pre, gold_labels=lb, threshold=1.0)
        sm += asc

    return sm / len(df)


def eval_open_ended(filename):
    df = torch.load(filename)
    count = 0
    for tmp in df:
        lbs = tmp['ground_truth']
        lbs = [tmp['answer'].lower() for tmp in lbs if tmp['answer_confidence'] == "yes"]

        if isinstance(tmp['answer'], list):
            pre = tmp['answer'][0].lower()
        else:
            pre = tmp['answer'].lower()
        if pre in lbs:
            count += 1
    return count / len(df)


def eval_rs(filename):
    df = torch.load(filename)
    count = 0
    for tmp in df:
        lb = tmp['ground_truth']['answer']
        if isinstance(tmp['answer'], list):
            pre = tmp['answer'][0]
        else:
            pre = tmp['answer']
        if lb.lower() == pre.lower():
            count += 1
    return count / len(df)


def my_eval(dom, modules, model):
    suffix = '-'.join(modules)
    filename = f"/results/{model}/{dom}/{model}_{suffix}_ans.pt"

    if dom == "ad":
        score = eval_ad(filename, suffix)
    elif dom == "med":
        score = eval_med(filename)
    elif dom == "doc":
        score = eval_doc(filename)
    elif dom == "com":
        score = eval_open_ended(filename)
    elif dom == "rs":
        score = eval_rs(filename)
    else:
        print("Invalid domain name!")
        score = 0

    return score

domains = ['ad', 'med', 'rs', 'com', 'doc']
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="specific models from ['llava','blip']")
args = parser.parse_args()
md_flag=args.model
if md_flag == "llava":
    # LLAVA
    module_epochs = [[],['lang'],['vision'],['mmproj'],['lang','vision','mmproj']]
else:
    # InstructBLIP
    module_epochs = [[], ['lang'], ['encoder'], ['qformer'], ['encoder', 'qformer'], ['lang', 'encoder', 'qformer']]

for modules in module_epochs:
    for dom in domains:
        print(dom, modules, md_flag)
        score = my_eval(dom, modules, md_flag)
        print(score)
