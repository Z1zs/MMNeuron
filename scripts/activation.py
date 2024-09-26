from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from mask_model import reset_dict, change_forward, load_mask
from types import MethodType
import os
from load_bench import load_pair
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"


def save_log(exp_token, save_path, save_name, resu):
    index = int(resu['vision_len'] // exp_token)
    save_name = f"{save_name}_{index}.log"
    os.makedirs(save_path, exist_ok=True)
    full_path = save_path + "/" + save_name
    if index > 0 and not os.path.exists(full_path):
        result = {}
        for k, v in resu.items():
            if torch.is_tensor(v):
                result[k] = resu[k].to("cpu")
        torch.save(result, full_path)
        print(f"{full_path} saved!")


def eval_load(processor, model, exp_token=1e+5):
    doms = ['ad', 'med', 'rs', 'com', 'doc']

    for dom in doms:
        res = reset_dict(_model, md_flag)
        change_forward(None, _model, res, md_flag)
        dt = load_pair(dom,"train",True)
        for i,tp  in enumerate(dt):
            prompt, image, answer, index = tp
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
            outputs = model.forward(**inputs)
            save_log(exp_token, f"data/{md_flag}_activation/{dom}", dom, res)

            if i >= 10000:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="specific models from ['llava','blip']")
    args = parser.parse_args()
    md_flag=args.model
    if md_flag=="llava":
        _processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        _model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
    else:
        _processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
        _model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")

    _model.to(device)
    eval_load(_processor, _model)
