import os
from mask_model import reset_dict, change_forward, load_mask
import torch
import argparse
from types import MethodType
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from load_bench import load_pair

stop = 1000
domains = ['ad', 'med', 'rs', 'com', 'doc']
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_template(prompt, model_flag):
    if model_flag == "llava":
        template = '''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:'''
        return template.format(prompt=prompt)
    elif model_flag == "blip":
        prompt = prompt.replace("<image>\n", "")
        context = "N/A"
        prompt = prompt.replace(f"Context: {context}", "")
        prompt = prompt.replace("\nAnswer the question using a single word or phrase.", " Short Answer:")
        prompt = "<image>" + prompt
        return prompt


def infer(dom, processor, model, result, save_name, model_flag):
    dt = load_pair(dom,"val",False)
    data = []
    for j, tp in enumerate(dt):
        if j > stop:
            break
        # preprocessing
        prompt, image, answer, index = tp
        prompt = get_template(prompt, model_flag)
        print(prompt)
        # generate
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.9,
                max_length=512,
                min_length=1,
            )
        # get generated text
        if model_flag == "blip":
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        else:
            out_wo_prompt = outputs[:, inputs.input_ids.shape[-1]:]
            generated_text = processor.batch_decode(out_wo_prompt, skip_special_tokens=True)
        print(generated_text)
        # append log
        log = dict(ground_truth=answer, index=index, answer=generated_text)
        data.append(log)

        # clear cache
        del outputs, inputs
        torch.cuda.empty_cache()

    # save log
    for k, v in result.items():
        result[k] = v.to("cpu")
    os.makedirs(f"results/{model_flag}/{dom}/", exist_ok=True)
    torch.save(data, f"results/{model_flag}/{dom}/{save_name}_ans.pt")
    print(f"results/{model_flag}/{dom}/{save_name}_ans.pt saved!")

    # clear cache
    del result, data
    torch.cuda.empty_cache()


if __name__ == "__main__":
    
    #load model
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="specific model from ['llava','blip']")
    args = parser.parse_args()
    md_flag = args.model
    if md_flag == "llava":
        # LLAVA
        _processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
        _model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
        module_epochs = [[],['lang'],['vision'],['mmproj'],['lang','vision','mmproj']]
    else:
        # InstructBLIP
        _model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        _processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        module_epochs = [[], ['lang'], ['encoder'], ['qformer'], ['encoder', 'qformer'], ['lang', 'encoder', 'qformer']]

    _model.to(device)
    _model.eval()
    print("model loaded!")

    # generate reply
    for modules in module_epochs:
        suffix = '-'.join(modules)
        llava_masks = load_mask(modules, md_flag)

        for i, d in enumerate(domains):
            mask = llava_masks[i]
            res = reset_dict(_model, md_flag)
            change_forward(mask, _model, res, md_flag)
            infer(d, _processor, _model, res, f"{md_flag}_{suffix}", md_flag)
