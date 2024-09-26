import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, InstructBlipProcessor, \
    InstructBlipForConditionalGeneration
from load_bench import load_pair
import argparse


class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,) + output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))
        attn_output = self.block.self_attn.activations
        self.attn_mech_output_unembedded = self.unembed_matrix(self.norm(attn_output))
        attn_output += args[0]
        self.intermediate_res_unembedded = self.unembed_matrix(self.norm(attn_output))
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_output_unembedded = self.unembed_matrix(self.norm(mlp_output))
        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()

    def get_attn_activations(self):
        return self.block.self_attn.activations


class MMHelper:
    def __init__(self, model_flag):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_flag == "llava":
            self.tokenizer = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
            self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf",).to(self.device)
        else:
            self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b").to(self.device)
            self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

        for i, layer in enumerate(self.model.language_model.model.layers):
            self.model.language_model.model.layers[i] = BlockOutputWrapper(layer, self.model.language_model.lm_head,
                                                                           self.model.language_model.model.norm)

    def get_logits(self, text, img):
        inputs = self.tokenizer(text, img, return_tensors="pt").to(self.device)
        seq_len = inputs['input_ids'].shape[1]
        with torch.no_grad():
            self.model.forward(**inputs)
        return None, seq_len

    def set_add_attn_output(self, layer, add_output):
        self.model.language_model.model.layers[layer].attn_add_tensor(add_output)

    def get_attn_activations(self, layer):
        return self.model.language_model.model.layers[layer].get_attn_activations()

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, index=0):
        logi = decoded_activations[0][index]
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][index], dim=-1)
        entropy = torch.distributions.Categorical(logits=logi).entropy().to("cpu")

        values, indices = torch.topk(softmaxed, 10)
        values, indices = values.to("cpu"), indices.to("cpu")
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))

        _log = {"vocab": list(zip(tokens, probs_percent)), "entropy": entropy}
        return _log

    def decode_all_layers(self, text, img, print_attn_mech=True, print_intermediate_res=True, print_mlp=True,
                          print_block=True):
        _, seq_len = self.get_logits(text, img)

        model_log = []
        for i, layer in enumerate(self.model.language_model.model.layers):
            layer_log = []
            if print_attn_mech:
                self.print_decoded_activations(layer.attn_mech_output_unembedded, 0)
            if print_intermediate_res:
                self.print_decoded_activations(layer.intermediate_res_unembedded, 0)
            if print_mlp:
                self.print_decoded_activations(layer.mlp_output_unembedded, 0)
            if print_block:
                for j in range(layer.block_output_unembedded.shape[1]):
                    token_log = self.print_decoded_activations(layer.block_output_unembedded, j)
                    layer_log.append(token_log)
            model_log.append(layer_log)
        return {"log": model_log, "seq_len": seq_len}


if __name__ == "__main__":
    num = 100
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="specific models from ['llava','blip']")
    args = parser.parse_args()
    md_flag=args.model

    domains = ['ad', 'med', 'rs', 'com', 'doc']
    
    model = MMHelper(md_flag)
    for dom in domains:
        dt = load_pair(dom)
        data = []
        for j, tp in enumerate(dt):
            if (j + 1) % num == 0:
                torch.save(data, f"hidden_states/{md_flag}/{dom}/logit-len-{j // num}.pt")
                print(f"hidden_states/{md_flag}/{dom}/logit-len-{j // num}.pt saved!")
                data = []
            prompt, image, answer, idx = tp
            log = model.decode_all_layers(prompt, image,
                                          print_intermediate_res=False, print_mlp=False, print_block=True,
                                          print_attn_mech=False)
            data.append(log)
