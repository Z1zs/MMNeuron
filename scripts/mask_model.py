import torch
from types import MethodType
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from load_bench import load_pair

device = "cuda" if torch.cuda.is_available() else "cpu"


def llava_factory(mask, idx, flag, result):
    def llama_forward(self, x):
        x1 = self.act_fn(self.gate_proj(x))
        ##################################
        # mask
        if mask is not None:
            x1.index_fill_(2, mask, 0)
        # record
        result['lang_log'][idx, :] += (x1 > 0).sum(dim=(0, 1))
        result['lang_value'][idx, :] += x1.sum(dim=(0, 1))
        if idx == 1:
            result['lang_len'] += (x1.shape[0] * x1.shape[1])#token_num
        ###################################
        x2 = self.up_proj(x)
        return self.down_proj(x1 * x2)

    def mmproj_forward(self, hidden_states):
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        #######################
        # mask
        if mask is not None:
            hidden_states.index_fill_(2, mask, 0)
        # record
        result['mmproj_len'] += (hidden_states.shape[0] * hidden_states.shape[1])
        result['mmproj_log'][0, :] += (hidden_states > 0).sum(dim=(0, 1))
        result['mmproj_value'][0, :] += hidden_states.sum(dim=(0, 1))
        #########################
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def clip_forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        ############################
        # mask
        if mask is not None:
            hidden_states.index_fill_(2, mask, 0)
        # record
        if idx == 1:
            result['vision_len'] += (hidden_states.shape[0] * hidden_states.shape[1])
        result['vision_log'][idx, :] += (hidden_states > 0).sum(dim=(0, 1))
        result['vision_value'][idx, :] += hidden_states.sum(dim=(0, 1))
        #############################
        hidden_states = self.fc2(hidden_states)
        return hidden_states

    if flag == "llama":
        return llama_forward
    elif flag == "clip":
        return clip_forward
    if flag == "mmproj":
        return mmproj_forward


def blip_factory(mask, idx, flag, result):
    def llama_forward(self, x):
        x1 = self.act_fn(self.gate_proj(x))
        ##################################
        # mask
        if mask is not None:#domain, layer, indices
            x1.index_fill_(2, mask, 0)
        # record
        if x1.shape[1] > 1:  # avoid sampling
            result['lang_log'][idx, :] += (x1 > 0).sum(dim=(0, 1))
            result['lang_value'][idx, :] += x1.sum(dim=(0, 1))
            if idx == 1:
                result['lang_len'] += (x1.shape[0] * x1.shape[1])
        ###################################
        x2 = self.up_proj(x)
        return self.down_proj(x1 * x2)

    def qformer_forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        ############################
        # mask
        if mask is not None:
            hidden_states.index_fill_(2, mask, 0)
        # record
        if idx == 1:
            result['qformer_len'] += (hidden_states.shape[0] * hidden_states.shape[1])
        result['qformer_log'][idx, :] += (hidden_states > 0).sum(dim=(0, 1))
        result['qformer_value'][idx, :] += hidden_states.sum(dim=(0, 1))
        return hidden_states

    def query_forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        ############################
        # mask
        if mask is not None:
            hidden_states.index_fill_(2, mask, 0)
        # record
        if idx == 1:
            result['query_len'] += (hidden_states.shape[0] * hidden_states.shape[1])
        result['query_log'][idx, :] += (hidden_states > 0).sum(dim=(0, 1))
        result['query_value'][idx, :] += hidden_states.sum(dim=(0, 1))
        #############################
        return hidden_states

    def encoder_forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        ############################
        # mask
        if mask is not None:
            hidden_states.index_fill_(2, mask, 0)
        # record
        if idx == 1:
            result['encoder_len'] += (hidden_states.shape[0] * hidden_states.shape[1])
        result['encoder_log'][idx, :] += (hidden_states > 0).sum(dim=(0, 1))
        result['encoder_value'][idx, :] += hidden_states.sum(dim=(0, 1))
        #############################
        hidden_states = self.fc2(hidden_states)
        return hidden_states

    if flag == "llama":
        return llama_forward
    elif flag == "encoder":
        return encoder_forward
    elif flag == "qformer":
        return qformer_forward
    elif flag == "query":
        return query_forward


def reset_dict(_model, md_flag):
    if md_flag == "llava":
        result = {}

        language_num_layers = _model.language_model.config.num_hidden_layers
        language_intermediate_size = _model.language_model.config.intermediate_size
        result['lang_log'] = torch.zeros(language_num_layers, language_intermediate_size, dtype=torch.int32).to(device)
        result['lang_value'] = torch.zeros(language_num_layers, language_intermediate_size, dtype=torch.float64).to(
            device)

        vision_num_layers = _model.vision_tower.config.num_hidden_layers
        vision_intermediate_size = _model.vision_tower.config.intermediate_size
        result['vision_log'] = torch.zeros(vision_num_layers, vision_intermediate_size, dtype=torch.int32).to(device)
        result['vision_value'] = torch.zeros(vision_num_layers, vision_intermediate_size, dtype=torch.float64).to(
            device)

        mmproj_intermediate_size = _model.multi_modal_projector.linear_1.out_features
        result['mmproj_log'] = torch.zeros(1, mmproj_intermediate_size, dtype=torch.int32).to(device)
        result['mmproj_value'] = torch.zeros(1, mmproj_intermediate_size, dtype=torch.float64).to(device)

        result['mmproj_len'] = torch.tensor(0)
        result['lang_len'] = torch.tensor(0)
        result['vision_len'] = torch.tensor(0)

        return result

    elif md_flag == 'blip':
        result = {}

        language_num_layers = _model.language_model.config.num_hidden_layers
        language_intermediate_size = _model.language_model.config.intermediate_size
        result['lang_log'] = torch.zeros(language_num_layers, language_intermediate_size, dtype=torch.int32).to(device)
        result['lang_value'] = torch.zeros(language_num_layers, language_intermediate_size, dtype=torch.float64).to(
            device)

        vision_num_layers = _model.vision_model.config.num_hidden_layers
        vision_intermediate_size = _model.vision_model.config.intermediate_size
        result['encoder_log'] = torch.zeros(vision_num_layers, vision_intermediate_size, dtype=torch.int32).to(device)
        result['encoder_value'] = torch.zeros(vision_num_layers, vision_intermediate_size, dtype=torch.float64).to(
            device)

        qformer_num_layers = _model.qformer.config.num_hidden_layers
        qformer_intermediate_size = _model.qformer.config.intermediate_size
        result['qformer_log'] = torch.zeros(qformer_num_layers, qformer_intermediate_size, dtype=torch.int32).to(device)
        result['qformer_value'] = torch.zeros(qformer_num_layers, qformer_intermediate_size, dtype=torch.float64).to(
            device)
        result['query_log'] = torch.zeros(qformer_num_layers, qformer_intermediate_size, dtype=torch.int32).to(device)
        result['query_value'] = torch.zeros(qformer_num_layers, qformer_intermediate_size, dtype=torch.float64).to(
            device)

        result['qformer_len'] = torch.tensor(0)
        result['query_len'] = torch.tensor(0)
        result['lang_len'] = torch.tensor(0)
        result['encoder_len'] = torch.tensor(0)

        return result
    else:
        print("Please check your flag to reset the result dict!")
    return {}


def llava_change_forward(masks, _model, result):
    language_num_layers = _model.language_model.config.num_hidden_layers
    vision_num_layers = _model.vision_tower.config.num_hidden_layers
    # log
    for i in range(language_num_layers):
        obj = _model.language_model.model.layers[i].mlp
        obj.forward = MethodType(llava_factory(None, i, "llama", result), obj)
    for i in range(vision_num_layers):
        obj = _model.vision_tower.vision_model.encoder.layers[i].mlp
        obj.forward = MethodType(llava_factory(None, i, "clip", result), obj)
    obj = _model.multi_modal_projector
    obj.forward = MethodType(llava_factory(None, 0, "mmproj", result), obj)
    # mask
    if masks is None:
        return None
    if "lang" in masks:
        for i, layer_mask in enumerate(masks['lang']):
            obj = _model.language_model.model.layers[i].mlp
            obj.forward = MethodType(llava_factory(layer_mask.to(device), i, "llama", result), obj)
    if "vision" in masks:
        # we require that vision and mmporj to be present as a whole
        for i, layer_mask in enumerate(masks['vision']):
            obj = _model.vision_tower.vision_model.encoder.layers[i].mlp
            obj.forward = MethodType(llava_factory(layer_mask.to(device), i, "clip", result), obj)
        obj = _model.multi_modal_projector
        obj.forward = MethodType(llava_factory(masks['mmproj'][0].to(device), 0, "mmproj", result), obj)
    if "mmprj" in masks:
        obj = _model.multi_modal_projector
        obj.forward = MethodType(llava_factory(masks['mmproj'][0].to(device), 0, "mmproj", result), obj)


def blip_change_forward(masks, _model, result):
    language_num_layers = _model.language_model.config.num_hidden_layers
    vision_num_layers = _model.vision_model.config.num_hidden_layers
    qformer_num_layers = _model.qformer.config.num_hidden_layers
    # log
    for i in range(language_num_layers):
        obj = _model.language_model.model.layers[i].mlp
        obj.forward = MethodType(blip_factory(None, i, "llama", result), obj)
    for i in range(vision_num_layers):
        obj = _model.vision_model.encoder.layers[i].mlp
        obj.forward = MethodType(blip_factory(None, i, "encoder", result), obj)
    for i in range(qformer_num_layers):
        obj = _model.qformer.encoder.layer[i].intermediate
        obj.forward = MethodType(blip_factory(None, i, "qformer", result), obj)

        obj = _model.qformer.encoder.layer[i].intermediate_query
        obj.forward = MethodType(blip_factory(None, i, "query", result), obj)
    # mask
    if masks is None:
        return None
    if "lang" in masks:
        for i, layer_mask in enumerate(masks['lang']):
            obj = _model.language_model.model.layers[i].mlp
            obj.forward = MethodType(blip_factory(layer_mask.to(device), i, "llama", result), obj)
    if "encoder" in masks:
        for i, layer_mask in enumerate(masks['encoder']):
            obj = _model.vision_model.encoder.layers[i].mlp
            obj.forward = MethodType(blip_factory(layer_mask.to(device), i, "encoder", result), obj)
    if 'qformer' in masks:
        # we require that qformer and query to be present as a whole
        for i, (qformer_mask, query_mask) in enumerate(zip(masks['qformer'], masks['query'])):
            obj = _model.qformer.encoder.layer[i].intermediate
            obj.forward = MethodType(blip_factory(qformer_mask.to(device), i, "qformer", result), obj)

            obj = _model.qformer.encoder.layer[i].intermediate_query
            obj.forward = MethodType(blip_factory(query_mask.to(device), i, "query", result), obj)


def change_forward(masks, _model, result, md_flag):
    if md_flag == "llava":
        llava_change_forward(masks, _model, result)
    elif md_flag == "blip":
        blip_change_forward(masks, _model, result)
    else:
        print("Nothing has been changed since there are no suitable model.")


def todevice(masks):
    for mask in masks:
        if mask is not None:
            mask.to(device)
    return masks


def load_llava_mask(flag):
    lang_mask = torch.load("/hpc2hdd/home/yxu409/jiahaohuo/VQA/data/instruct_mask/llava-v1.6-lang")
    vision_mask = torch.load("/hpc2hdd/home/yxu409/jiahaohuo/VQA/data/instruct_mask/llava-v1.6-vision")
    mmproj_mask = torch.load("/hpc2hdd/home/yxu409/jiahaohuo/VQA/data/instruct_mask/llava-v1.6-mmproj")
    ms = []
    for la, v, m in zip(lang_mask, vision_mask, mmproj_mask):
        tdc = {}
        if "lang" in flag:
            tdc['lang'] = todevice(la)
        if 'vision' in flag:
            tdc['vision'] = todevice(v)
            tdc['mmproj'] = todevice(m)
        if 'mmproj' in flag:
            tdc['mmproj'] = todevice(m)
        ms.append(tdc)
    return ms


def load_blip_mask(flag):
    lang_mask = torch.load("/hpc2hdd/home/yxu409/jiahaohuo/VQA/data/blip_mask/blip-lang")
    encoder_mask = torch.load("/hpc2hdd/home/yxu409/jiahaohuo/VQA/data/blip_mask/blip-encoder")
    qformer_mask = torch.load("/hpc2hdd/home/yxu409/jiahaohuo/VQA/data/blip_mask/blip-qformer")
    query_mask = torch.load("/hpc2hdd/home/yxu409/jiahaohuo/VQA/data/blip_mask/blip-query")

    ms = []
    for la, e, qf, qu in zip(lang_mask, encoder_mask, qformer_mask, query_mask):
        tdc = {}
        if "lang" in flag:
            tdc['lang'] = todevice(la)
        if 'encoder' in flag:
            tdc['encoder'] = todevice(e)
        if 'qformer' in flag:
            tdc['qformer'] = todevice(qf)
            tdc['query'] = todevice(qu)
        ms.append(tdc)
    return ms


def load_mask(flag, md_flg):
    if md_flg == "llava":
        return load_llava_mask(flag)
    elif md_flg == "blip":
        return load_blip_mask(flag)
    else:
        print("No matchable model!")
        return [{} for _ in range(5)]


def create_fake_mask(masks):
    fake = []
    for ms in masks:
        fk = {}
        for k, v in ms.items():
            t = []
            for vv in v:
                if v == "lang":
                    mx = 11008 + 1
                elif v == "vision":
                    mx = 4096 + 1
                elif v == "encoder":
                    mx = 6144 + 1
                elif v == "qformer":
                    mx = 3072 + 1
                else:
                    mx = 2048 + 1
                tt = torch.randint_like(torch.tensor(vv), 0, mx)
                t.append(tt)
            fk[k] = t
        fake.append(fk)
    return fake


def infer(processor, model):
    dt = load_pair("ad")
    prompt, image, answer, index = next(dt)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=256,
        min_length=1,
    )


if __name__ == "__main__":
    _processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")

    _model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf",
                                                               low_cpu_mem_usage=True)
    _model.to(device)
    res = reset_dict(_model, "llava")
    llava_masks = load_mask(['lang', 'vision'], 'llava')
    change_forward(llava_masks, _model, res, "llava")
    infer(_processor, _model)
    print(res)
    print("LLaVA success!")
