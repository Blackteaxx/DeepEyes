import os
import json
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import argparse
import torch
from tqdm import tqdm
import math
from io import BytesIO
from PIL import Image
import base64
import io
from openai import OpenAI
import requests
import copy
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='qwen', help='Model name for result save')
parser.add_argument('--api_key', type=str, default='EMPTY', help='API key')
parser.add_argument('--api_url', type=str, default='http://10.39.19.140:8000/v1', help='API URL')
parser.add_argument('--hrbench_path', type=str, default=None, help='Path to the HR-Bench dataset')
parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')
parser.add_argument('--eval_model_name', type=str, default=None, help='Model name for evaluation')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--fix_bbox', action='store_true', help='Fix bbox coordinate mapping (map model coords to original image coords)')
parser.add_argument('--test_types', type=str, default='hr_bench_4k,hr_bench_8k', help='Comma-separated test types')
args = parser.parse_args()

FIX_BBOX = args.fix_bbox

openai_api_key = args.api_key
openai_api_base = args.api_url

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
if args.eval_model_name is None:
    response = requests.get(f"{openai_api_base}/models")
    models = response.json()
    eval_model_name = models['data'][0]['id']
else:
    eval_model_name = args.eval_model_name

hrbench_path = args.hrbench_path
test_types = args.test_types.split(',')
save_path = args.save_path
save_path = os.path.join(save_path, args.model_name)
os.makedirs(save_path, exist_ok=True)

print(f"[CONFIG] fix_bbox={FIX_BBOX}, test_types={test_types}, model={eval_model_name}")

SYSTEM_PROMPT_V2 = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>"""

USER_PROMPT_V2 = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "

instruction_prompt_system = SYSTEM_PROMPT_V2

instruction_prompt_before = """Question: {question}
Options: {options}
""" + USER_PROMPT_V2

user_prompt = USER_PROMPT_V2

start_token = "<tool_call>"
end_token = "</tool_call>"

def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def process(idx):
    idx, df, fix_bbox = idx
    anno = df.iloc[idx]
    img = anno['image']
    img = decode_base64_to_image(img)
    ori_image = copy.deepcopy(img)
    ori_width, ori_height = img.size

    # NOTE: original code passes width as height -- we keep this as-is
    # because the model was served the same way, so bbox coords match this space
    resize_w, resize_h = smart_resize(ori_width, ori_height, factor=IMAGE_FACTOR)
    img = img.resize((resize_w, resize_h), resample=Image.BICUBIC)
    width_scale = ori_width / resize_w
    height_scale = ori_height / resize_h

    question = anno['question']
    answer = anno['answer']
    answer_str = anno[answer]
    options = ['A. ' + anno['A'], 'B. ' + anno['B'], 'C. ' + anno['C'], 'D. ' + anno['D']]
    category = anno['category']

    option_str = "\n"
    for i in range(len(options)):
        option_str += options[i] + '\n'

    prompt = instruction_prompt_before.format(question=question, options=option_str)
    pil_img = ori_image.copy()

    base64_image = encode_pil_image_to_base64(img)

    messages = [
        {
            "role": "system",
            "content": instruction_prompt_system,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, "max_pixels": MAX_PIXELS,},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    print_messages = [
        {
            "role": "system",
            "content": instruction_prompt_system,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}, "max_pixels": MAX_PIXELS,},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_message = messages
    response_message = ""
    status = 'success'
    try_count = 0
    turn_idx = 0
    try:
        while '</answer>' not in response_message:
            if '</answer>' in response_message and '<answer>' in response_message:
                break
            if try_count > 10:
                break

            params = {
                "model": eval_model_name,
                "messages": chat_message,
                "temperature": 0.0,
                "max_tokens": 8192,
                "stop": ["<|im_end|>\n".strip(), "</tool_call>"],
            }
            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content
            
            if start_token in response_message:
                action_list = response_message.split(start_token)[1].split(end_token)[0].strip()
                action_list = eval(action_list)

                bbox_str = action_list['arguments']['bbox_2d']
                left, top, right, bottom = bbox_str

                if fix_bbox:
                    # FIX: map bbox from resized coords to original image coords
                    left = max(0, min(int(left * width_scale), ori_width))
                    top = max(0, min(int(top * height_scale), ori_height))
                    right = max(0, min(int(right * width_scale), ori_width))
                    bottom = max(0, min(int(bottom * height_scale), ori_height))

                cropped_image = pil_img.crop((left, top, right, bottom))
                
                crop_w = right - left
                crop_h = bottom - top
                if fix_bbox:
                    # FIX: pass (height, width) in correct order
                    new_h, new_w = smart_resize(crop_h, crop_w, factor=IMAGE_FACTOR)
                else:
                    # Original buggy: passes width as height
                    new_w, new_h = smart_resize(crop_w, crop_h, factor=IMAGE_FACTOR)
                
                cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
                cropped_pil_image = encode_pil_image_to_base64(cropped_image)
                cropped_pil_image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cropped_pil_image}"}}

                content_f = []
                content_f.append({"type": "text", "text": "<tool_response>"})
                content_f.append(cropped_pil_image_content)
                content_f.append({"type": "text", "text": user_prompt})
                content_f.append({"type": "text", "text": "</tool_response>"})

                _message =[
                    {"role": "assistant", "content": response_message},
                    {"role": "user", "content": content_f}
                ]
                chat_message.extend(_message)

                p_message =[
                    {"role": "assistant", "content": response_message},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,"}},
                        {"type": "text", "text": user_prompt},
                    ]}
                ]
                print_messages.extend(p_message)
                turn_idx += 1
            else:
                print_messages.append({"role": "assistant", "content": response_message})

            try_count += 1
    except Exception as e:
        print(f"Error!!!!", e)
        status = 'error'
                
    if '</answer>' in response_message and '<answer>' in response_message:
        output_text = response_message.split('<answer>')[1].split('</answer>')[0].strip()
    else:
        output_text = response_message

    save_info = {}
    save_info['question'] = question
    save_info['answer'] = answer
    save_info['answer_str'] = answer_str
    save_info['pred_ans'] = output_text
    save_info['pred_output'] = print_messages
    save_info['category'] = category
    save_info['status'] = status

    return save_info


if __name__ == "__main__":
    
    for test_type in test_types:
        save_json = []
        tsv_path = os.path.join(hrbench_path, test_type + '.tsv')
        save_name = f"result_{test_type}_{args.model_name}.jsonl"
        df = pd.read_csv(tsv_path, sep='\t')
        rows_len = df.shape[0]
        pool = multiprocessing.Pool(processes=args.num_workers)
        idx_args = [[idx, df, FIX_BBOX] for idx in range(rows_len)]
        
        with tqdm(total=len(idx_args), desc=f"Processing HRBench {test_type}: ") as pbar:
            for result in pool.imap(process, idx_args):
                if result is not None:
                    save_json.append(result)
                    pbar.update(1)

        pool.close()
        pool.join()

        with open(os.path.join(save_path, save_name), 'w') as f:
            for item in save_json:
                f.write(json.dumps(item) + '\n')
    
    # Quick rule-based accuracy
    print("\n===== Rule-based Accuracy =====")
    for test_type in test_types:
        save_name = f"result_{test_type}_{args.model_name}.jsonl"
        result_path = os.path.join(save_path, save_name)
        with open(result_path) as f:
            res = [json.loads(line) for line in f]
        by_cat = {}
        for r in res:
            pred = r['pred_ans'].strip()
            if pred and pred[0] in 'ABCDEF' and (len(pred) == 1 or pred[1] in '.、 ,'):
                acc = 1.0 if pred[0] == r['answer'] else 0.0
            elif r['answer'] in pred:
                acc = 1.0
            else:
                acc = 0.0
            cat = r['category']
            if cat not in by_cat:
                by_cat[cat] = [0, 0]
            by_cat[cat][0] += acc
            by_cat[cat][1] += 1
        total_c = sum(v[0] for v in by_cat.values())
        total_n = sum(v[1] for v in by_cat.values())
        print(f"\n{test_type}: {int(total_c)}/{total_n} = {total_c/total_n*100:.1f}%")
        for cat, (c, t) in sorted(by_cat.items()):
            print(f"  {cat}: {int(c)}/{t} = {c/t*100:.1f}%")
