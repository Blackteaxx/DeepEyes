import argparse
import ast
import base64
import copy
import html
import io
import json
import math
import os
from io import BytesIO

import pandas as pd
import requests
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="qwen", help="Model name for result save")
parser.add_argument("--api_key", type=str, default="EMPTY", help="API key")
parser.add_argument("--api_url", type=str, default="http://10.39.19.140:8000/v1", help="API URL")
parser.add_argument("--hrbench_path", type=str, required=True, help="Path to the HR-Bench dataset")
parser.add_argument("--save_path", type=str, required=True, help="Path to save the report")
parser.add_argument("--eval_model_name", type=str, default=None, help="Model name for evaluation")
parser.add_argument("--test_type", type=str, default="hr_bench_4k", help="Single HR-Bench split to inspect")
parser.add_argument("--sample_index", type=int, required=True, help="Single sample index in the TSV file")
parser.add_argument("--fix_bbox", action="store_true", help="Map model bbox coords back to original image coords")
parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of tool turns")
args = parser.parse_args()


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

START_TOKEN = "<tool_call>"
END_TOKEN = "</tool_call>"

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28


def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    if image.mode != "RGB":
        image = image.convert("RGB")
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def round_by_factor(number, factor):
    return round(number / factor) * factor


def ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor


def floor_by_factor(number, factor):
    return math.floor(number / factor) * factor


def smart_resize(height, width, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
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


def get_eval_model_name(api_base, requested_model_name):
    if requested_model_name is not None:
        return requested_model_name
    response = requests.get(f"{api_base}/models")
    response.raise_for_status()
    models = response.json()
    return models["data"][0]["id"]


def safe_parse_json_like(text):
    candidates = [text.strip()]
    if END_TOKEN in text:
        candidates.append(text.split(END_TOKEN)[0].strip())
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            return ast.literal_eval(candidate)
        except Exception:
            pass
    raise ValueError(f"Failed to parse tool call payload: {text}")


def extract_tool_call(response_text):
    if START_TOKEN not in response_text:
        return None
    payload = response_text.split(START_TOKEN, 1)[1].strip()
    return safe_parse_json_like(payload)


def extract_answer(response_text):
    if "<answer>" in response_text and "</answer>" in response_text:
        return response_text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return response_text.strip()


def clamp_bbox(bbox, width, height):
    left, top, right, bottom = bbox
    left = max(0, min(int(left), width))
    top = max(0, min(int(top), height))
    right = max(0, min(int(right), width))
    bottom = max(0, min(int(bottom), height))
    return [left, top, right, bottom]


def map_bbox_to_original(bbox, width_scale, height_scale, ori_width, ori_height):
    left, top, right, bottom = bbox
    mapped = [
        int(left * width_scale),
        int(top * height_scale),
        int(right * width_scale),
        int(bottom * height_scale),
    ]
    return clamp_bbox(mapped, ori_width, ori_height)


def bbox_position_summary(bbox, width, height):
    left, top, right, bottom = bbox
    box_w = max(0, right - left)
    box_h = max(0, bottom - top)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    center_x_ratio = center_x / max(width, 1)
    center_y_ratio = center_y / max(height, 1)
    area_ratio = (box_w * box_h) / max(width * height, 1)

    def bucket(value):
        if value < 1.0 / 3.0:
            return "left_or_top"
        if value < 2.0 / 3.0:
            return "middle"
        return "right_or_bottom"

    horizontal = bucket(center_x_ratio)
    vertical = bucket(center_y_ratio)
    region_map = {
        ("left_or_top", "left_or_top"): "top-left",
        ("middle", "left_or_top"): "top-center",
        ("right_or_bottom", "left_or_top"): "top-right",
        ("left_or_top", "middle"): "center-left",
        ("middle", "middle"): "center",
        ("right_or_bottom", "middle"): "center-right",
        ("left_or_top", "right_or_bottom"): "bottom-left",
        ("middle", "right_or_bottom"): "bottom-center",
        ("right_or_bottom", "right_or_bottom"): "bottom-right",
    }
    region = region_map[(horizontal, vertical)]
    return {
        "region": region,
        "center_x_ratio": round(center_x_ratio, 4),
        "center_y_ratio": round(center_y_ratio, 4),
        "area_ratio": round(area_ratio, 6),
        "box_width": box_w,
        "box_height": box_h,
    }


def ensure_min_canvas_size(image, max_width=1000):
    width, height = image.size
    if width <= max_width:
        return image
    scale = max_width / float(width)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.BICUBIC)


def draw_bbox_panel(base_image, bbox, title_lines, color=(255, 64, 64)):
    image = base_image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    box = clamp_bbox(bbox, image.width, image.height)
    draw.rectangle(box, outline=color, width=max(2, image.width // 300))

    text = "\n".join(title_lines)
    text_box = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    box_w = text_box[2] - text_box[0]
    box_h = text_box[3] - text_box[1]
    margin = 10
    panel = [margin, margin, margin + box_w + 12, margin + box_h + 12]
    draw.rounded_rectangle(panel, radius=8, fill=(0, 0, 0))
    draw.multiline_text((margin + 6, margin + 6), text, fill=(255, 255, 255), font=font, spacing=4)
    return image


def compose_turn_board(original_panel, model_panel, crop_panel, meta_lines, save_path):
    font = ImageFont.load_default()
    original_panel = ensure_min_canvas_size(original_panel, max_width=1200)
    model_panel = ensure_min_canvas_size(model_panel, max_width=1200)
    crop_panel = ensure_min_canvas_size(crop_panel, max_width=1200)

    max_top_height = max(original_panel.height, model_panel.height)
    top_gap = 24
    left_width = max(original_panel.width, model_panel.width)
    top_width = left_width + top_gap + crop_panel.width

    dummy = Image.new("RGB", (10, 10), "white")
    dummy_draw = ImageDraw.Draw(dummy)
    meta_text = "\n".join(meta_lines)
    text_bbox = dummy_draw.multiline_textbbox((0, 0), meta_text, font=font, spacing=4)
    meta_height = text_bbox[3] - text_bbox[1]

    width = top_width + 40
    height = original_panel.height + model_panel.height + crop_panel.height + meta_height + 96
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)

    draw.text((20, 16), "Original image with bbox", fill=(0, 0, 0), font=font)
    canvas.paste(original_panel, (20, 36))
    draw.text((20, 36 + original_panel.height + 12), "Model input image with bbox", fill=(0, 0, 0), font=font)
    canvas.paste(model_panel, (20, 36 + original_panel.height + 32))

    crop_x = 20 + left_width + top_gap
    draw.text((crop_x, 16), "Crop sent back to model", fill=(0, 0, 0), font=font)
    canvas.paste(crop_panel, (crop_x, 36))

    meta_y = max(36 + original_panel.height + 32 + model_panel.height, 36 + crop_panel.height) + 24
    draw.rounded_rectangle((20, meta_y, width - 20, meta_y + meta_height + 20), radius=12, fill=(255, 255, 255))
    draw.multiline_text((32, meta_y + 10), meta_text, fill=(0, 0, 0), font=font, spacing=4)
    canvas.save(save_path)


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def sanitize_messages_for_json(messages, image_refs):
    image_iter = iter(image_refs)
    serialized = []
    for message in messages:
        new_message = {"role": message["role"]}
        content = message["content"]
        if isinstance(content, str):
            new_message["content"] = content
        else:
            new_content = []
            for item in content:
                if item.get("type") == "image_url":
                    new_content.append(
                        {
                            "type": "image_url",
                            "image_path": next(image_iter, None),
                            "max_pixels": item.get("max_pixels"),
                        }
                    )
                else:
                    new_content.append(item)
            new_message["content"] = new_content
        serialized.append(new_message)
    return serialized


def render_report_html(report_path, trace, output_dir):
    turns_html = []
    for turn in trace["turns"]:
        images_html = []
        for key in ["board_path", "original_overlay_path", "model_overlay_path", "crop_sent_path"]:
            rel_path = turn.get(key)
            if rel_path:
                images_html.append(
                    f'<div class="img-card"><div class="img-title">{html.escape(key)}</div>'
                    f'<img src="{html.escape(rel_path)}" alt="{html.escape(key)}"></div>'
                )

        turn_html = f"""
        <section class="turn">
          <h2>Turn {turn["turn_idx"]}</h2>
          <pre>{html.escape(json.dumps(turn["tool_info"], indent=2, ensure_ascii=False))}</pre>
          <h3>Input To Model</h3>
          <pre>{html.escape(json.dumps(turn["request_messages"], indent=2, ensure_ascii=False))}</pre>
          <h3>Assistant Output</h3>
          <pre>{html.escape(turn["assistant_raw"])}</pre>
          <div class="image-grid">
            {''.join(images_html)}
          </div>
        </section>
        """
        turns_html.append(turn_html)

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>HRBench Single Sample Trace</title>
  <style>
    body {{
      font-family: monospace;
      margin: 24px;
      background: #f6f6f6;
      color: #111;
    }}
    h1, h2, h3 {{
      margin-bottom: 8px;
    }}
    section {{
      background: white;
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #f0f0f0;
      padding: 12px;
      border-radius: 8px;
      overflow-x: auto;
    }}
    .image-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-top: 12px;
    }}
    .img-card {{
      background: #fafafa;
      border: 1px solid #e2e2e2;
      border-radius: 10px;
      padding: 12px;
    }}
    .img-title {{
      font-weight: bold;
      margin-bottom: 8px;
    }}
    img {{
      width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid #ddd;
    }}
  </style>
</head>
<body>
  <section>
    <h1>HRBench Single Sample Trace</h1>
    <pre>{html.escape(json.dumps(trace["summary"], indent=2, ensure_ascii=False))}</pre>
  </section>
  <section>
    <h2>Initial Request</h2>
    <h3>System Prompt</h3>
    <pre>{html.escape(trace["system_prompt"])}</pre>
    <h3>User Prompt</h3>
    <pre>{html.escape(trace["user_prompt"])}</pre>
    <div class="image-grid">
      <div class="img-card">
        <div class="img-title">original_image_path</div>
        <img src="{html.escape(trace["original_image_path"])}" alt="original_image_path">
      </div>
      <div class="img-card">
        <div class="img-title">model_input_image_path</div>
        <img src="{html.escape(trace["model_input_image_path"])}" alt="model_input_image_path">
      </div>
    </div>
  </section>
  {''.join(turns_html)}
</body>
</html>
"""
    save_text(report_path, html_text)


def build_prompt(anno):
    options = [
        "A. " + anno["A"],
        "B. " + anno["B"],
        "C. " + anno["C"],
        "D. " + anno["D"],
    ]
    option_str = "\n" + "\n".join(options) + "\n"
    prompt = instruction_prompt_before.format(question=anno["question"], options=option_str)
    return prompt, options


def run_single_sample():
    openai_api_base = args.api_url
    eval_model_name = get_eval_model_name(openai_api_base, args.eval_model_name)
    client = OpenAI(api_key=args.api_key, base_url=openai_api_base)

    tsv_path = os.path.join(args.hrbench_path, args.test_type + ".tsv")
    df = pd.read_csv(tsv_path, sep="\t")
    if args.sample_index < 0 or args.sample_index >= len(df):
        raise IndexError(f"sample_index {args.sample_index} is out of range for {tsv_path} with {len(df)} rows")

    anno = df.iloc[args.sample_index]
    img = decode_base64_to_image(anno["image"])
    ori_image = copy.deepcopy(img)
    ori_width, ori_height = ori_image.size

    # Keep the model-side resize behavior aligned with the original ablation script.
    resize_w, resize_h = smart_resize(ori_width, ori_height, factor=IMAGE_FACTOR)
    model_input_image = ori_image.resize((resize_w, resize_h), resample=Image.BICUBIC)
    width_scale = ori_width / resize_w
    height_scale = ori_height / resize_h

    prompt, options = build_prompt(anno)
    sample_tag = f"{args.test_type}_idx_{args.sample_index:05d}"
    output_dir = os.path.join(args.save_path, args.model_name, sample_tag)
    assets_dir = os.path.join(output_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    original_image_path = os.path.join(assets_dir, "original.png")
    model_input_image_path = os.path.join(assets_dir, "model_input.png")
    ori_image.save(original_image_path)
    model_input_image.save(model_input_image_path)

    base64_image = encode_pil_image_to_base64(model_input_image)
    messages = [
        {"role": "system", "content": instruction_prompt_system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, "max_pixels": MAX_PIXELS},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    image_refs = [os.path.relpath(model_input_image_path, output_dir)]
    turns = []
    response_message = ""
    status = "success"

    for turn_idx in range(args.max_steps + 1):
        request_messages = sanitize_messages_for_json(messages, image_refs)
        params = {
            "model": eval_model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 8192,
            "stop": ["<|im_end|>\n".strip(), "</tool_call>"],
        }

        try:
            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content or ""
        except Exception as exc:
            status = "error"
            response_message = f"[API_ERROR] {exc}"
            turns.append(
                {
                    "turn_idx": turn_idx,
                    "request_messages": request_messages,
                    "assistant_raw": response_message,
                    "tool_info": {"error": str(exc)},
                }
            )
            break

        tool_call = extract_tool_call(response_message)
        if tool_call is None:
            turns.append(
                {
                    "turn_idx": turn_idx,
                    "request_messages": request_messages,
                    "assistant_raw": response_message,
                    "tool_info": {"type": "final_answer_or_plain_text"},
                }
            )
            break

        bbox_model = tool_call["arguments"]["bbox_2d"]
        bbox_model = [int(round(v)) for v in bbox_model]
        bbox_model_clamped = clamp_bbox(bbox_model, resize_w, resize_h)
        bbox_original_mapped = map_bbox_to_original(bbox_model, width_scale, height_scale, ori_width, ori_height)

        if args.fix_bbox:
            crop_bbox_used = bbox_original_mapped
            crop_resize_h, crop_resize_w = smart_resize(
                max(1, crop_bbox_used[3] - crop_bbox_used[1]),
                max(1, crop_bbox_used[2] - crop_bbox_used[0]),
                factor=IMAGE_FACTOR,
            )
        else:
            crop_bbox_used = bbox_model
            crop_resize_w, crop_resize_h = smart_resize(
                max(1, crop_bbox_used[2] - crop_bbox_used[0]),
                max(1, crop_bbox_used[3] - crop_bbox_used[1]),
                factor=IMAGE_FACTOR,
            )

        crop_image_raw = ori_image.crop(tuple(crop_bbox_used))
        crop_image_sent = crop_image_raw.resize((crop_resize_w, crop_resize_h), resample=Image.BICUBIC)

        position_original = bbox_position_summary(bbox_original_mapped, ori_width, ori_height)
        position_model = bbox_position_summary(bbox_model_clamped, resize_w, resize_h)

        label = tool_call.get("arguments", {}).get("label", "")
        original_overlay = draw_bbox_panel(
            ori_image,
            bbox_original_mapped,
            [
                f"turn={turn_idx}",
                f"mapped_bbox={bbox_original_mapped}",
                f"region={position_original['region']}",
                f"center=({position_original['center_x_ratio']:.3f}, {position_original['center_y_ratio']:.3f})",
                f"area_ratio={position_original['area_ratio']:.6f}",
                f"label={label}" if label else "label=<empty>",
            ],
            color=(255, 64, 64),
        )
        model_overlay = draw_bbox_panel(
            model_input_image,
            bbox_model_clamped,
            [
                f"turn={turn_idx}",
                f"model_bbox={bbox_model_clamped}",
                f"region={position_model['region']}",
                f"center=({position_model['center_x_ratio']:.3f}, {position_model['center_y_ratio']:.3f})",
                f"area_ratio={position_model['area_ratio']:.6f}",
            ],
            color=(32, 128, 255),
        )

        board_lines = [
            f"sample={sample_tag}",
            f"fix_bbox={args.fix_bbox}",
            f"label={label}" if label else "label=<empty>",
            f"model_bbox={bbox_model}",
            f"model_bbox_clamped={bbox_model_clamped}",
            f"mapped_bbox_original={bbox_original_mapped}",
            f"crop_bbox_used={crop_bbox_used}",
            f"original_region={position_original['region']}",
            f"model_region={position_model['region']}",
            f"crop_sent_size={list(crop_image_sent.size)}",
            f"assistant_output_preview={response_message[:300]}",
        ]

        original_overlay_path = os.path.join(assets_dir, f"turn_{turn_idx:02d}_original_overlay.png")
        model_overlay_path = os.path.join(assets_dir, f"turn_{turn_idx:02d}_model_overlay.png")
        crop_sent_path = os.path.join(assets_dir, f"turn_{turn_idx:02d}_crop_sent.png")
        board_path = os.path.join(assets_dir, f"turn_{turn_idx:02d}_board.png")

        original_overlay.save(original_overlay_path)
        model_overlay.save(model_overlay_path)
        crop_image_sent.save(crop_sent_path)
        compose_turn_board(original_overlay, model_overlay, crop_image_sent, board_lines, board_path)

        crop_b64 = encode_pil_image_to_base64(crop_image_sent)
        content_f = [
            {"type": "text", "text": "<tool_response>"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
            {"type": "text", "text": user_prompt},
            {"type": "text", "text": "</tool_response>"},
        ]

        messages.extend(
            [
                {"role": "assistant", "content": response_message},
                {"role": "user", "content": content_f},
            ]
        )
        image_refs.append(os.path.relpath(crop_sent_path, output_dir))

        turns.append(
            {
                "turn_idx": turn_idx,
                "request_messages": request_messages,
                "assistant_raw": response_message,
                "tool_info": {
                    "label": label,
                    "tool_call": tool_call,
                    "bbox_model": bbox_model,
                    "bbox_model_clamped": bbox_model_clamped,
                    "bbox_original_mapped": bbox_original_mapped,
                    "crop_bbox_used": crop_bbox_used,
                    "position_model": position_model,
                    "position_original": position_original,
                    "crop_sent_size": list(crop_image_sent.size),
                },
                "original_overlay_path": os.path.relpath(original_overlay_path, output_dir),
                "model_overlay_path": os.path.relpath(model_overlay_path, output_dir),
                "crop_sent_path": os.path.relpath(crop_sent_path, output_dir),
                "board_path": os.path.relpath(board_path, output_dir),
            }
        )
    else:
        status = "max_steps_exceeded"

    final_answer = extract_answer(response_message)
    summary = {
        "status": status,
        "test_type": args.test_type,
        "sample_index": args.sample_index,
        "question": anno["question"],
        "options": options,
        "ground_truth_answer": anno["answer"],
        "ground_truth_answer_text": anno[anno["answer"]],
        "pred_answer": final_answer,
        "category": anno["category"],
        "fix_bbox": args.fix_bbox,
        "model_name": args.model_name,
        "eval_model_name": eval_model_name,
        "output_dir": output_dir,
        "turn_count": len(turns),
    }

    trace = {
        "summary": summary,
        "system_prompt": instruction_prompt_system,
        "user_prompt": prompt,
        "original_image_path": os.path.relpath(original_image_path, output_dir),
        "model_input_image_path": os.path.relpath(model_input_image_path, output_dir),
        "turns": turns,
    }

    trace_path = os.path.join(output_dir, "trace.json")
    report_path = os.path.join(output_dir, "report.html")
    summary_path = os.path.join(output_dir, "summary.json")
    save_json(trace_path, trace)
    save_json(summary_path, summary)
    render_report_html(report_path, trace, output_dir)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Trace JSON: {trace_path}")
    print(f"Report HTML: {report_path}")


if __name__ == "__main__":
    run_single_sample()
