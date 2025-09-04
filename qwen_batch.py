import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from peft import PeftModel, PeftConfig
import torch
import json
import random
from PIL import Image
from tqdm import tqdm

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
lora_path = None
# save_path = '/LLaMA-Factory/saves/cultrual_heritage/qwen2_5vl-7b/origin'
lora_path = "/LLaMA-Factory/saves/cultrual_heritage/qwen2_5vl-7b/lora/cn_lora_fixed"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16
)
model.to('cuda')
processor = AutoProcessor.from_pretrained(model_path)
if lora_path is not None:
    config = PeftConfig.from_pretrained(lora_path)
    model = PeftModel.from_pretrained(model, lora_path)
    save_path = lora_path

# from testing data
testing_root = '/LLaMA-Factory/dataset/testing_dataset_modify'
ann_path = os.path.join(testing_root, 'test_v6_0715.json')
image_root = os.path.join(testing_root, 'imgs')
with open(ann_path, 'r') as f:
    ann = json.load(f)

result_list = []
for result in tqdm(ann['images']):
    image_path = os.path.join(image_root, result['image_file'])
    for question in result['questions']:
        answer = question['groundtruth']
        question_format = question.get('question_format', 'single_selection')
        options = question['options']
        if question_format == 'single_selection':
            prompt = f"{question['question']}\nPlease choose the most appropriate answer from the following options: {', '.join(options)}\n\nImportant: Only output the option content, do not provide any explanation or reasoning."
            system_prompt = "You are a professional architectural façade analysis assistant, capable of observing building images and answering questions about architectural features. Please choose the most appropriate answer from the given options. Important: Only output the option content, do not add any explanation, reasoning, or extra text."
        elif question_format == 'multiple_selection':
            prompt = f"{question['question']}\nPlease select all applicable answers from the following options (multiple selections allowed): {', '.join(options)}\nSeparate multiple answers with commas.\n\nImportant: Only output the option content, do not provide any explanation or reasoning."
            system_prompt = "You are a professional architectural façade analysis assistant, capable of observing building images and answering questions about architectural features. For multiple-choice questions, please select all applicable answers from the given options and separate multiple answers with commas. Important: Only output the option content, do not add any explanation, reasoning, or extra text."
        else:
            prompt = f"{question['question']}\nPlease choose the most appropriate answer from the following options: {', '.join(options)}\n\nImportant: Only output the option content, do not provide any explanation or reasoning."
            system_prompt = "You are a professional architectural façade analysis assistant, capable of observing building images and answering questions about architectural features. Please choose the most appropriate answer from the given options. Important: Only output the option content, do not add any explanation, reasoning, or extra text."

        prompt = '<image>\n' + system_prompt + '\n' + prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        result_dict = {'Question:': question, 'gpt_answer:': output_text[0], 'Groundtruth:': answer, 'question_format': question['question_format'], 'question_type': question['question_type']}
        result_list.append(result_dict)
with open(os.path.join(save_path, 'result_new_test.json'), 'w', encoding='utf-8') as f:
    json.dump(result_list, f, ensure_ascii=False, indent=2)