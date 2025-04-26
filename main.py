import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from utils import *
import argparse
import random
from evaluation import ICL_evaluation#, calibration_evaluation
from pathlib import Path
#from FFN_manipulate import *
#from attention_manipulate import *

# INITIAL ARGUMENTS
parser = argparse.ArgumentParser(description='Initial arguments.')

#Added support for different LLMs
parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.1',
                    help='Hugging Face model name or local path')
parser.add_argument('--hf_token', type=str, default=None,
                    help='Hugging Face access token for gated models')
parser.add_argument('--use_auto_device_map', type=str, choices=["true", "false"], default="true",
                    help='Use device_map=auto for large models (True recommended)')
parser.add_argument('--cuda_device_id', default='0', type=str,
                    help='If not using auto device map, specify which CUDA device')

parser.add_argument('--seed',  type=int, default=10, help='Random seed')
parser.add_argument('--dataset_name',  type=str, default='sst2', help='Dataset Name')
parser.add_argument('--format_index',  type=int, default=None, help='Gen various prompt format')
parser.add_argument('--order_index',  type=int, default=None, help='Gen various prompt order')
parser.add_argument('--num_shot',  type=int, default=1, help='Number of shot')
parser.add_argument('--UniBias', type=str, choices=["true", "false"], default="false", help="Enable UniBias")
parser.add_argument('--Calibration', type=str, choices=["true", "false"], default="false", help="Enable Calibration")

#Added support for Representation Engineering modifications to LLM activations
parser.add_argument('--RepE', type=str, choices=["true", "false"], default="false", help="Enable Representation Engineering")
parser.add_argument('--layers', type=int, nargs='+', default=[], help="Layers to inject representantion control into") 
parser.add_argument('--tokens', type=int, default=10, help="Max new tokens to generate") 
parser.add_argument('--resume', type=str, choices=["true", "false"], default="false", help="Enable resume of previous interrupted evaluation")

args = parser.parse_args()

model_name = args.model_name
hf_token = args.hf_token or os.environ.get("HF_TOKEN")
use_auto_device_map = args.use_auto_device_map.lower() == "true"
cuda_device_id = args.cuda_device_id
seed_value = args.seed
random.seed(seed_value)
dataset_name = args.dataset_name
format_index = args.format_index
order_index = args.order_index
num_shot = args.num_shot
Unibias = args.UniBias.lower() == "true"
Calibration = args.Calibration.lower() == "true"
RepE = args.RepE.lower() == "true"
rep_layers = args.layers
new_tokens = args.tokens
eval_resume = args.resume.lower() == "true"


# MODEL SETUP
device = torch.device("cuda:0")
model_kwargs = {
    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32
}

if hf_token:
    model_kwargs["token"] = hf_token

if use_auto_device_map:
    model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_id
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token if hf_token else None)

model.eval()

mlm_head = model.lm_head
norm = model.model.norm
model_id = args.model_name.split("/")[-1] 
record_file_path = f'./results_{model_id}/{dataset_name}_seed{seed_value}.json'

file_path = Path(record_file_path)
file_path.parent.mkdir(parents=True, exist_ok=True)


# EVALUATION
def main():

    # Load dataset and generate prompts

    # gen prompts with different examples for different random seeds
    if not order_index and not format_index:
        prompt_list, test_labels, demonstration, test_sentences = prepare_dataset_test(dataset_name, num_shot=num_shot)
        validate_data = prepare_dataset_validate(dataset_name, demonstration)

    # gen prompts with different prompt formatting
    if format_index:
        prompt_list, test_labels, demonstration, test_sentences, ans_label_list = gen_test_data_format(dataset_name, format_index)
        validate_data = gen_validate_data_format(dataset_name, demonstration, format_index)

    # gen prompts with different example order
    if order_index:
        prompt_list, test_labels, demonstration, test_sentences, rand_example_sample_index_order = gen_test_data_order(dataset_name, order_index)
        validate_data = gen_validate_data_order(dataset_name, demonstration)

    # labels of the dataset
    ans_label_list = task_labels(dataset_name)
    # find all possible token ids for labels
    gt_ans_ids_list = find_possible_ids_for_labels(ans_label_list, tokenizer)

    s = ""
    if RepE:
        s += f"layers: {rep_layers}"
    write_json(record_file_path, 
               "num shot: " + str(num_shot) + " new tokens: " + str(new_tokens) + " repE: " + str(RepE) + s)

    # evaluate performance
    final_acc, all_label_probs, cf = ICL_evaluation(model, prompt_list, test_labels, 
                                                    gt_ans_ids_list, dataset_name, 
                                                    repE=RepE, gen_tokens=new_tokens,
                                                    layers=rep_layers, resume=eval_resume)
    write_json(record_file_path, final_acc + str(cf))


if __name__ == "__main__":
    main()