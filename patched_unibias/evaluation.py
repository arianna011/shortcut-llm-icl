import torch
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from utils import *
from transformers import pipeline
import sys
import os
from typing import List,Union
import gc
from accelerate.utils import release_memory
from tqdm import tqdm
import csv, json
from representation_engineering import repe_pipeline_registry

def ICL_evaluation(model: transformers.PreTrainedModel, 
                   tokenizer: transformers.AutoTokenizer,
                   device: torch.device,
                   prompt_list: List[str], 
                   labels: Union[List[int], List[str]], 
                   gt_ans_ids_list: List[List[int]], 
                   dataset_name: str, 
                   gen_tokens: int = 10,
                   repE: bool = False, 
                   layers: List[int] = [],
                   activations_path: str = "",
                   save_every: int = 100, 
                   save_dir: str = "results_tmp", 
                   resume: bool = False,
                   failures_csv_path: str = "") -> tuple[float, List[List[float]], List[List[int]]]:
    """
    Evaluate the ICL accuracy of the input model on the provided prompts.

    Params:
        model: the LLM to evaluate
        tokenizer: the tokenizer to use
        device: the device to use
        prompt_list: a list of prompts to feed to the model
        labels: ground truth labels for the task at hand
        gt_ans_ids_list: list of possible token ids for the representation of each ground truth label
        dataset_name: name of the dataset
        gen_tokens: maximum number of new tokens to generate
        repE: whether to use Representantion Engineering
        layers: list of hidden layer numbers in which representation control must be injected
        activations: path to the file containing the activations for representantion control
        save_every: number of samples to process before storing intermediate results
        save_dir: where to store intermediate results
        resume: whether to restart an interrupted evaluation from the last stored intermediate results
        failures_csv_path: a path to a CSV file where to store misclassified examples

    Returns:
        (float) final classification accuracy
        (list[list[float]]) all label probabilities for each prompt
        (list[list[int]]) confusion matrix

    """
    predictions, all_label_probs = [], []
    tokenizer.pad_token = tokenizer.eos_token

    fail_examples = []

    # resume logic
    start_index = 0 
    model_name = model.name_or_path.split("/")[-1] 
    resume_dir = save_dir+"_"+model_name+"_"+dataset_name
    if repE:
      resume_dir += "_repE"
    os.makedirs(resume_dir, exist_ok=True)

    model.eval()

    if repE:
      
      print("Using Representation Engineering")
      repe_pipeline_registry()

      # initialize a control pipeline for Mistral
      control_method="reading_vec" # add a scaled version of a direction vector to internal activations during forward pass

      rep_control_pipeline = pipeline(
          "rep-control",
          model=model,
          tokenizer=tokenizer,
          layers=layers, # layers to inject control in
          control_method=control_method)
      
      # load pre-saved direction vectors
      activations = torch.load(activations_path)

    # resume an interrupted evaluation
    if resume:
        result_files = sorted(
            [f for f in os.listdir(resume_dir) if f.endswith(".pkl")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        if result_files:
            last_file = result_files[-1]
            with open(os.path.join(resume_dir, last_file), "rb") as f:
                data = pickle.load(f)
                predictions = data["predictions"]
                all_label_probs = data["all_label_probs"]
                start_index = data["indices"][-1] + 1
            print(f"Resuming from index {start_index} using {last_file}")


    for index in tqdm(range(start_index, len(prompt_list)), desc="Processing"):
        
        prompt = prompt_list[index]

        with torch.no_grad():

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            prompt_length = len(tokenizer.decode(
                input_ids[0],
                skip_special_tokens=True))

            gc.collect()
            torch.cuda.empty_cache() 
            
            if repE:  
              output_dict = rep_control_pipeline(prompt, 
                                                activations=activations,  
                                                max_new_tokens=gen_tokens, 
                                                do_sample=False, 
                                                repetition_penalty=1.1)
            else:
              output_dict = model.generate(input_ids=inputs["input_ids"], 
                                            attention_mask=attention_mask,
                                            max_new_tokens=gen_tokens, 
                                            return_dict_in_generate=True, 
                                            output_scores=True, 
                                            pad_token_id=tokenizer.eos_token_id)
              
            outputs = output_dict['sequences'] # token IDs of the prompt + generated text
            scores = output_dict['scores'] # list of tensors (each shape [1, vocab_size]) with logits for each generated token step

            outputs = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0] # (batch size = 1)
            outputs_only = outputs[prompt_length:]

            if dataset_name == 'copa':
                answer_logits = scores[1][0] # llama output '' before numbers (1 or 2)
            else:
                answer_logits = scores[0][0]

            answer_probs = F.softmax(answer_logits, dim=-1)
            prediction_i, predict_prob_list = predict_label(answer_probs, gt_ans_ids_list)
            predictions.append(prediction_i)
            all_label_probs.append(predict_prob_list)

            # optionally save failure examples
            if failures_csv_path:
                ground_truth = labels[index]
                if prediction_i != ground_truth:
                     fail_examples.append({
                        "index": index,
                        "prompt": prompt.replace('\n', ' ').replace('\r', ' ').strip(),
                        "prediction": int(prediction_i),
                        "ground_truth": int(ground_truth),
                        "output_text": outputs_only.strip().replace('\n', ' ').replace('\r', ' '),
                        "probabilities": json.dumps(predict_prob_list)
                    })

            # free memory 
            del inputs, input_ids, attention_mask, output_dict, scores, answer_probs
            gc.collect()
            torch.cuda.empty_cache()
            release_memory(model)
       
        # save partial results
        if (index + 1) % save_every == 0 or index == len(prompt_list) - 1:
            partial_file = os.path.join(resume_dir, f"results_{index + 1}.pkl")
            with open(partial_file, "wb") as f:
                pickle.dump({
                    "predictions": predictions,
                    "all_label_probs": all_label_probs,
                    "indices": list(range(index + 1))
                }, f)
            tqdm.write(f"Saved checkpoint at step {index + 1} → {partial_file}")

            # evaluate intermediate performance
            cf = confusion_matrix(labels[:len(predictions)], predictions)
            normalized_cf = cf / cf.sum(axis=1, keepdims=True)
            tqdm.write(f"Partial normalized confusion matrix:\n{np.array2string(normalized_cf, precision=2)}")
            acc, _ = classification_accuracy(predictions, labels[:len(predictions)])
            tqdm.write(f"Partial accuracy: {acc}")

    acc, _ = classification_accuracy(predictions, labels[:len(predictions)])
    print('Final classification accuracy:', acc)
    print(cf)
    final_accuracy = '\n\nclassification_accuracy: ' + str(acc)

    # store fail examples on file
    if fail_examples:
        with open(failures_csv_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fail_examples[0].keys(), quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for ex in fail_examples:
                writer.writerow({
                "index": ex["index"],
                "prompt": ex["prompt"],
                "prediction": ex["prediction"],
                "ground_truth": ex["ground_truth"],
                "output_text": ex["output_text"],
                "probabilities": json.dumps(ex["probabilities"])
            })
        print(f"\nSaved {len(fail_examples)} failure cases to {failures_csv_path}")
    

    return final_accuracy, all_label_probs, cf



def calibration_evaluation(model, all_label_probs, gt_ans_ids_list, test_sentences, test_labels, demonstration):
    from main import record_file_path, dataset_name, seed_value
    is_sentence_pair = isinstance(test_sentences[0], list)
    # CC
    if is_sentence_pair:
        repeat_num = len(test_sentences[0])
        base_content_free_inputs = ["N/A", "", "[MASK]"]
        content_free_inputs = [[item] * repeat_num for item in base_content_free_inputs]
    else:
        content_free_inputs = ["N/A", "", "[MASK]"]
    content_free_prompt_list, _, _, _ = prepare_dataset_test(dataset_name, content_free_inputs, demonstration)
    p_cc = get_p_content_free(model, content_free_prompt_list, gt_ans_ids_list, dataset_name)
    acc_calibrated_cc, c_m = eval_accuracy(np.array(all_label_probs), test_labels, mode="diagonal_W", p_cf=p_cc)
    write_json(record_file_path, 'CC_calibrate: ' + str(acc_calibrated_cc) + str(c_m))
    print('CC_calibrate: ' + str(acc_calibrated_cc))
    # DC
    content_free_inputs = sample_random_texts(texts=test_sentences, n_sample=20, seed=seed_value)
    print(f"random texts for estimating prior: \n{content_free_inputs}")
    content_free_prompt_list, _, _, _ = prepare_dataset_test(dataset_name, content_free_inputs, demonstration)
    p_dc = get_p_content_free(model, content_free_prompt_list, gt_ans_ids_list, dataset_name)
    acc_calibrated_dc, c_m = eval_accuracy(np.array(all_label_probs), test_labels, mode="diagonal_W", p_cf=p_dc)
    write_json(record_file_path, 'DC_calibrate: ' + str(acc_calibrated_dc) + str(c_m))
    print('DC_calibrate: ' + str(acc_calibrated_dc))
    # PC
    estimate_prompt_list = gen_PC_calibration_data(dataset_name, demonstration)
    estimate_label_ps, estimate_label_logps = get_logp_estimate_data(model, estimate_prompt_list, gt_ans_ids_list, dataset_name)
    vecs = np.array(estimate_label_logps)
    max_cla = -1000000
    best_seed = 0
    for seed in range(100):
        gmm = GaussianMixture(n_components=len(set(test_labels)), random_state=seed)
        gmm.fit(vecs)
        documents_to_class = gmm.predict(vecs)
        centers = gmm.means_
        row_ind, col_ind = linear_sum_assignment(centers.max() - centers)
        cla = centers[row_ind, col_ind].sum()
        # print(cla, centers)
        if cla > max_cla:
            max_cla = cla
            best_seed = seed
    gmm = GaussianMixture(n_components=len(set(test_labels)), random_state=best_seed)
    gmm.fit(vecs)
    # documents_to_class = gmm.predict(vecs)
    centers = gmm.means_
    row_ind, col_ind = linear_sum_assignment(centers.max() - centers)
    test_vecs = np.log(np.array(all_label_probs))
    documents_to_class = gmm.predict(test_vecs)
    predictions = [int(col_ind[documents_to_class[i]]) for i in range(len(test_vecs))]
    acc_calibrated_pc, _ = classification_accuracy(predictions, test_labels)
    c_m = confusion_matrix(test_labels, predictions)
    write_json(record_file_path, 'PC_calibrate: ' + str(acc_calibrated_pc) + str(c_m))
    print('PC_calibrate: ' + str(acc_calibrated_pc))

def get_logp_estimate_data(model, estimate_prompt_list, gt_ans_ids_list, dataset_name):
    from main import tokenizer, device
    print('evaluate content free probs')
    all_p_y = []
    all_logp_y = []
    for index, prompt in enumerate(estimate_prompt_list):
        print(f"Evaluating: {index+1}/{len(estimate_prompt_list)}")
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            prompt_length = len(tokenizer.decode(
                input_ids[0],
                skip_special_tokens=True,
                # clean_up_tokenization_spaces=True,
            ))
            output_dict = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10, return_dict_in_generate=True, output_scores=True)
            outputs = output_dict['sequences']
            scores = output_dict['scores']
            outputs = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
            outputs_only = outputs[prompt_length:]
            if dataset_name == 'copa':
                answer_logits = scores[1][0]
            else:
                answer_logits = scores[0][0]
            answer_probs = F.softmax(answer_logits, dim=-1)
            p_y = []
            logp_y = []
            for i_id_list in gt_ans_ids_list:
                prob_i = torch.max(answer_probs[torch.tensor(i_id_list)]).item()
                p_y.append(prob_i)
                logp_y.append(np.log(prob_i))
            all_p_y.append(p_y)
            all_logp_y.append(logp_y)
    return all_p_y, all_logp_y

def get_p_content_free(model, content_free_prompt_list, gt_ans_ids_list, dataset_name):
    from main import tokenizer, device
    index = 0
    print('evaluate content free probs')
    all_p_y = []
    for prompt in content_free_prompt_list:
        index += 1
        print(index)
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            prompt_length = len(tokenizer.decode(
                input_ids[0],
                skip_special_tokens=True,
                # clean_up_tokenization_spaces=True,
            ))
            output_dict = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10, return_dict_in_generate=True, output_scores=True)
            outputs = output_dict['sequences']
            scores = output_dict['scores']
            outputs = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
            outputs_only = outputs[prompt_length:]
            if dataset_name == 'copa':
                answer_logits = scores[1][0] # llama output '' before numbers (1 or 2)
            else:
                answer_logits = scores[0][0]
            answer_probs = F.softmax(answer_logits, dim=-1)
            p_y = []
            for i_id_list in gt_ans_ids_list:
                prob_i = torch.max(answer_probs[torch.tensor(i_id_list)]).item()
                p_y.append(prob_i)
            all_p_y.append(p_y)
    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y)
    return p_y

def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False

    correctness_list = []
    assert len(all_label_probs) == len(test_labels)
    true_labels = []
    pred_labels = []
    for label_probs, true_label in zip(all_label_probs, test_labels):
        label_probs = label_probs / np.sum(label_probs) # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b

        ans_label = np.argmax(calibrate_label_probs)
        true_labels.append(true_label)
        pred_labels.append(ans_label)
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    print(f"Confusion matrix: \n {confusion_matrix(true_labels, pred_labels)}")
    return np.mean(correctness_list), confusion_matrix(true_labels, pred_labels)


def predict_label(answer_logit, gt_ans_ids_list):
    prediction_logit_list = []
    # for each label, take the maximum logit among all compatible tokens
    for i_id_list in gt_ans_ids_list:
        logit_i = torch.max(answer_logit[torch.tensor(i_id_list)]).item()
        prediction_logit_list.append(logit_i)
    # take the label with highest score
    prediction = torch.max(torch.tensor(prediction_logit_list), dim=0)[1]
    return prediction, prediction_logit_list

def classification_accuracy(predictions, labels):
    if len(predictions) != len(labels):
        raise ValueError("Both lists must have the same length")

    incorrect_indices = [i for i, (p, l) in enumerate(zip(predictions, labels)) if int(p) != int(l)]
    correct_predictions = sum(int(p) == int(l) for p, l in zip(predictions, labels))
    total_predictions = len(predictions)

    return correct_predictions / total_predictions, incorrect_indices