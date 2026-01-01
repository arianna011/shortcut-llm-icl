# Shortcut Detection and Mitigation via Representation Engineering
**Representation Engineering** (Zou et al., [2025](https://arxiv.org/abs/2310.01405)) has proven effective at manipulating the hidden representations of Large Language Models (LLMs) to amplify or suppress specific behaviors, such as honesty, power-seeking or toxicity. 

This project investigates whether Representation Engineering can be adapted to **detect and mitigate shortcut learning in In-Context Learning (ICL)**, a phenomenon that undermines the robustness and generalization of LLMs when exposed to out-of-distribution inputs. 

The proposed framework is lightweight and interpretable, operating directly in representation space, without additional training or parameter updates. Experiments conducted on [Mistral 7B Instruct v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) demonstrate consistent improvements over standard baselines on widely used NLP benchmarks, including **RTE**, **MNLI**, **SST-2** and **MMLU**, highlighting the potential of this approach for improving model robustness under shortcut-prone settings.

This repository contains the code accompanying a **Master’s Degree thesis in Computer Science**, completed at **Sapienza University of Rome** during the academic year **2024/2025**.

## Notebooks (Google Colab)

The following notebooks provide a direct interface with the proposed framework and are intended for use in **Google Colab**:

- **[Shortcut_Mitigation_Steps.ipynb](https://github.com/arianna011/shortcut-llm-icl/tree/main/Shortcut_Mitigation_Steps.ipynb)**  
  Executes the full shortcut mitigation pipeline step by step, from dataset loading to hidden activation extraction, representation-level steering and evaluation on the selected benchmark.

- **[Shortcut_Mitigation_Evaluation.ipynb](https://github.com/arianna011/shortcut-llm-icl/tree/main/Shortcut_Mitigation_Evaluation.ipynb)**  
  Launches evaluation runs for the proposed mitigation approach. The notebook is easily customizable via interactive Google Colab parameters, includes logging to _Weights & Biases_ and supports hyperparameter sweeps.

- **[Qualitative_Analysis_Detection_Mitigation.ipynb](https://github.com/arianna011/shortcut-llm-icl/tree/main/Qualitative_Analysis_Detection_Mitigation.ipynb)**  
  Performs qualitative experiments to assess the effectiveness of both shortcut detection and mitigation, and generates the plots discussed in the [thesis](https://github.com/arianna011/shortcut-llm-icl/tree/main/thesis.pdf).


## Code Structure

The notebooks rely on the following modules:

- **[extract_activations](https://github.com/arianna011/shortcut-llm-icl/tree/main/extract_activations)**  
  Utilities for data preprocessing and extraction of hidden representations associated with shortcut-related behaviors using Representation Reading methods, enabling subsequent Representation Control interventions.

- **[patched_unibias](https://github.com/arianna011/shortcut-llm-icl/tree/main/patched_unibias)**  
  An extension of the original [UniBias repository](https://github.com/hzzhou01/UniBias) (Zhou et al., 2024), a framework for evaluating LLMs under In-Context Learning across multiple NLP benchmarks. The codebase was extended to support Representation Engineering interventions at inference time and to log artifacts and results to Weights & Biases.

- **[representation_engineering](https://github.com/arianna011/shortcut-llm-icl/tree/main/representation_engineering)**  
  A modified import of the original [Representation Engineering repository](https://github.com/andyzoujm/representation-engineering) (Zou et al., 2025), adapted to ensure compatibility with the proposed framework and to implement missing functionalities.

## Data

The folder **[data/ShortcutSuite](https://github.com/arianna011/shortcut-llm-icl/tree/main/data/ShortcutSuite)** contains an import of the shortcut-injected datasets used to construct contrastive examples for extracting shortcut directions in representation space, sourced from the original [ShortcutSuite repository](https://github.com/yyhappier/ShortcutSuite).