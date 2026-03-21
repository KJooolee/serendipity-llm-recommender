# Multi-Faceted LLM Recommendation System for Enhancing Serendipity

🎉 **[Upcoming Publication]** *This paper has been accepted for publication. Details and citation information will be updated soon.*

📊 **[View Presentation Slides](./Multi-Faceted_LLM_Recommendation_System_for_Enhancing_Serendipity.pdf)**

This repository contains the codebase for our research on enhancing serendipity and diversity in LLM-based recommendation systems without significantly compromising user preference matching.

## 📖 Overview

[cite_start]While traditional recommendation systems like LRURec and LlamaRec excel at accuracy, they often lead to **filter bubbles** by repeatedly exposing users to similar content[cite: 30, 34, 35]. [cite_start]On the other hand, diversity-focused models (e.g., DGRRec, IDSR) often sacrifice overall performance to break these bubbles[cite: 58, 59]. [cite_start]Our methodology bridges this gap by leveraging both **item** and **category** preferences, extracting highly diverse yet culturally/personally relevant candidates, and ranking them using a Large Language Model (LLM)[cite: 185, 186, 189].

### Key Contributions
* [cite_start]**Phase 1. Dual Candidate Extraction & Embedding Projection**: Utilizes LRURec to build both an *Item Recommendation Model* and a *Category Recommendation Model*[cite: 216, 219, 296].
* **Controllable Diversity**: Introduces a diversity parameter `k` to select the number of user-preferred categories. [cite_start]Within these chosen categories, `n` items are extracted to form the final candidate set[cite: 288, 289, 290].
* [cite_start]**Phase 2. LLM-based Re-Ranking**: Similar to the LlamaRec architecture, we pass both LRURec's item and category representations through an MLP projector, allowing the LLM to deeply understand both item-level and category-level characteristics simultaneously[cite: 228, 229, 296, 345].
* [cite_start]**Superior N-Diversity Performance**: Achieves state-of-the-art performance on the *N-Diversity* dataset (recommending categories the user has never interacted with), outperforming all baseline models including generic and diversity-specific recommenders[cite: 67, 513].

## 🗂️ Dataset
* [cite_start]Evaluated on the **Amazon Review Dataset (Toys & Games)**[cite: 429].
* [cite_start]Constructed specific **N-Diversity data samples** to strictly evaluate the model's ability to recommend unexplored categories outside the user's past interaction history[cite: 481].

## 📁 Repository Structure

* `train_retriever.py`: Script to train the foundational LRURec retriever models for Items and Categories.
* `train_ranker.py`: Script to train the LLM ranker (e.g., Llama 2) using LoRA and train the MLP projectors.
* `model/`: Contains model architecture definitions (LRURec, LLM wrappers, MLP Projectors).
* `dataloader/`: Data loading pipelines and iterators.
* `datasets/`: Dataset processing scripts.
* `trainer/`: Training loops for retrievers and text-generation LLMs.
* `config.py`: Configuration settings, hyper-parameters, and wandb setup.

## 🚀 Getting Started

### Prerequisites

You will need an environment with PyTorch and CUDA configured. Install the required dependencies:

```bash
pip install -r requirements.txt
