# Multi-Faceted LLM Recommendation System for Enhancing Serendipity

🎉 **[Upcoming Publication]** *This paper has been accepted for publication. Details and citation information will be updated soon.*

📊 **[View Presentation Slides](./Multi-Faceted_LLM_Recommendation_System_for_Enhancing_Serendipity.pdf)**

This repository contains the codebase for our research on enhancing serendipity and diversity in LLM-based recommendation systems without significantly compromising user preference matching.

## 📖 Overview

While traditional recommendation systems like LRURec and LlamaRec excel at accuracy, they often lack diversity. On the other hand, diversity-focused models (e.g., DGRRec, IDSR) often sacrifice overall performance. Our methodology bridges this gap by leveraging both **item** and **category** preferences, extracting highly diverse yet culturally/personally relevant candidates, and ranking them using a Large Language Model (LLM).

### Key Contributions
* **Dual Candidate Extraction**: Utilizes LRURec to build both an *Item Recommendation Model* and a *Category Recommendation Model*.
* **Controllable Diversity**: Introduces a diversity parameter `k` to select the number of user-preferred categories. Within these chosen categories, `n` items are extracted to form the final candidate set.
* **LLM-based Ranking with Projectors**: Similar to the LlamaRec architecture, we pass both LRURec's item and category representations through an MLP projector, allowing the LLM to deeply understand both item-level and category-level characteristics simultaneously.
* **Superior N-Diversity Performance**: Achieves state-of-the-art performance on the *N-Diversity* dataset (recommending categories the user has never interacted with), outperforming all baseline models including generic and diversity-specific recommenders.

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
