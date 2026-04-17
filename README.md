# RobustPrompt: Achieving Semantic Invariance in Enterprise LLMs

This project addresses a critical challenge in enterprise AI systems: **syntactic sensitivity** in Large Language Models (LLMs). LLMs often generate vastly different answers for semantically identical queries depending on phrasing, tone, slang, or typos.

In enterprise domains such as HR automation, inconsistent responses can create loss of employee trust, compliance risks or unpredictable AI behavior

This project introduces **RobustPrompt**, an end-to-end system that combines **RAG (Retrieval-Augmented Generation)** for factual grounding and **rDPO (Robust Direct Preference Optimization)** for behavioral alignment. 

**The Goal:** Achieve *semantic invariance* i.e ensuring that all paraphrases of a query produce the exact same stable, professional answer.

---

## 📖 Project Overview

Enterprise assistants must handle messy, real-world queries. Traditional LLM systems fail because:
* **Vanilla LLMs** → Change personas and give inconsistent facts.
* **Standard RAG** → Fixes the facts, but *not* the behavioral inconsistency (tone still shifts).

| Formal Query | Real User Query (Messy) |
| :--- | :--- |
| *"What is my remaining sick leave balance?"* | *"yo how many sickdays i got left??"* |

Our solution trains the model using **adversarial red-teaming data** and **preference optimization** to enforce a rigid, professional behavior across all syntactic paraphrases.

---

## ⚙️ Key Components

### 1. Agentic Adversarial Dataset Generation
An automated red-teaming agent generates training data by:
1. Sampling HR policy intents.
2. Degraded queries using slang, typos, grammar errors, and informal language.
3. Sending degraded queries to the baseline model to collect unstable, sensitive responses.

Each example is packaged into a contrastive preference pair:
* **Prompt:** The messy user query
* **Chosen:** The correct, perfectly formatted HR policy answer
* **Rejected:** The unstable or hallucinated response from the baseline model

### 2. rDPO Training with QLoRA
We explicitly aligned the model's latent space to punish syntactic sensitivity using:
* `Llama-3-8B-Instruct`
* **QLoRA** (4-bit quantization for memory efficiency)
* **Direct Preference Optimization (rDPO)**

**Advantages over RLHF/SFT:** No separate reward model required, highly stable training, lower compute costs, and it teaches the model what *not* to do (unlike standard SFT).

### 3. Retrieval-Augmented Generation (RAG)
To prevent the model from hallucinating facts, all responses are grounded using:
* **FAISS** vector search (Exact Euclidean Distance `IndexFlatL2`)
* Dense sentence embeddings (`all-MiniLM-L6-v2`)
* A domain-specific HR policy knowledge base

---

## 🏗️ Final Architecture

The final system successfully decouples and solves both requirements for Enterprise AI: **behavioral alignment** and **factual grounding**.

```text
[ Messy User Query ] 
        ↓
[ Vector Search (FAISS) ] ➔ Retrieves exact HR Policy Context
        ↓
[ Fine-tuned Llama-3 (rDPO + LoRA) ] ➔ Filters syntax, aligns behavior
        ↓
[ Stable, Professional & Grounded Response ]
```

---

## 📊 Experimental Results

We conducted a full-spectrum ablation study to isolate the improvements of our methodology.

| Model Configuration | Factual Accuracy | Semantic Stability (BERTScore F1) | Latency Overhead |
| :--- | :--- | :--- | :--- |
| **Vanilla Llama-3-8B** | Low | 88.91% | Baseline |
| **RAG Only (FAISS)** | High | 93.73% | Baseline |
| **rDPO Only (No RAG)** | Low *(Hallucinates)* | 90.75% | Baseline |
| **RobustPrompt (RAG + rDPO)** | **High (90.0%)** | **95.89%** | **Zero overhead** |

> **Key Finding:** RAG fixes the facts. DPO fixes the behavior. Combining both solves the enterprise problem entirely without introducing the inference latency of an LLM router.

---

## 📂 Repository Structure

```text
RobustPrompt/
│
├── Agentic_Dataset_Generator.ipynb          # Generates adversarial       
queries via red-teaming
├── Dataset_Generator_Notebook.ipynb         # Creates the final rDPO dataset (Chosen/Rejected)
├── Training_Pipeline_Notebook.ipynb         # Fine-tunes Llama-3-8B using rDPO + QLoRA
├── Baseline_Testing(LLM+RAG).ipynb          # Evaluates baseline model and standard RAG
├── Testing_Pipeline(Adapter+RAG)_Notebook.ipynb # Tests the fine-tuned rDPO model with RAG
├── UI_Notebook.ipynb                        # Interactive Gradio UI to test the HR assistant
│
├── hr_policies.txt                          # HR policy knowledge base used for FAISS retrieval
├── hr_evaluation_clusters.json              # HR intent clusters used for BERTScore evaluation
└── rdpo_dataset_final.json                  # Final generated dataset for rDPO training
```

---

## 🗄️ Dataset Details

The custom dataset contains 50 enterprise HR intents covering: Leave management, Payroll, Benefits, IT policies, Compliance, and Onboarding.

**Format Example:**
```json
{
  "prompt": "yo how many sickdays i got left??",
  "chosen": "You can check your remaining sick leave balance in the HR portal...",
  "rejected": "To calculate your leave balance follow these steps..."
}
```

---

## 🔬 Evaluation Method & Ablation Study

Traditional metrics like BLEU or Exact Match cannot measure true response consistency. We utilized **pairwise BERTScore (F1)** to measure the cosine similarity between responses across paraphrases. A higher score guarantees higher semantic invariance.

### The "Mode Collapse" Insight (Ablation Study)
When we tested **DPO without RAG**, the model successfully learned the correct professional HR persona, but it hallucinated facts to maintain that persona (e.g., *“According to Marriott HR policy...”*). 
**Conclusion:** Behavioral alignment (DPO) and factual grounding (RAG) are complementary; neither is sufficient on its own.

---

## 🚀 How to Run

1. **Generate Dataset:**
   Run `Agentic_Dataset_Generator.ipynb` to create `rdpo_dataset_final.json`.
2. **Train the Model:**
   Run `Training_Pipeline_Notebook.ipynb` to train the Llama-3-8B model using QLoRA and DPOTrainer.
3. **Run Baseline Experiments:**
   Run `Baseline_Testing(LLM+RAG).ipynb` to get stability metrics for the Vanilla and RAG baselines.
4. **Test the Final System:**
   Run `Testing_Pipeline(Adapter+RAG)_Notebook.ipynb` to evaluate the full RobustPrompt pipeline.
5. **Launch the Demo:**
   Run `UI_Notebook.ipynb` to launch the interactive Gradio web interface and compare the models live.

---

## 🛠️ Technologies Used
* **Python**
* **Hugging Face Transformers**
* **TRL** (`DPOTrainer`)
* **PEFT** (LoRA / QLoRA)
* **FAISS** (Vector Database)
* **Sentence Transformers** (`all-MiniLM-L6-v2`, `roberta-large` for BERTScore)
* **Meta Llama-3-8B-Instruct**

---
## Model in Action

<div align="center">

<div style="display:flex; gap:10px;">
  <img src="https://github.com/user-attachments/assets/5d4041ab-7a63-42b6-aaee-ce447f0d5d58" width="32%">
  <img src="https://github.com/user-attachments/assets/0ba96209-bb79-4df1-b489-e94efd4587f5" width="32%">
  <img src="https://github.com/user-attachments/assets/6b10fb95-daaf-45bc-a773-5d293639b768" width="32%">
</div>

<br>

<div style="display:flex; gap:10px; justify-content:center;">
  <img src="https://github.com/user-attachments/assets/f12914ab-f5a1-4aec-9f08-09e1148fae82" width="48%">
  <img src="https://github.com/user-attachments/assets/8f16075a-8c80-4fbf-bfce-efdcc7672f6c" width="48%">
</div>

</div>


