# Deep Learning Approaches for Vietnamese Student Feedback Classification

This project benchmarks five deep learning architectures for automated sentiment classification of Vietnamese student course-evaluation feedback across four classes: Positive, Negative, Suggestion, and Neutral.

## 🎯 Project Objectives

* Compare the in-domain performance of KimCNN, BiLSTM+Attention, RCNN, Transformer Encoder, and PhoBERT on a self-collected Vietnamese corpus.
* Analyze the contribution of the Attention mechanism via ablation study.
* Evaluate model robustness under character-level noise (typos and diacritic removal).
* Establish a reproducible benchmark pipeline for Vietnamese NLP text classification.

## 📊 Key Results

* All five models achieved perfect in-domain classification (Accuracy = 1.000, Macro-F1 = 1.000).
* Removing Attention from BiLSTM causes Macro-F1 to collapse from 1.000 → 0.103.
* KimCNN is more robust under typos (Accuracy 0.950 at Typo-10%) compared to BiLSTM+Attention (0.862).
* Both models fail completely under full diacritic removal (Accuracy ≈ 0.0).

## 📁 Project Structure

* `NLP_Final_.ipynb`: Main notebook — training, evaluation, ablation, and robustness analysis.
* `student_feedback.csv`: Self-collected dataset of 2,050 Vietnamese student feedback samples.
* `label_guideline.pdf`: Annotation guideline defining 4 labels + inter-annotator agreement (Cohen's κ = 0.73).
* `outputs/`: Generated results including:
  * `results_table.csv`: Full results table (models × metrics).
  * `cm_phobert.png`, `cm_bilstm.png`, etc.: Confusion matrices for all models.
  * `ablation.png`: Ablation study chart.
  * `robustness.png`: Robustness analysis chart.
  * `model_comparison.png`: Cross-model performance comparison.
  * `history_*.png`: Training and validation curves per model.

## 🛠️ Installation and Usage

**1. Environment Setup**

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch transformers underthesea scikit-learn pandas numpy matplotlib seaborn
```

**Hardware:** NVIDIA T4 GPU (Google Colab free tier)  
**Python version:** 3.10+

**2. Run the Notebook**

Upload `NLP_Final_.ipynb` and `student_feedback.csv` to Google Colab, then select **Runtime → Run all**.

Or run locally:

```bash
jupyter notebook NLP_Final_.ipynb
```

**3. Random Seeds**

All experiments use fixed seeds for full reproducibility:

```python
SEEDS = [42, 123, 777]
```

## 📊 Dataset

* **Size:** 2,050 Vietnamese student course-evaluation texts.
* **Labels:** Positive (530), Negative (520), Suggestion (500), Neutral (500).
* **Split:** 70% train / 10% val / 20% test (stratified).
* **Inter-annotator agreement:** Cohen's κ = 0.73 (Substantial Agreement).

> ⚠️ All personally identifiable information has been removed from the dataset.

## 👤 Author

**Minh Nguyen Ngoc** · MSSV: 22010704  
Phenikaa University, Ha Noi, Vietnam  
Advisor: ThS. Vũ Hoàng Diệu  
Submission date: February 28, 2026
