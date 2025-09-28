#  French Sentiment Analysis with DistilBERT Fine-tuning on Allociné Dataset


A complete pipeline for fine-tuning DistilBERT on French movie reviews sentiment analysis using the renowned Allociné dataset. This project demonstrates exceptional performance improvements through transfer learning, achieving 83% accuracy with only 2 minutes of training.

##  Key Results

###  Performance Metrics


| Metric | Baseline | Fine-tuned | **Improvement** |
|--------|----------|------------|-----------------|
| **Accuracy** | 53.8% | **83.0%** | **+29.5 points** |
| **F1-Score** | 0.4460 | **0.827** | **+38.0 points** |
| **Precision** | 0.5970 | **0.858** | **+26.1 points** |
| **Recall** | 0.5350 | **0.830** | **+29.5 points** |



##  About the Allociné Dataset

This project uses the [Allociné dataset](https://huggingface.co/datasets/allocine), a reference benchmark for French sentiment analysis:

- **200k French movie reviews** (2006-2020)
- **Source**: Allociné.fr user community
- **Classes**: Binary (Negative: 0, Positive: 1)
- **Our sample**: 2,000 training / 400 test examples
- **Academic reference**: [Théophile Blard (2020)](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert)

##  Project Architecture

```
Allociné Dataset (200k reviews) → Stratified Sampling (2k train)
    ↓
Baseline DistilBERT Evaluation (50% accuracy)
    ↓
Fine-tuning with Optimized Hyperparameters
    ↓
Final Model (79% accuracy) + Comprehensive Analysis
```

##  Project Structure

```
french-sentiment-analysis-allocine/
├──   notebooks/                    # Step-by-step Jupyter notebooks
│   ├── 01_dataset_preparation.ipynb    # Load & explore Allociné dataset
│   ├── 02_baseline_evaluation.ipynb    # Evaluate pre-trained model
│   └── 03_model_fine_tuning.ipynb     # Fine-tune DistilBERT
├──  data/                         # Data samples
└──  results/                      # Metrics and analysis
```

##  Quick Start

### Prerequisites

- Python 3.8+
- GPU recommended (CUDA)
- Google Colab (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/rachigh/french-sentiment-analysis-allocine.git
cd french-sentiment-analysis-allocine

# Install dependencies
pip install -r requirements.txt
```

### Usage

1. **Open notebooks in Google Colab** (recommended for GPU access)
2. **Run sequentially**:
   - `01_dataset_preparation.ipynb` → Load Allociné dataset
   - `02_baseline_evaluation.ipynb` → Baseline performance
   - `03_model_fine_tuning.ipynb` → Fine-tuning process

### Quick Inference Example

```python
from transformers import pipeline

# Load fine-tuned model
classifier = pipeline(
    "text-classification",
    model="./fine_tuned_model_allocine",
    return_all_scores=True
)

# Analyze sentiment
result = classifier("Ce film est absolument magnifique ! Une œuvre d'art.")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.892}, {'label': 'NEGATIVE', 'score': 0.108}]
```

##  Methodology & Technical Details

###  Fine-tuning Configuration

- **Base Model**: `distilbert-base-multilingual-cased`
- **Training Data**: 2,000 stratified samples
- **Epochs**: 3
- **Learning Rate**: 2e-5
- **Batch Size**: 32 (effective, with gradient accumulation)
- **Optimizations**: FP16, warmup, weight decay

###  Key Findings

**Strengths Identified:**
-  **Massive improvement**: 58% relative accuracy gain
-  **Efficient training**: 40.4 accuracy points per minute
-  **Well-calibrated**: No high-confidence errors
-  **Reproducible**: Standardized dataset and methodology

**Limitations Identified:**
-  **Limited training data**: 2k vs 160k available samples
-  **Model choice**: DistilBERT vs specialized French models
- **Performance gap**: 18.4 points below state-of-the-art

###  Recommendations for Further Improvement

1. **Model Upgrade**: Switch to CamemBERT (+5-10% expected)
2. **Data Scaling**: Use full 160k dataset
3. **Advanced Techniques**: Ensemble methods, data augmentation
4. **Target Performance**: 90-96% accuracy achievable


##  Technologies Used

- **Transformers**: State-of-the-art NLP models
- **PyTorch**: Deep learning framework
- **Datasets**: Efficient data handling
- **Scikit-learn**: Evaluation metrics
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Visualizations
- **Google Colab**: GPU acceleration

##  References & Resources

- **Dataset**: [Allociné on Hugging Face](https://huggingface.co/datasets/allocine)
- **Original Paper**: [Théophile Blard (2020) - French sentiment analysis with BERT](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert)
- **SOTA Model**: [tf-allocine](https://huggingface.co/tblard/tf-allocine)
- **Documentation**: [Transformers Fine-tuning Guide](https://huggingface.co/docs/transformers/training)

