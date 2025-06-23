# Shakespeare Text Completion using LSTM

A deep learning project that generates Shakespearean text completions using Long Short-Term Memory (LSTM) neural networks. This project trains a bidirectional LSTM model on Shakespeare's plays to predict and complete text in the style of Shakespeare.

![Shakespeare](https://img.shields.io/badge/Shakespeare-Text%20Generation-8b4513)
![LSTM](https://img.shields.io/badge/LSTM-Neural%20Network-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Google Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-yellow)

## 🎭 Project Overview

This project implements a sophisticated text completion system trained on Shakespeare's complete works. The model can:
- Predict the next words in a given text sequence
- Generate text completions in Shakespearean style
- Provide probability distributions for multiple word predictions
- Offer an interactive web interface for real-time text generation

## ✨ Features

- **Bidirectional LSTM Architecture**: Uses multiple LSTM layers for better context understanding
- **Interactive Web Interface**: Beautiful Shakespeare-themed UI for text generation
- **Customizable Parameters**: Adjustable temperature and word count for generation
- **Pre-trained Examples**: Common Shakespearean phrases for quick testing
- **Probability Visualization**: Shows prediction confidence with visual bars
- **Real-time Prediction**: Instant text completion as you type

## 🏗️ Model Architecture

The LSTM model consists of:
- **Embedding Layer**: Word embeddings for text representation
- **3 Bidirectional LSTM Layers**: 
  - 256 units (with return sequences)
  - 192 units (with return sequences) 
  - 128 units (final layer)
- **Dropout Layers**: Regularization to prevent overfitting
- **Dense Output Layer**: Softmax activation for word probability distribution

## 📊 Dataset

- **Source**: Shakespeare's complete plays from Kaggle
- **Dataset**: `kingburrito666/shakespeare-plays`
- **Content**: All lines from Shakespeare's plays
- **Preprocessing**: Text cleaning, tokenization, and sequence generation
- **Vocabulary**: Specialized Shakespeare words preserved (thee, thou, thy, etc.)

## 🚀 Getting Started

### Prerequisites

```bash
# Required Python packages
tensorflow>=2.8.0
numpy
pandas
matplotlib
seaborn
scikit-learn
kagglehub
ipywidgets
tqdm
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/shakespeare-text-completion-lstm.git
cd shakespeare-text-completion-lstm
```



2. **Open in Google Colab**
- Upload the notebook to Google Colab
- Ensure T4 GPU is enabled for training
- Run all cells to train the model

### Hardware Requirements

- **Recommended**: Google Colab with T4 GPU
- **RAM**: Minimum 12GB
- **Training Time**: ~2-3 hours with T4 GPU
- **Model Size**: ~50MB when saved

## 💻 Usage

### Training the Model

```python
# The notebook automatically:
# 1. Downloads Shakespeare dataset from Kaggle
# 2. Preprocesses text data
# 3. Creates training sequences
# 4. Trains bidirectional LSTM model
# 5. Saves model and tokenizer
```

### Text Generation

```python
# Use the interactive interface or call directly:
text_input = "To be or not to"
next_words, predictions = predict_next_words(
    model, tokenizer, text_input,
    num_words=5, temperature=0.8
)
```

### Interactive Interface

The notebook provides a beautiful web interface with:
- Text input area
- Example Shakespearean phrases
- Temperature control (0.1 - 1.5)
- Word count selection (1 - 20)
- Real-time prediction with probability bars

## 📈 Model Performance

- **Training Epochs**: ~15-20 epochs
- **Final Validation Accuracy**: ~8.9%
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam with learning rate scheduling
- **Callbacks**: Early stopping, model checkpointing, learning rate reduction

## 🎨 Sample Outputs

**Input**: "To be or not to"
**Output**: "be a king s a"

**Input**: "All the world's a"
**Output**: "stage and all the"

**Input**: "Friends Romans countrymen lend me"
**Output**: "your ears i come"

## 📁 File Structure

```
shakespeare-text-completion-lstm/
│
├── Word Completion using LSTM.ipynb    # Main notebook
├── Word Completion using LSTM.pdf      # Project documentation
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
└── models/                            # Saved models (after training)
    ├── shakespeare_word_model.keras   # Trained LSTM model
    └── tokenizer.pickle              # Text tokenizer
```

## 🔧 Technical Details

### Data Preprocessing
- Text cleaning and normalization
- Tokenization with vocabulary filtering
- Sequence generation with sliding windows
- One-hot encoding for categorical labels

### Model Training
- Batch size: 128
- Sequence length: Variable
- Vocabulary size: ~8000 unique words
- Training/Validation split: 80/20

### Prediction Algorithm
- Temperature-based sampling
- Top-k word selection
- Probability normalization
- Sequential word generation

## 🙏 Acknowledgments

- **Shakespeare's Works**: Public domain texts
- **Kaggle Dataset**: `kingburrito666/shakespeare-plays`
- **TensorFlow**: For the deep learning framework
- **Google Colab**: For providing T4 GPU access
- **Course**: Generative AI Assignment 1

## 🔮 Future Enhancements

- [ ] Character-level LSTM for more granular control
- [ ] Transformer architecture implementation
- [ ] Multiple author style transfer
- [ ] Fine-tuning on specific Shakespeare plays
- [ ] Web deployment with Flask/FastAPI
- [ ] Mobile app development

---

*"All the world's a stage, and all the men and women merely players."* - Shakespeare 
