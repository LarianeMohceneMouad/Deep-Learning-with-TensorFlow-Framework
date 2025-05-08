# Deep Learning with TensorFlow Framework

A comprehensive hands-on guide to building and understanding deep learning models using **TensorFlow** and **Keras**, covering everything from the basics of machine learning to advanced architectures like CNNs, RNNs, LSTMs, and hybrid models. This repository includes practical implementations for image classification, time-series forecasting, NLP tasks, and more.

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [What You'll Learn](#what-youll-learn)
3. [Directory Structure](#directory-structure)
4. [Technologies Used](#technologies-used)
5. [Installation & Setup](#installation--setup)
6. [Usage Instructions](#usage-instructions)
7. [Projects Included](#projects-included)
8. [Notebooks Overview](#notebooks-overview)
9. [Contributing](#contributing)
10. [License](#license)

---

## 📘 Introduction

This project is based on the **Deep Learning with TensorFlow Framework** tutorial that walks you through the fundamentals and advanced topics in deep learning using **TensorFlow 2.x** and **Keras**. Each concept is backed by a fully working Jupyter Notebook implementation, allowing you to learn by doing.

You’ll explore:
- Building neural networks from scratch
- Training models on real-world datasets (MNIST, Fashion MNIST, IMDB, etc.)
- Visualizing training with **TensorBoard**
- Implementing **CNNs**, **RNNs**, and **LSTMs**
- Creating **word embeddings** and visualizing them
- Preprocessing data and evaluating model performance
- Advanced techniques like **hybrid models**, **regularization**, and **hyperparameter tuning**

---

## 🧠 What You'll Learn

By completing this course, you will be able to:

| Category | Skills |
|---------|--------|
| **Basics** | Understand supervised vs unsupervised learning, scalars/vectors/matrices |
| **Neural Networks** | Build FCNs, CNNs, RNNs, LSTMs, and hybrid models |
| **NLP Tasks** | Tokenize text, build Word2Vec models, visualize word embeddings |
| **Time Series** | Predict stock prices using sequence modeling |
| **Computer Vision** | Classify images using CNNs and TensorBoard visualization |
| **Data Handling** | Normalize, preprocess, and pipeline data using TFDS |
| **Model Evaluation** | Track metrics, plot predictions, compute accuracy and loss |
| **Advanced Concepts** | Use dropout, early stopping, BPTT, hyperparameter tuning |

---

## 🗂️ Directory Structure

```
deep-learning-tensorflow/
│
├── Parts/                  # All Jupyter Notebook cells
│   ├── 01_Introduction_to_TensorBoard
│   ├── 02_Machine_Learning_Basics
│   ├── 03_Neural_Network_Fundamentals
│   ├── 04_Convolutional_Neural_Networks
│   ├── 05_Recurrent_Neural_Networks_and_LSTM
│   ├── 06_Word_Embeddings_and_NLP_Tasks
│   ├── 07_Data_Preprocessing_Pipeline
│   ├── 08_Model_Training_and_Evaluation
│   ├── 09_Advanced_Techniques_and_Hybrid_Models
│   └── 10_Applications_and_Projects
│
├── logs/                       # TensorBoard logs
├── datasets/                   # Downloaded datasets (optional)
├── utils/                      # Helper scripts
│   └── helpers.py              # Utility functions
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## ⚙️ Technologies Used

| Tool / Library | Purpose |
|---------------|---------|
| **TensorFlow / Keras** | Building and training deep learning models |
| **NumPy / Pandas** | Data manipulation and numerical operations |
| **Matplotlib / Seaborn** | Visualization of results |
| **NLTK** | Natural Language Processing (tokenization) |
| **Scikit-learn** | Preprocessing and evaluation |
| **yfinance** | Stock price data fetching |
| **t-SNE** | Dimensionality reduction for embedding visualization |
| **TensorBoard** | Model visualization and training tracking |

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- `pip` or `conda`

### Steps

```bash
# Clone the repo
git clone https://github.com/yourusername/deep-learning-tensorflow.git
cd deep-learning-tensorflow

# Install dependencies
pip install -r requirements.txt
```

### Optional: Start TensorBoard

After running any notebook that uses TensorBoard logging:

```bash
tensorboard --logdir=./logs
```

Then open your browser at: http://localhost:6006

---

## 📊 Usage Instructions

Each notebook is self-contained and follows a consistent format:

1. **Imports and Setup**
2. **Dataset Loading**
3. **Preprocessing**
4. **Model Building**
5. **Training**
6. **Evaluation**
7. **Visualization (with TensorBoard)**

You can run each notebook independently or sequentially to build up knowledge step-by-step.

---

## 📁 Projects Included

| Project | Description |
|--------|-------------|
| **House Price Prediction** | Regression using California housing dataset |
| **Fashion MNIST Classifier** | Image classification using CNN |
| **Stock Price Forecasting** | Time series prediction using LSTM |
| **IMDB Sentiment Analysis** | Text classification using RNN/LSTM |
| **Word Embedding Visualization** | Train Word2Vec model and visualize with t-SNE |
| **Multi-output Regression** | Predict multiple outputs using Linnerud dataset |
| **Hybrid CNN + LSTM Network** | Combine convolution and recurrence for complex sequences |

---

## 📝 Notebooks Overview

| Cell | Topic |
|---------|-------|
| `01_Introduction_to_TensorBoard` | Visualizing computation graphs, metrics, parameters, and embeddings |
| `02_Machine_Learning_Basics` | Supervised vs unsupervised learning, regression/classification |
| `03_Neural_Network_Fundamentals` | Dense layers, activation functions, compiling and training models |
| `04_Convolutional_Neural_Networks` | CNN components, architecture design, image classification |
| `05_Recurrent_Neural_Networks_and_LSTM` | RNNs, LSTMs, GRUs, sequence prediction, time-series forecasting |
| `06_Word_Embeddings_and_NLP_Tasks` | Skip-Gram Word2Vec, similarity search, t-SNE visualization |
| `07_Data_Preprocessing_Pipeline` | Normalization, one-hot encoding, train/test split, TFDS |
| `08_Model_Training_and_Evaluation` | Metrics tracking, callbacks, plotting true vs predicted values |
| `09_Advanced_Techniques_and_Hybrid_Models` | Dropout, early stopping, CNN-LSTM hybrids, BPTT |
| `10_Applications_and_Projects` | Real-world applications across domains: vision, finance, NLP |

---

## 🤝 Contributing

Contributions are welcome! If you'd like to add new notebooks, improve documentation, or fix bugs, feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📣 Feedback & Support

If you found this helpful or have suggestions, please consider giving it a star on GitHub or reaching out via email/social media.

Let me know if you’d like me to generate:
- A GitHub-ready `.md` file
- A downloadable ZIP bundle of all notebooks
- A sales page or course description for selling these notebooks

Happy learning! 🚀
