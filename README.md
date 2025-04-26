 Fake News Detection Using Hybrid CNN and BiLSTM

![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen)

## Project Overview

This project presents a robust deep learning model designed to classify news articles as **fake** or **real** with high accuracy. Leveraging a hybrid architecture combining **Convolutional Neural Networks (CNN)** and **Bidirectional Long Short-Term Memory (BiLSTM)** networks, the model captures both local features and long-range dependencies in textual data, achieving an impressive accuracy of **99%** on benchmark datasets.

## Motivation

The rapid spread of misinformation and fake news on social media and online platforms poses serious challenges to society, affecting public opinion, health, and democracy. Automated and reliable fake news detection systems are essential to mitigate these impacts and ensure the dissemination of trustworthy information.

## Key Features

- **Hybrid Model Architecture:** Combines CNN for extracting local textual patterns and BiLSTM for capturing contextual dependencies in both forward and backward directions.
- **High Accuracy:** Achieves 99% accuracy on the test dataset, outperforming many existing models.
- **Preprocessing Pipeline:** Includes text cleaning, tokenization, and embedding layer integration.
- **End-to-End Solution:** From data preprocessing to model training, evaluation, and prediction.
- **Scalable and Extensible:** Easily adaptable to different datasets and languages.

## Model Architecture

1. **Embedding Layer:** Converts words into dense vector representations.
2. **Convolutional Neural Network (CNN):** Extracts local n-gram features from the embedded text.
3. **Bidirectional LSTM (BiLSTM):** Captures long-term dependencies and context from both past and future tokens.
4. **Fully Connected Layers:** For classification into fake or real news.
5. **Output Layer:** Sigmoid activation for binary classification.

## Dataset

The model was trained and evaluated on publicly available datasets containing labeled news articles as fake or real. Typical datasets include:

- /kaggle/input/fake-and-real-news-dataset)
*

## Installation

1. Clone the repository:

## Technologies Used

- Python 3.x
- TensorFlow / Keras or PyTorch (specify which one you used)
- NumPy, Pandas, Scikit-learn
- NLTK / SpaCy (for text preprocessing)

## Future Work

- Extend the model to support multilingual fake news detection.
- Incorporate transformer-based architectures (e.g., BERT) for improved context understanding.
- Develop a web or mobile application for real-time fake news detection.
- Explore explainability techniques to interpret model predictions.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.


## Contact

For questions or collaboration, please contact:

- **Arshad Iqbal**  
- Email: arshadktk.uop@gmail.com  


---

Thank you for checking out this project! Together, let's fight misinformation and promote truth.


