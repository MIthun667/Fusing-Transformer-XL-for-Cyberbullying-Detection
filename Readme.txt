Algorithms and Code
    Identifying cyberbullying in languages other than English, particularly in Bengali, presents distinct challenges due to linguistic subtleties 
    and a lack of annotated datasets. In this study, we propose a novel method to detect cyberbullying in Bengali text using a Kaggle dataset. Our model, 
    named Fusion Transformer-XL, integrates Transformer-(Extra Large) XL with Bi-directional Gated Recurrent Units (BiGRU) and Bi-directional 
    Long Short-Term Memory (BiLSTM) networks to capture both long-range dependencies and contextual nuances effectively.

Model Architecture
    1. Transformer-XL: This component allows for capturing long-range dependencies in the text data, making it effective for lengthy conversations or comments, typical in social media posts.
    2. BiGRU-BiLSTM: The bi-directional layers enable the model to process information from both past and future contexts, further enhancing the semantic understanding of the text.
    3. The combination of Transformer-XL and BiGRU-BiLSTM enables our Fusion Transformer-XL model to outperform baseline models, achieving an accuracy of 98.17% and an F1-score of 98.18%.
Data Preprocessing
    1. Data Cleaning and Label Encoding: We performed extensive cleaning, including removing irrelevant symbols, stopwords, and performing label encoding of the categorical outputs.
    2. Handling Imbalanced Classes: To deal with class imbalance, we used upsampling techniques to ensure that all classes are adequately represented during training.
    3. Data Augmentation: We also applied augmentation techniques to artificially increase the training data, which helped improve the model's generalization capabilities.
Tokenization
    We employed a pre-trained tokenizer to convert the raw Bengali text into token representations. The tokenizer is fine-tuned to capture semantic nuances in the Bengali 
    language effectively.
Model Training and Evaluation
    1. The model is implemented using Python 3.11.9 and TensorFlow 2.15.0, with Adam optimization and early stopping regularization applied to fine-tune the hyperparameters.
    2. We used k-fold cross-validation to ensure the robustness and generalization of the model across various subsets of the data.
    3. The evaluation metrics tracked include accuracy, precision, recall, and F1-score to assess the model's performance.
    4. LIME (Local Interpretable Model-Agnostic Explanations) was used to generate interpretability insights for the model, explaining its predictions in a human-understandable way.
    5. We also conducted cross-dataset evaluation using an English cyberbullying detection dataset to assess the model's reliability and versatility across different languages.

Computational Resources
    The model was trained on an AMD Ryzen CPU, 16GB of RAM (clock speed: 3200Hz), and an RTX GeForce 2060 GPU with 12GB of memory. The training time and efficiency were monitored to 
    evaluate the practicality of deploying the model in real-world applications.

Hyperparameter Tuning
We optimized the following hyperparameters during training:
    1. The number of Transformer-XL blocks is 2 and attention heads is 2.
    2. The size of the GRU/LSTM hidden layers is 64.
    3. Optimizer is Adam.
    4. We found that a specific combination of these parameters produced the best results in terms of the accuracy 
    and F1-score, particularly for detecting cyberbullying in a resource-limited language like Bengali.

Reproducibility and Code Availability
    The code used for this study, including the implementation of the Fusion Transformer-XL model, data preprocessing scripts, and the training pipeline, 
    is available in a GitHub repository. It includes a detailed README file explaining how to set up the environment, run the code, and reproduce the results.

GitHub Repository: [https://github.com/MIthun667/Fusing-Transformer-XL-for-Cyberbullying-Detection/tree/main]
Datasets:
    1. Bengali Dataset: https://www.kaggle.com/datasets/cypher1337/dataset-for-cyberbully-detection-bengali-comments
    2. English Dataset: https://www.kaggle.com/datasets/momo12341234/cyberbully-detection-dataset