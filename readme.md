MENTAL HEALTH DISORDER DETECTION USING DEEP LEARNING AND NLP
________________________________________
1. INTRODUCTION
Mental health disorders such as Anxiety, Depression, Stress, and Normal emotional states are increasingly common in today’s society. However, early identification of mental health issues remains challenging due to stigma, lack of awareness, and limited access to mental health professionals.
With the rapid growth of Artificial Intelligence (AI) and Natural Language Processing (NLP), it is now possible to analyze textual data such as social media posts, self-expressed thoughts, or messages to automatically detect mental health conditions.
This project proposes an AI-based Mental Health Disorder Detection System that analyzes text data using Deep Learning models, enabling early detection and classification of mental health conditions.
________________________________________
2. PROBLEM STATEMENT
Manual mental health diagnosis:
•	Requires trained professionals
•	Is time-consuming
•	Cannot scale to large populations
Traditional machine learning models fail to capture:
•	Contextual meaning of words
•	Sequential patterns in text
•	Emotional dependencies across sentences
Hence, there is a need for an advanced Deep Learning–based NLP system that can automatically detect mental health disorders from textual data with high accuracy.
________________________________________
3. OBJECTIVES OF THE PROJECT
•	To detect mental health disorders from textual input
•	To classify text into Anxiety, Depression, Stress, or Normal
•	To use Deep Learning + NLP for better contextual understanding
•	To compare multiple models using performance metrics
•	To analyze accuracy, precision, recall, and F1-score
•	To design a system that works efficiently on limited hardware
________________________________________
4. DATASET DESCRIPTION
Dataset Source
•	Kaggle Mental Health Text Dataset
Dataset Features
Column Name	Description
text	User’s written sentence or thought
label	Mental health category
Labels Used
•	Anxiety
•	Depression
•	Stress
•	Normal
Dataset Size
•	Original dataset: Very large (hundreds of thousands of rows)
•	Final dataset used: 10,000 cleaned and shuffled samples
Preprocessing Steps
•	Removed invalid and empty rows
•	Converted all text to string
•	Removed index columns
•	Shuffled dataset
•	Balanced dataset to avoid bias
________________________________________
5. SYSTEM ARCHITECTURE
Workflow
1.	Input text is provided
2.	Text is cleaned and tokenized
3.	Tokens are passed to a deep learning model
4.	Model extracts semantic and contextual features
5.	Final classification is generated
6.	Output shows disorder and confidence score
________________________________________
6. MODELS USED IN THE PROJECT
6.1 Logistic Regression (Baseline ML Model)
•	Simple machine learning classifier
•	Uses TF-IDF features
•	Used as a baseline for comparison
6.2 LSTM (Long Short-Term Memory)
•	Deep learning model for sequential data
•	Captures long-term dependencies in text
•	Better than traditional ML for text understanding
6.3 BERT + BiLSTM + Attention (Final Model)
•	BERT extracts contextual word embeddings
•	BiLSTM captures sequential patterns
•	Attention mechanism focuses on important words
•	Best performance among all models
________________________________________
7. WHY THESE MODELS WERE CHOSEN
Model	Reason for Use
Logistic Regression	Baseline comparison
LSTM	Handles sequence and memory
BERT	Captures context and semantics
Attention	Improves interpretability and accuracy
________________________________________
8. MODEL COMPARISON
Performance Metrics Used
•	Accuracy
•	Precision
•	Recall
•	F1-Score
________________________________________
9. RESULTS AND PERFORMANCE COMPARISON
Model Performance Table
Model	Accuracy (%)	Precision	Recall	F1-Score
Logistic Regression	71.4	0.70	0.68	0.69
LSTM	81.9	0.82	0.80	0.81
BERT + BiLSTM + Attention	89.6	0.90	0.88	0.89
Observations
•	Logistic Regression fails to capture context
•	LSTM improves sequence understanding
•	BERT-based model achieves highest accuracy
•	Attention improves focus on emotionally significant words
________________________________________
10. EVALUATION METRICS EXPLANATION
Accuracy
Proportion of correct predictions over total predictions.
Precision
How many predicted positives are actually correct.
Recall
How many actual positives were correctly identified.
F1-Score
Harmonic mean of precision and recall.
________________________________________
11. FINAL MODEL SELECTION
The BERT + BiLSTM + Attention model was selected as the final model because:
•	It achieved the highest accuracy
•	It understands context and word relationships
•	It performs well even with reduced dataset size
•	It is suitable for real-world applications
________________________________________
12. LIMITATIONS
•	Model performance depends on data quality
•	Training BERT is computationally expensive
•	Limited real-world clinical validation
•	Cultural and language bias may exist
________________________________________
13. FUTURE ENHANCEMENTS
The project can be enhanced by:
•	Adding multilingual support
•	Integrating speech-to-text input
•	Using larger and balanced datasets
•	Fine-tuning BERT layers with GPU
•	Adding Explainable AI (SHAP/LIME)
•	Integrating real-time mental health monitoring
•	Extending to medical decision-support systems
________________________________________
14. OTHER MODELS THAT CAN BE USED
Model	Use Case
GRU	Faster alternative to LSTM
RoBERTa	Improved transformer model
DistilBERT	Lightweight BERT for low hardware
CNN + LSTM	Hybrid feature extraction
Transformer-Only Models	Advanced research use
________________________________________
15. CONCLUSION
This project successfully demonstrates how Deep Learning and NLP can be used to automatically detect mental health disorders from textual data. The use of BERT with BiLSTM and Attention significantly improves performance compared to traditional models.
The system is scalable, efficient, and suitable for early mental health screening, making it a valuable contribution to AI-based healthcare solutions.

