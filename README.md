# Customer Feedback Trend Analysis System

## Overview
The **Customer Feedback Trend Analysis System** is a Machine Learning–based project designed to analyze and classify customer reviews into **positive, neutral, and negative sentiments**. The system leverages **Natural Language Processing (NLP)** techniques and multiple machine learning algorithms to extract meaningful insights from textual data.

It also provides an interactive **Flask-based web application** where users can input text, use voice input, or upload bulk data for analysis.

---

## Features
- Sentiment classification based on ratings:
  - Negative (≤ 2)
  - Neutral (= 3)
  - Positive (≥ 4)
- Data preprocessing and cleaning
- Handling missing values
- Exploratory Data Analysis (EDA)
- Data visualization using graphs
- WordCloud generation for different sentiments
- Multiple ML models for comparison
- Model evaluation using standard metrics
- Web interface with:
  - Text input
  - Voice input
  - Bulk CSV upload
- Download results in CSV and PDF formats
- Text-to-speech output for predictions

---

## 🧠 Methodology

### 1. Data Preprocessing
- Loaded dataset using Pandas  
- Inspected data using `.info()` and `.describe()`  
- Handled missing values using `isnull().sum()`  
- Removed noisy/unnecessary text  

### 2. Sentiment Labeling
- Ratings mapped to sentiment categories:
  - ≤ 2 → Negative  
  - 3 → Neutral  
  - ≥ 4 → Positive  

### 3. Exploratory Data Analysis
- Sentiment distribution  
- Rating distribution  
- Sentiment vs rating comparison  
- Category-wise analysis  

### 4. Text Processing (NLP)
- Cleaned textual data  
- Applied **TF-IDF Vectorization**  

### 5. Model Training
- Multinomial Naive Bayes  
- Logistic Regression  
- Decision Tree  
- Random Forest  

### 6. Model Evaluation
- Accuracy Score  
- Classification Report  
- Confusion Matrix  

### 7. Prediction System
- Predicts sentiment for:
  - Manual text input  
  - Voice input  
  - Bulk CSV data  

---

## Web Application

The trained model is deployed using **Flask** with the following functionalities:

- **Text Input:** Enter a review and get instant prediction  
- **Voice Input:** Speak a review for analysis  
- **Bulk Upload:** Upload CSV file for large-scale sentiment analysis  
- **Download Options:** Export results as CSV or PDF  
- **Voice Output:** System speaks the prediction result  

---

## 🛠️ Technologies Used
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- WordCloud  
- Flask  
- SpeechRecognition  
- pyttsx3  
- ReportLab  

---

## 📦 Installation

```bash
git clone https://github.com/devanshg-20/Customer-Feedback-Trend-Analysis-System.git
cd Customer-Feedback-Trend-Analysis-System
pip install -r requirements.txt

python app.py

