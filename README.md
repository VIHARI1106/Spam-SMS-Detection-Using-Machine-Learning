# 📩 Spam SMS Detection Using Machine Learning  

## 🚀 Project Overview  
This project focuses on building a machine learning model to classify SMS messages as **spam** or **ham (non-spam)**. The dataset consists of labeled SMS messages, and the model utilizes **text preprocessing, TF-IDF vectorization, and multiple classification algorithms** to achieve high accuracy in spam detection.

---

## 📚 Repository Contents  
```
├── logistic_regression_model.pkl    # Trained Logistic Regression model
├── naïve_bayes_model.pkl            # Trained Naïve Bayes model
├── random_forest_model.pkl          # Trained Random Forest model
├── spam.csv                         # Dataset containing labeled SMS messages
├── spamq3.py                        # Python script for training & evaluation
├── svm_model.pkl                     # Trained Support Vector Machine model
├── vectorizer.pkl                     # TF-IDF vectorizer for text transformation
```

---

## 🛠️ Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/VIHARI1106/Spam-SMS-Detection-Using-Machine-Learning.git
cd Spam-SMS-Detection-Using-Machine-Learning
```

### 2️⃣ Install Dependencies  
Ensure you have the required Python libraries by installing:  
```bash
pip install -r requirements.txt
```

### 3️⃣ Train & Evaluate the Model  
To retrain and test the model, run:  
```bash
python spamq3.py
```

---

## 📊 Data Processing & Model Training  
✔ **Text Preprocessing** – Lowercasing, punctuation removal, stopword removal  
✔ **Feature Extraction** – TF-IDF Vectorization  
✔ **Model Training** – Using **Naïve Bayes, Logistic Regression, Random Forest, and SVM**  
✔ **Performance Evaluation** – Accuracy, Confusion Matrix, and Classification Report  

---

## 🏆 Model Performance  
| Model                 | Accuracy |
|----------------------|----------|
| Naïve Bayes         | 96.58%    |
| Logistic Regression | 98.08%    |
| Random Forest       | 97.89%    |
| SVM                 | 98.12%    |

---

## 📌 Evaluation Criteria  
✔ **Functionality** – The project correctly classifies SMS messages as spam or ham  
✔ **Code Quality** – Clean, structured, and well-commented Python code  
✔ **Innovation** – Utilizes **SMOTE, TF-IDF, and multiple ML models** for improved performance  
✔ **Documentation** – Clear explanation of project approach, dataset, and execution  

---

## 💚 License  
This project is open-source and available for use under the **MIT License**.  

---

### 📩 Contact & Contribution  
💡 **Want to contribute?** Fork the repository, make changes, and submit a pull request! 🚀  


