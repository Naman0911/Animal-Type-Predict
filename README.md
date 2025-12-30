# Animal Prediction Using Naive Bayes

This project implements a **Naive Bayes machine learning algorithm** to predict animal categories using the **Zoo Dataset**.  
The model achieves an accuracy of **97%**, demonstrating that the dataset is highly suitable for probabilistic classification.

---

## 📌 Project Overview

The goal of this project is to classify animals into different categories based on their biological characteristics using **Bernoulli Naive Bayes**.

The project covers:
- Data preprocessing and feature selection
- Dropping non-informative identifier columns
- Training a Naive Bayes classifier
- Model evaluation and interpretation

---

## 🧬 Dataset Description

The Zoo dataset consists of animals described by **binary biological features**, such as:

- hair
- feathers
- eggs
- milk
- airborne
- aquatic
- predator
- toothed
- backbone
- breathes
- venomous
- fins
- legs
- tail
- domestic
- catsize

### ⚠️ Important Note on `animal_name`

The `animal_name` column is **removed before training** because:
- It is an identifier, not a predictive feature
- Encoding it would introduce **data leakage**
- It does not generalize to unseen data

Dropping this column ensures a **fair and valid evaluation**.

---

## 🧠 Model Used

- **Algorithm:** Bernoulli Naive Bayes  
- **Reason:** Best suited for binary feature vectors  
- **Learning Type:** Supervised classification  

---

## 📈 Results

The trained Naive Bayes model achieved:

- ✅ **Accuracy: 97%**
- Strong precision and recall across most animal classes
- Consistent performance on unseen test data

---

## 📊 Evaluation Metrics

```text
Accuracy: 97%
```

---

## 🧪 Sample Code Snippet

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

df = df.drop(columns=['animal_name'])

X = df.drop('class_type', axis=1)
y = df['class_type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = BernoulliNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 📁 Project Structure

```
Animal-Prediction/
├── dataset/
├── notebooks/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📜 License

MIT License

---

## 👤 Author

**Naman Upadhyay**  
GitHub: https://github.com/Naman0911
