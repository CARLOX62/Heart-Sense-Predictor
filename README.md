# ❤️ LifeLine – Heart Failure Prediction App

**LifeLine** is an intelligent web application that predicts the risk of heart failure based on patient health metrics. Built with Flask and powered by machine learning, it helps healthcare professionals make faster and more informed decisions.

![LifeLine Screenshot](https://github.com/CARLOX62/Heart-Sense-Predictor/blob/main/Screenshot%20(115).png)

## 🔗 Live Demo

🚀 [Click here to try LifeLine](https://lifeline-predictor.onrender.com)  

---

## 📋 Features

- 🔍 **Heart Failure Risk Prediction** – Based on clinical parameters like age, blood pressure, ejection fraction, etc.
- 🧠 **Random Forest Classifier** – Pre-trained model with ~89% accuracy.
- 📊 **Probability Output** – Shows confidence levels for predictions.
- 📁 **Downloadable Report** – Users can export the prediction results.
- 📈 **Dataset Preview** – View the dataset that powers the prediction.
- ✅ **Input Validation** – Handles common input errors cleanly.
- 🖥️ **Simple & Responsive UI** – Built with HTML and CSS for ease of use.

---

## 🛠️ Tech Stack

| Component        | Technology                 |
|------------------|-----------------------------|
| Frontend         | HTML, CSS                  |
| Backend          | Python, Flask              |
| Machine Learning | scikit-learn, SMOTE, XGBoost |
| Data Processing  | Pandas, NumPy              |
| Model Persistence| joblib                     |
| Visualization    | Matplotlib (Jupyter)       |

---

## 📁 Project Structure

```
MultipleFiles/
├── app.py                          # Main Flask application
├── index.html                      # Frontend form template
├── best_heart_failure_model.pkl    # Trained Random Forest model
├── scaler.pkl                      # StandardScaler used for feature scaling
├── heart_failure_clinical_records_dataset (1).csv  # Training data
├── heart.ipynb                     # Model training and evaluation notebook
├── requirements.txt                # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### ✅ Prerequisites
- Python 3.7+
- pip (Python package installer)

### 📦 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/lifeline-heart-failure.git
cd lifeline-heart-failure/MultipleFiles

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# OR
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Visit `https://lifeline-predictor.onrender.com` in your browser.

---

## 🧪 Usage Guide

1. Open the app in your browser.
2. Fill out patient details in the prediction form:
   - Age
   - Anaemia (0 = No, 1 = Yes)
   - Creatinine Phosphokinase
   - Diabetes (0 = No, 1 = Yes)
   - Ejection Fraction
   - High Blood Pressure (0 = No, 1 = Yes)
   - Platelets
   - Serum Creatinine
   - Serum Sodium
   - Sex (0 = Female, 1 = Male)
   - Smoking (0 = No, 1 = Yes)
   - Time (Follow-up duration in days)
3. Click **Predict**.
4. View the **prediction result** and download the **prediction report**.

---

## 🤖 Machine Learning Pipeline

- **Dataset**: Heart Failure Clinical Records Dataset (299 samples, 13 features)
- **Preprocessing**:
  - No missing values
  - Feature scaling with `StandardScaler`
  - Class balancing using `SMOTE`
- **Model Training**:
  - Evaluated Logistic Regression, SVM, XGBoost, Random Forest
  - **Best Model**: Random Forest Classifier (~89% accuracy)
- **Model Persistence**: Stored using `joblib`
- **Evaluation**:
  - Accuracy, Confusion Matrix, Classification Report
  - Feature importance visualized in Jupyter Notebook

---

## 🌍 Deployment

To deploy on [Render](https://render.com):

1. Push code to GitHub.
2. Create a new Web Service in Render.
3. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
4. Add environment if needed.
5. Done! 🎉

---

## 🤝 Contributing

We welcome all contributions — whether it's fixing a bug, improving UI, or suggesting a new feature.

```bash
# Fork the repo
# Create a branch: git checkout -b feature-name
# Make changes, commit and push
# Submit a Pull Request
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

**Developer**: Aniket Kumar  
**Email**: aniketkumarsonu62@gmail.com  
**GitHub**: [@CARLOX62](https://github.com/CARLOX62)

---

> “Prevention is better than cure. LifeLine helps predict the unseen.”
