import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(df):
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna('', inplace=True)

    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y, df

def evaluate_model(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
    for train_idx, test_idx in kf.split(X):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        scores['accuracy'].append(accuracy_score(y[test_idx], y_pred))
        scores['precision'].append(precision_score(y[test_idx], y_pred, average='weighted', zero_division=0))
        scores['recall'].append(recall_score(y[test_idx], y_pred, average='weighted', zero_division=0))
        scores['f1_score'].append(f1_score(y[test_idx], y_pred, average='weighted'))

    return {metric: round(np.mean(values) * 100, 2) for metric, values in scores.items()}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "No file uploaded!"

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return "Invalid file type. Only CSV files are allowed!"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    df = pd.read_csv(filepath)
    X, y, processed_df = preprocess_data(df)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    results = [{'model': name, **evaluate_model(model, X, y)} for name, model in models.items()]
    
    dataset_info = {'shape': processed_df.shape}
    chart_data = {'models': [r['model'] for r in results], 'accuracies': [r['accuracy'] for r in results]}

    return render_template('results.html', results=results, dataset_info=dataset_info, chart_data=chart_data)

if __name__ == '__main__':
    app.run(debug=True)
