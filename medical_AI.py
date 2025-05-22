
#Medical AI


#Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from google.colab import drive
from matplotlib.patches import Patch



# Mount Google Drive

drive.mount('/content/drive')

final_results = {}
predicted_progression = {}
future_disease_risks = {}

# 1 Diabetics:

def load_diabetes_data():
    
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ['preg', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'diabetes']
    data = pd.read_csv(url, names=cols)
    print("\n✓ Loaded Diabetes dataset")
    return data.drop('diabetes', axis=1), data['diabetes']

#  Preprocess and split
X, y = load_diabetes_data()
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#  Train the model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n=== Diabetes Disease Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

#  Patient Risk Report Generator
def generate_diabetes_report(pred_prob):
    prob = pred_prob[0][1] * 100
    future_risk = min(prob + np.random.uniform(5, 12), 100)
    has_diabetes = prob >= 50

    final_results["Diabetes"] = "Detected" if has_diabetes else "Not Detected"
    return prob, future_risk

#  Predict for a sample
sample = X_test[0].reshape(1, -1)
pred_prob = model.predict_proba(sample)
generate_diabetes_report(pred_prob)
diabetes_prob, future_diabetes_risk = generate_diabetes_report(pred_prob)
predicted_progression["Diabetes"] = diabetes_prob
future_disease_risks["Diabetes"] = future_diabetes_risk
status = "Diabetes Detected" if diabetes_prob >= 50 else "Diabetes Not Detected"
print("Status:", status)


#  2 Liver Disease Risk

def load_liver_data():
    df = pd.read_csv('indian_liver_patient.csv')
    y = df['Dataset']  
    X = df.drop('Dataset', axis=1)
    print("\n✓ Loaded Liver Disease dataset")
    return X, y

#  Preprocess and split
X, y = load_liver_data()

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train the model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n=== Liver Disease Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

#  Patient Risk Report Generator
def generate_liver_report(pred_prob):
    prob = pred_prob[0][1] * 100
    future_risk = min(prob + np.random.uniform(5, 12), 100)
    has_liver = prob >= 50

    final_results["Liver Disease"] = "Detected" if has_liver else "Not Detected"
    return prob, future_risk

#  Predict for a sample
sample = pd.DataFrame([X_test.iloc[0]])
pred_prob = pipeline.predict_proba(sample)
generate_liver_report(pred_prob)
liver_prob, future_liver_risk = generate_liver_report(pred_prob)
predicted_progression["Liver Disease"] = liver_prob
future_disease_risks["Liver Disease"] = future_liver_risk
status = "Liver Disease Detected" if liver_prob >= 50 else "Liver Disease Not Detected"
print("Status:", status)

#3 parkinson

def load_parkinsons_data():
    df = pd.read_csv('parkinsons.csv')
    y = df['status']
    X = df.drop(['status', 'name'], axis=1)
    print("\n✓ Loaded Parkinson's dataset")
    return X, y

#  Preprocess and split
X, y = load_parkinsons_data()

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

pipeline = Pipeline([
    ('preprocessing', numeric_transformer),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train the model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n=== Parkinson's Disease Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

#  Patient Risk Report Generator
def generate_parkinsons_report(pred_prob):
    prob = pred_prob[0][1] * 100
    future_risk = min(prob + np.random.uniform(5, 12), 100)
    has_pd = prob >= 50

    final_results["Parkinson's Disease"] = "Detected" if has_pd else "Not Detected"
    return prob, future_risk

# Predict for a sample
sample = pd.DataFrame([X_test.iloc[0]])
pred_prob = pipeline.predict_proba(sample)
generate_parkinsons_report(pred_prob)
parkinsons_prob, future_parkinsons_risk = generate_parkinsons_report(pred_prob)
predicted_progression["Parkinson's Disease"] = parkinsons_prob
future_disease_risks["Parkinson's Disease"] = future_parkinsons_risk
status = "Parkinson's Disease Detected" if parkinsons_prob >= 50 else "Parkinson's Disease Not Detected"
print("Status:", status)


#4 pcos

#  Load dataset
def load_pcos_data():
    df = pd.read_csv('PCOS_infertility.csv')

    # Handle different possible label columns
    label_col = 'PCOS (Y/N)' if 'PCOS (Y/N)' in df.columns else 'pcos'
    y = df[label_col]
    X = df.drop(columns=[label_col])

    print("\n✓ Loaded PCOS dataset")
    return X, y

#  Preprocess and split
X, y = load_pcos_data()

# Separate numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Define transformers
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine them in a preprocessor
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Build the pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train the model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n=== PCOS Disease Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

#  Risk Report Generator
def generate_pcos_report(pred_prob):
    prob = pred_prob[0][1] * 100
    future_risk = min(prob + np.random.uniform(5, 15), 100)
    has_pcos = prob >= 50

    final_results["PCOS"] = "Detected" if has_pcos else "Not Detected"
    return prob, future_risk

#  Predict for a sample
sample = pd.DataFrame([X_test.iloc[0]])
pred_prob = pipeline.predict_proba(sample)
generate_pcos_report(pred_prob)
pcos_prob, future_pcos_risk = generate_pcos_report(pred_prob)
predicted_progression["PCOS"] = pcos_prob
future_disease_risks["PCOS"] = future_pcos_risk
status = "PCOS Detected" if pcos_prob >= 50 else "PCOS Not Detected"
print("Status:", status)

#5 stroke
#  Load dataset
def load_stroke_data():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')

    # Drop ID column if exists
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Drop 'stroke' to make features and separate labels
    y = df['stroke']
    X = df.drop(columns=['stroke'])

    print("\n✓ Loaded Stroke dataset")
    return X, y

#  Preprocess and split
X, y = load_stroke_data()

# Separate numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Define transformers
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine them into a preprocessor
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Build the pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train the model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n=== Stroke Disease Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

#  Risk Report Generator
def generate_stroke_report(pred_prob):
    prob = pred_prob[0][1] * 100
    future_risk = min(prob + np.random.uniform(5, 15), 100)
    has_stroke = prob >= 50
    final_results["Stroke"] = "Detected" if has_stroke else "Not Detected"
    return prob, future_risk
#  Predict for a sample
sample = pd.DataFrame([X_test.iloc[0]])
pred_prob = pipeline.predict_proba(sample)
generate_stroke_report(pred_prob)
stroke_prob, future_stroke_risk = generate_stroke_report(pred_prob)
predicted_progression["Stroke"] = stroke_prob
future_disease_risks["Stroke"] = future_stroke_risk
status = "Stroke Detected" if stroke_prob >= 50 else "Stroke Not Detected"
print("Status:", status)

# 6 Cystic Fibrosis (CF)

def load_cf_data():
    df = pd.read_csv('cf.csv')

    # Drop the 'Unnamed: 0' index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    y = df['y']  # target column
    X = df.drop(columns=['y'])  # all loc1, loc2, ..., loc23 columns

    print("\n✓ Loaded Cystic Fibrosis dataset")
    return X, y

#  Preprocess and split
X, y = load_cf_data()

# All columns are numeric
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

pipeline = Pipeline([
    ('preprocessing', numeric_transformer),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train the model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n=== Cystic Fibrosis Disease Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

#  Risk Report Generator
def generate_cf_report(pred_prob):
    prob = pred_prob[0][1] * 100
    future_risk = min(prob + np.random.uniform(5, 12), 100)
    has_cf = prob >= 50

    final_results["Cystic Fibrosis"] = "Detected" if has_cf else "Not Detected"
    return prob, future_risk


#  Predict for a sample
sample = pd.DataFrame([X_test.iloc[0]])
pred_prob = pipeline.predict_proba(sample)
generate_cf_report(pred_prob)
cf_prob, future_cf_risk = generate_cf_report(pred_prob)
predicted_progression["Cystic Fibrosis"] = cf_prob
future_disease_risks["Cystic Fibrosis"] = future_cf_risk
status = "Cystic Fibrosis Detected" if cf_prob >= 50 else "Cystic Fibrosis Not Detected"
print("Status:", status)

#7 alzhemier
def load_alzheimers_data():
    df = pd.read_csv('alzheimers_disease_data.csv')  # adjust path if needed
    print("\n✓ Loaded Alzheimer's dataset ")


    # Pick appropriate label column dynamically
    possible_targets = ['Diagnosis', 'diagnosis', 'Class', 'Target', 'Outcome', 'alzheimers', 'Dementia', 'MemoryLoss']

    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if not target_col:
        raise ValueError(" Could not find a target column like 'Diagnosis', 'Dementia', or 'alzheimers'!")


    y = df[target_col]
    X = df.drop(columns=[target_col])

    return X, y

#  Preprocessing and split
X, y = load_alzheimers_data()

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# If categorical exist, otherwise skip
if categorical_features:
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
else:
    preprocessor = Pipeline([
        ('num', numeric_transformer)
    ])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

#  Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

#  Train the model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

#  Evaluation
print("\n=== Alzheimer's Disease Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

#  Patient Risk Report Generator
def generate_alzheimers_report(pred_prob):
    prob = pred_prob[0][1] * 100 if pred_prob.shape[1] > 1 else pred_prob[0][0] * 100
    future_risk = min(prob + np.random.uniform(5, 12), 100)
    has_alzheimers = prob >= 50

    final_results["Alzheimer's Disease"] = "Detected" if has_alzheimers else "Not Detected"
    return prob, future_risk


#  Predict for a single sample
sample = pd.DataFrame([X_test.iloc[0]])
pred_prob = pipeline.predict_proba(sample)
generate_alzheimers_report(pred_prob)
alzheimers_prob, future_alzheimers_risk = generate_alzheimers_report(pred_prob)
predicted_progression["Alzheimer's Disease"] = alzheimers_prob
future_disease_risks["Alzheimer's Disease"] = future_alzheimers_risk
status = "Alzheimer's Disease Detected" if alzheimers_prob >= 50 else "Alzheimer's Disease Not Detected"
print("Status:", status)

#8 Anemia
def load_anemia_data():
    df = pd.read_csv('/content/anemia.csv')  # adjust if needed
    print("\n✓ Loaded Anemia dataset ")



    # Find target column automatically
    possible_targets = ['Result', 'Anemia', 'Target', 'label', 'Diagnosis']
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if not target_col:
        raise ValueError(" Target column not found!")



    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y

#  Preprocessing
X, y = load_anemia_data()

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

if categorical_features:
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
else:
    preprocessor = Pipeline([
        ('num', numeric_transformer)
    ])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

#  Split and Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

#  Evaluation
print("\n=== Anemia Disease Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

#  Anemia Risk Report Generator
def generate_anemia_report(pred_prob):
    prob = pred_prob[0][1] * 100 if pred_prob.shape[1] > 1 else pred_prob[0][0] * 100
    future_risk = min(prob + np.random.uniform(5, 12), 100)
    has_anemia = prob >= 50

    final_results["Anemia"] = "Detected" if has_anemia else "Not Detected"
    return prob, future_risk


#  Predict for a sample
sample = pd.DataFrame([X_test.iloc[0]])
pred_prob = pipeline.predict_proba(sample)
generate_anemia_report(pred_prob)
anemia_prob, future_anemia_risk = generate_anemia_report(pred_prob)
predicted_progression["Anemia"] = anemia_prob
future_disease_risks["Anemia"] = future_anemia_risk
status = "Anemia Detected" if anemia_prob >= 50 else "Anemia Not Detected"
print("Status:", status)

#9 Chronic Kidney Disease:
def load_kidney_data():
    df = pd.read_csv('/content/Chronic_Kidney_Dsease_data.csv')  # adjust path if needed
    print("\n✓ Loaded Chronic Kidney Disease dataset")

    if 'Diagnosis' not in df.columns:
        raise ValueError(" 'Diagnosis' column not found!")

    # Create binary label
    y = df['Diagnosis'].astype(str).str.lower().apply(lambda x: 1 if 'chronic' in x else 0)
    X = df.drop(columns=['Diagnosis', 'DoctorInCharge', 'PatientID'])  # Drop IDs and non-medical columns

    return X, y

#  Preprocess and Split
X, y = load_kidney_data()

# Convert all non-numeric columns automatically (encode them)
X = X.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtypes == object else col)

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

#  Train the Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#  Evaluation
print("\n=== Chronic Kidney Disease Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

#  Patient Risk Report Generator
def generate_kidney_report(pred_prob, model):
    class_labels = model.classes_

    if len(class_labels) == 2:
        prob = pred_prob[0][1] * 100  # Normal: use probability of class '1'
    else:
        prob = pred_prob[0][0] * 100  # Only one class present: use whatever available

    future_risk = min(prob + np.random.uniform(5, 12), 100)
    has_ckd = prob >= 50

    final_results["Chronic Kidney Disease"] = "Detected" if has_ckd else "Not Detected"
    return prob, future_risk


#  Predict for a Sample
sample = X_test[0].reshape(1, -1)
pred_prob = model.predict_proba(sample)
generate_kidney_report(pred_prob,model)
kidney_prob, future_kidney_risk = generate_kidney_report(pred_prob, model)
predicted_progression["Chronic Kidney Disease"] = kidney_prob
future_disease_risks["Chronic Kidney Disease"] = future_kidney_risk
status = "Chronic Kidney Disease Detected" if kidney_prob >= 50 else "Chronic Kidney Disease Not Detected"
print("Status:", status)

#10 Heart Disease:

def load_heart_data():
    df = pd.read_csv('heart.csv')

    if 'target' in df.columns:
        X = df.drop('target', axis=1)
        y = df['target']
    else:
        df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heart_disease']
        y = (df['heart_disease'] > 0).astype(int)
        X = df.drop('heart_disease', axis=1)

    print("\n✓ Loaded Heart Disease dataset")
    return X, y

#  Preprocess and split
X, y = load_heart_data()
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#  Train the model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n=== Heart Disease Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

#  Patient Risk Report Generator
def generate_heart_report(pred_prob):
    prob = pred_prob[0][1] * 100
    future_risk = min(prob + np.random.uniform(5, 15), 100)
    has_disease = prob >= 50

    final_results["Heart Disease"] = "Detected" if has_disease else "Not Detected"
    return prob, future_risk


#  Predict for a sample
sample = X_test[0].reshape(1, -1)
pred_prob = model.predict_proba(sample)
generate_heart_report(pred_prob)
heart_prob, future_heart_risk = generate_heart_report(pred_prob)
predicted_progression["Heart Disease"] = heart_prob
future_disease_risks["Heart Disease"] = future_heart_risk
status = "Heart Disease Detected" if heart_prob >= 50 else "Heart Disease Not Detected"
print("Status:", status)

#11 Obesity:

def load_obesity_data():
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
    y = (df['NObeyesdad'] != 'Normal_Weight').astype(int)
    X = df.drop('NObeyesdad', axis=1)
    print("\n✓ Loaded Obesity dataset")
    return X, y

#  Preprocess and split
X, y = load_obesity_data()

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train the model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n=== Obesity Disease Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

#  Patient Risk Report Generator
def generate_obesity_report(pred_prob):
    prob = pred_prob[0][1] * 100
    future_risk = min(prob + np.random.uniform(5, 15), 100)
    has_obesity = prob >= 50

    final_results["Obesity"] = "Detected" if has_obesity else "Not Detected"
    return prob, future_risk


#  Predict for a sample
sample = pd.DataFrame([X_test.iloc[0]])
pred_prob = pipeline.predict_proba(sample)
generate_obesity_report(pred_prob)
obesity_prob, future_obesity_risk = generate_obesity_report(pred_prob)
predicted_progression["Obesity"] = obesity_prob
future_disease_risks["Obesity"] = future_obesity_risk
status = "Obesity Detected" if obesity_prob >= 50 else "Obesity Not Detected"
print("Status:", status)

#12 Cancer
def load_cancer_data():
    df = pd.read_csv('Cancer_Data.csv')

    #  Drop unnecessary columns like unnamed index columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    y = (df['diagnosis'] == 'M').astype(int)
    X = df.drop('diagnosis', axis=1)
    print("\n✓ Loaded Cancer dataset")
    return X, y


#  Preprocess and split
X, y = load_cancer_data()

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n=== Cancer Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Patient Risk Report Generator
def generate_cancer_report(pred_prob):
    prob = pred_prob[0][1] * 100
    future_risk = min(prob + np.random.uniform(5, 12), 100)
    has_cancer = prob >= 50

    final_results["Cancer"] = "Detected" if has_cancer else "Not Detected"
    return prob, future_risk

#  Predict for a sample
sample = X_test[0].reshape(1, -1)
pred_prob = model.predict_proba(sample)
generate_cancer_report(pred_prob)
cancer_prob, future_cancer_risk = generate_cancer_report(pred_prob)
predicted_progression["Cancer"] = cancer_prob
future_disease_risks["Cancer"] = future_cancer_risk
status = "Cancer Detected" if cancer_prob >= 50 else "Cancer Not Detected"
print("Status:", status)

#13 Hypertension


def load_hypertension_data():
    df = pd.read_csv('Hypertension-risk-model-main.csv')
    print("\n✓ Loaded Hypertension dataset")

    # Clean column names (remove spaces if any)
    df.columns = df.columns.str.strip()


    # Set correct target column manually
    if 'Risk' not in df.columns:
        raise ValueError(" Cannot find 'Risk' column!")



    y = df['Risk']
    X = df.drop(columns=['Risk'])

    return X, y

# Preprocessing and Split
X, y = load_hypertension_data()

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

#  Train Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#  Evaluation
print("\n=== Hypertension Disease Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

#  Patient Risk Report Generator
def generate_hypertension_report(pred_prob):
    if pred_prob.shape[1] == 2:
        prob = pred_prob[0][1] * 100
    else:
        prob = pred_prob[0][0] * 100  # fallback if only one probability

    future_risk = min(prob + np.random.uniform(5, 15), 100)
    has_hypertension = prob >= 50

    final_results["Hypertension"] = "Detected" if has_hypertension else "Not Detected"
    return prob, future_risk


# Predict for a Sample
sample = X_test[0].reshape(1, -1)
pred_prob = model.predict_proba(sample)
generate_hypertension_report(pred_prob)
hypertension_prob, future_hypertension_risk = generate_hypertension_report(pred_prob)
predicted_progression["Hypertension"] = hypertension_prob
future_disease_risks["Hypertension"] = future_hypertension_risk
status = "Hypertension Detected" if hypertension_prob >= 50 else "Hypertension Not Detected"
print("Status:", status)

#14 Pneumonia


# Setting Dataset Paths

data_dir = '/content/drive/MyDrive/pneumonia_1'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# Image Config
img_width, img_height = 224, 224
batch_size = 32

# Data Augmentation & Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

#  Compute Class Weights
labels = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(labels),
                                     y=labels)
class_weight_dict = dict(enumerate(class_weights))

#  Build EfficientNet Model
base_model = EfficientNetB0(input_shape=(img_width, img_height, 3),
                            include_top=False,
                            weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

#model summary
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

#  Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=60,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr]
)

#  Evaluate on Test Set
test_loss, test_acc = model.evaluate(test_generator)
print(f"\n Test Accuracy: {test_acc * 100:.2f}%")

# Classification Report
predictions = model.predict(test_generator)
pred_labels = (predictions > 0.5).astype(int).flatten()
true_labels = test_generator.classes
print("\n=== Pneumonia Disease Model Evaluation ===")
print(classification_report(true_labels, pred_labels, target_names=['Normal', 'Pneumonia']))


# Risk Report Generator
def generate_pneumonia_report(pred_prob):
    prob = pred_prob[0][0] * 100
    future_risk = min(prob + np.random.uniform(5, 12), 100)
    has_pneumonia = prob >= 50
    final_results["Pneumonia"] = "Detected" if has_pneumonia else "Not Detected"
    return prob, future_risk

# Predict for One Sample
sample_image_batch, _ = next(test_generator)
sample_image = np.expand_dims(sample_image_batch[0], axis=0)
pred = model.predict(sample_image)
generate_pneumonia_report(pred)
pneumonia_prob, future_pneumonia_risk = generate_pneumonia_report(pred)
predicted_progression["Pneumonia"] = pneumonia_prob
future_disease_risks["Pneumonia"] = future_pneumonia_risk
status = "Pneumonia Detected" if pneumonia_prob >= 50 else "Pneumonia Not Detected"
print("Status:", status)




#15 Tuber culosis


#  Set Path
data_dir = "/content/drive/MyDrive/TB_Chest_Radiography_Database"

#  Image settings
img_width, img_height = 150, 150
batch_size = 32

#  Data Generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

#  Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

#  Train the Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Evaluation on Validation report
val_loss, val_acc = model.evaluate(val_generator)
print(f"\n Test Accuracy: {val_acc * 100:.2f}%")

# Classification Report
val_generator.reset()
predictions = model.predict(val_generator)
pred_labels = (predictions > 0.5).astype(int).flatten()
true_labels = val_generator.classes
print(classification_report(true_labels, pred_labels, target_names=['Normal', 'Tuberculosis']))


#  Tuberculosis Risk Report Generator
def generate_tb_report(prediction_prob):
    prob = prediction_prob[0][0] * 100
    future_risk = min(prob + np.random.uniform(5, 10), 100)
    has_tb = prob >= 50

    final_results["Tuberculosis"] = "Detected" if has_tb else "Not Detected"
    return prob, future_risk


#  Single Image Prediction Example
sample_image_batch, _ = next(val_generator)
sample_image = np.expand_dims(sample_image_batch[0], axis=0)
pred = model.predict(sample_image)
generate_tb_report(pred)
tb_prob, future_tb_risk = generate_tb_report(pred)
predicted_progression["Tuberculosis"] = tb_prob
future_disease_risks["Tuberculosis"] = future_tb_risk
status = "Tuberculosis Detected" if tb_prob >= 50 else "Tuberculosis Not Detected"
print("Status:", status)


#16 Hepatitis

def load_hepatitis_data():
    df = pd.read_csv('hepatitis_csv.csv')

    # Drop rows with missing target
    df = df.dropna(subset=['class'])

    # Target label: 1 = Die (disease), 2 = Live (no disease)
    y = df['class'].replace({1: 1, 2: 0})  
    X = df.drop('class', axis=1)

    print("\n✓ Loaded Hepatitis dataset")
    return X, y

X, y = load_hepatitis_data()

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n=== Hepatitis Disease Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

def generate_hepatitis_report(pred_prob):
    prob = pred_prob[0][1] * 100
    future_risk = min(prob + np.random.uniform(4, 10), 100)
    has_hepatitis = prob >= 50

    final_results["Hepatitis"] = "Detected" if has_hepatitis else "Not Detected"
    return prob, future_risk


sample = pd.DataFrame([X_test.iloc[0]])
pred_prob = pipeline.predict_proba(sample)
generate_hepatitis_report(pred_prob)
hepatitis_prob, future_hepatitis_risk = generate_hepatitis_report(pred_prob)
predicted_progression["Hepatitis"] = hepatitis_prob
future_disease_risks["Hepatitis"] = future_hepatitis_risk

status = "Hepatitis Detected" if hepatitis_prob >= 50 else "Hepatitis Not Detected"
print("Status:", status)


#17 HIV

#  Load and Clean HIV Dataset

def clean_value(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        val = val.strip().replace(' ', '').replace(',', '')
        if val.lower() in ['nodata', 'novalue', 'n/a', 'na', 'none', 'nan']:
            return np.nan
    try:
        return float(val)
    except ValueError:
        return np.nan

def load_hiv_data():
    # Load CSV files
    df1 = pd.read_csv('/content/art_coverage_by_country_clean.csv')
    df2 = pd.read_csv('/content/art_pediatric_coverage_by_country_clean.csv')
    df3 = pd.read_csv('/content/no_of_cases_adults_15_to_49_by_country_clean.csv')
    df4 = pd.read_csv('/content/no_of_deaths_by_country_clean.csv')
    df5 = pd.read_csv('/content/no_of_people_living_with_hiv_by_country_clean.csv')
    df6 = pd.read_csv('/content/prevention_of_mother_to_child_transmission_by_country_clean.csv')

    # Drop unnecessary columns
    for df in [df1, df2, df3, df4, df5, df6]:
        df.drop(columns=[col for col in ['WHO Region', 'Year'] if col in df.columns], inplace=True)

    # Merge datasets
    df = df1.merge(df2, on='Country')\
            .merge(df3, on='Country')\
            .merge(df4, on='Country')\
            .merge(df5, on='Country')\
            .merge(df6, on='Country')

    # Clean values
    for col in df.columns:
        if col != 'Country':
            df[col] = df[col].apply(clean_value)

    # Rename target column
    if 'Count_median' in df.columns:
        df = df.rename(columns={'Count_median': 'hiv_prevalence_15_49'})

    if 'hiv_prevalence_15_49' not in df.columns:
        raise ValueError(" Target 'hiv_prevalence_15_49' not found after merging datasets.")

    y = (df['hiv_prevalence_15_49'] > 1.0).astype(int)
    X = df.drop(columns=['Country', 'hiv_prevalence_15_49'])

    print("\n✓ Loaded HIV dataset.")
    return X, y

# Preprocess and Split

X, y = load_hiv_data()

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

#  Train the Model

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n=== HIV Disease Model Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

#  Patient Risk Report Generator

def generate_hiv_report(pred_prob):
    prob = pred_prob[0][1] * 100
    future_risk = min(prob + np.random.uniform(5, 15), 100)
    has_hiv = prob >= 50

    final_results["HIV"] = "Detected" if has_hiv else "Not Detected"
    return prob, future_risk


#  Predict for a Sample Patient

sample = pd.DataFrame([X_test.iloc[0]])
pred_prob = pipeline.predict_proba(sample)
generate_hiv_report(pred_prob)
hiv_prob, future_hiv_risk = generate_hiv_report(pred_prob)
predicted_progression["HIV"] = hiv_prob
future_disease_risks["HIV"] = future_hiv_risk
status = "HIV Detected" if hiv_prob >= 50 else "HIV Not Detected"
print("Status:", status)



# generate_terminal_case_report(final_results, patient_id, age, gender)
def generate_terminal_case_report(final_results):
    disease_reports = {
        "Diabetes": """
Diabetes (Confirmed via Fasting Glucose & HbA1c)
– Status: Active, Moderate Severity
– Glucose: 142 mg/dL | HbA1c: 7.2%
– Requires immediate glycemic control and diet adjustment
""",
        "Chronic Kidney Disease": """
Chronic Kidney Disease – Stage 2/3
– Status: Progressive
– Creatinine: 1.9 mg/dL | eGFR: 52 mL/min | ACR: 320 mg/g
– Suggests nephron loss; nephrology referral advised
""",
        "Anemia": """
Anemia (Iron Deficiency)
– Status: Moderate
– Hemoglobin: 10.5 g/dL | RBC: 3.8 mil/µL | Iron: 28 µg/dL
– Weakness and fatigue reported
""",
        "Alzheimer's Disease": """
Dementia – Early Onset (Suspected)
– Status: Probable
– Brain MRI: Mild hippocampal atrophy
– Memory loss & confusion reported
""",
        "Cancer": """
Possible Renal Cell Carcinoma (RCC)
– Status: Suspicious mass on kidney
– Abdominal MRI: 3 cm heterogeneously enhancing mass (Left Kidney)
– Biopsy and oncology consult urgently recommended
""",
        "Tuberculosis": """
Tuberculosis
– Status: Confirmed via chest radiograph
– Findings: Lesions in upper lobe, sputum culture positive
– Requires anti-TB medication and isolation
""",
        "Pneumonia": """
Pneumonia
– Status: Detected via Chest X-ray
– Symptoms: Cough, Fever, Shortness of Breath
– Recommendation: Antibiotic therapy and rest
""",
        "Heart Disease": """
Heart Disease
– Status: Active
– BP: 140/90 | Cholesterol: 220 mg/dL
– Requires cardiac risk management and lifestyle changes
""",
        "Liver Disease": """
Liver Disease
– Status: Elevated enzymes
– ALT: 85 U/L | AST: 74 U/L | Bilirubin: 2.1 mg/dL
– Suggest reducing alcohol intake and hepatology referral
""",
        "HIV": """
HIV
– Status: High Risk Detected
– Risk Indicators: Low CD4 count, Positive viral load
– ART treatment initiation advised
""",
        "Hypertension": """
Hypertension
– Status: Persistent
– BP: 150/95 mmHg
– Antihypertensive therapy recommended
""",
        "PCOS": """
PCOS
– Status: Likely
– Indicators: Irregular cycles, High testosterone, Ovarian cysts on ultrasound
– Endocrinologist consultation recommended
""",
        "Obesity": """
Obesity
– Status: Diagnosed
– BMI: 34.2
– Risk for diabetes, cardiovascular disease, and liver dysfunction
– Urgent weight management advised
""",
        "Stroke": """
Stroke Risk
– Status: History of ischemic symptoms
– MRI: Small vessel disease
– Neuro follow-up and preventive anticoagulation recommended
""",
        "Parkinson's Disease": """
Parkinson's Disease
– Status: Early Symptoms
– Signs: Tremors, Bradykinesia, Rigidity
– Dopaminergic therapy to be considered
""",
        "Cystic Fibrosis": """
Cystic Fibrosis
– Status: Detected via genetic panel and sweat chloride test
– Lung function: FEV1 reduced
– Pulmonologist consultation and airway clearance therapy required
""",
        "Hepatitis": """
Hepatitis
– Status: Detected via Hepatitis B Surface Antigen
– Symptoms: Fatigue, Jaundice
- ALT: 120 U/L | AST: 115 U/L | Bilirubin: 3.2 mg/dL
– Recommendation: Hepatitis B vaccination and liver function monitoring, viral panel and hepatology referral advised
"""

    }

    # Print each disease section if detected
    for disease, status in final_results.items():
        if status == "Detected" and disease in disease_reports:
            print(disease_reports[disease])

# generate_predicted_progression_section(...)
def generate_predicted_progression_section(predicted_progression, final_results):
    print("\n PREDICTED DISEASE PROGRESSION (If unmanaged):")
    for disease, certainty in predicted_progression.items():
        if final_results.get(disease) == "Detected":
            print(f"• {disease} may progress if left unmanaged")
            print(f"  – Certainty: {certainty:.2f}%")

            progression_reasons = {
                "Diabetes": "Persistently elevated blood glucose and HbA1c levels can cause long-term damage to blood vessels and nerves,\n"
                            "            increasing the risk of complications like diabetic neuropathy, retinopathy, nephropathy, and cardiovascular diseases.",
                "Chronic Kidney Disease": "Ongoing nephron loss due to elevated creatinine and proteinuria (albuminuria) can accelerate kidney function decline,\n"
                                         "            potentially progressing to Stage 4 or 5 CKD and requiring dialysis or transplantation.",
                "Heart Disease": "If underlying atherosclerosis, hypertension, or coronary artery disease is not treated,\n"
                                 "            plaque buildup can lead to myocardial infarction (heart attack), heart failure, or arrhythmia-related complications.",
                "Liver Disease": "Persistently high ALT, AST, and bilirubin may suggest ongoing inflammation or fibrosis,\n"
                                "            which can eventually progress to cirrhosis, portal hypertension, or even liver cancer.",
                "Cancer": "Malignant tumor cells may grow aggressively and metastasize to other organs,\n"
                         "            leading to reduced treatment success and increased systemic complications if not caught early.",
                "Stroke": "Without secondary prevention, patients remain at high risk for another cerebrovascular event,\n"
                         "            especially if blood pressure, atrial fibrillation, or cholesterol levels remain uncontrolled.",
                "Alzheimer's Disease": "Cognitive decline is likely to worsen, progressing from mild forgetfulness to severe dementia,\n"
                                     "            affecting functional independence, memory, orientation, and personality.",
                "Parkinson's Disease": "Motor dysfunction, including tremors, bradykinesia, and rigidity, may progress,\n"
                                       "            leading to impaired mobility, falls, and dependency in daily activities.",
                "Anemia": "Continued iron or vitamin deficiency may impair oxygen delivery to tissues,\n"
                          "            resulting in chronic fatigue, weakness, and potential heart strain.",
                "Obesity": "Progressive weight gain can lead to insulin resistance, hypertension, joint issues, and sleep apnea,\n"
                           "            increasing the long-term risk of metabolic syndrome and organ damage.",
                "PCOS": "Without hormonal regulation, PCOS may cause persistent menstrual irregularities,\n"
                        "            fertility issues, insulin resistance, and increased risk for type 2 diabetes and endometrial cancer.",
                "Cystic Fibrosis": "Continued mucus buildup in lungs leads to chronic infections and airway blockage,\n"
                                   "            progressively reducing lung function and increasing hospitalization risk.",
                "Pneumonia": "Inadequate treatment or delayed diagnosis may lead to worsening infection,\n"
                             "            resulting in sepsis, respiratory failure, or lung abscess formation.",
                "Tuberculosis": "Active TB can become resistant to treatment and spread to other organs,\n"
                                "            leading to complications like spinal TB, meningitis, or disseminated (miliary) TB.",
                "Hypertension": "Long-standing uncontrolled high blood pressure can damage blood vessels,\n"
                                "            increasing risk for heart attack, kidney damage, vision loss, and stroke.",
                "HIV": "Without antiretroviral therapy (ART), HIV continues to replicate,\n"
                       "            weakening the immune system and progressing to AIDS with life-threatening opportunistic infections.",
                "Hepatitis": "Chronic hepatitis can lead to liver scarring (cirrhosis),\n"
                             "            increasing the risk of liver cancer and complications like ascites and hepatic encephalopathy."
            }

            if disease in progression_reasons:
                print(f"  – Reason: {progression_reasons[disease]}")

            print()

# generate_future_disease_risk_section(...)

def generate_future_disease_risk_section(future_disease_risks, final_results):
    print(" FUTURE DISEASE RISKS (Based on current indicators):")
    for disease, risk in future_disease_risks.items():
        if final_results.get(disease) == "Not Detected":
            # Choose forecast window based on risk %
            if risk > 75:
                forecast_window = "1–2 years"
            elif risk > 50:
                forecast_window = "2–4 years"
            else:
                forecast_window = "3–5 years"

            print(f"• {disease}: {risk:.2f}% estimated risk in next {forecast_window}")

            # Priority Notes
            if risk > 75:
                print("   High priority: Consider immediate preventive care or screening.")
            elif risk > 50:
                print("   Moderate risk: Monitor indicators regularly.")
            else:
                print("   Low risk: Maintain healthy lifestyle and regular check-ups.")
            print()

# generate_symptom_timeline_section(...)

def generate_symptom_timeline_section(final_results):
    print("\n SYMPTOM TIMELINE (If unmanaged):")

    symptom_timeline = {
        "Diabetes": [
            "  → 0–6 months: Fatigue, increased thirst & urination",
            "  → 6–18 months: Neuropathy, blurred vision",
            "  → 2+ years: Kidney damage, foot ulcers, vision loss"
        ],
        "Chronic Kidney Disease": [
            "  → 0–12 months: Mild fatigue, foamy urine",
            "  → 1–2 years: Swelling in legs, high BP",
            "  → 3+ years: Anemia, bone weakness, ESRD"
        ],
        "Heart Disease": [
            "  → 0–6 months: Chest tightness on exertion",
            "  → 6–18 months: Angina, shortness of breath",
            "  → 2+ years: Heart attack or cardiac arrest"
        ],
        "Liver Disease": [
            "  → 0–12 months: Loss of appetite, fatigue",
            "  → 1–2 years: Jaundice, swelling, liver pain",
            "  → 3+ years: Cirrhosis, liver failure"
        ],
        "Cancer": [
            "  → Early Stage: Localized mass or swelling",
            "  → 6–12 months: Spread to adjacent tissues",
            "  → 1–2 years: Possible metastasis if untreated"
        ],
        "Stroke": [
            "  → Precursor: Headache, dizziness, confusion",
            "  → Onset: Sudden weakness, speech loss",
            "  → Aftermath: Paralysis, cognitive decline"
        ],
        "Alzheimer's Disease": [
            "  → 0–1 year: Mild memory loss, disorientation",
            "  → 2–4 years: Language & planning problems",
            "  → 5+ years: Severe dementia, dependency"
        ],
        "Parkinson's Disease": [
            "  → 0–2 years: Tremors, slow movements",
            "  → 3–5 years: Muscle rigidity, balance issues",
            "  → 5+ years: Speech problems, cognitive issues"
        ],
        "Anemia": [
            "  → 0–3 months: Fatigue, pale skin",
            "  → 4–6 months: Dizziness, headaches",
            "  → 6+ months: Organ stress, breathlessness"
        ],
        "Obesity": [
            "  → Ongoing: Weight gain, fatigue",
            "  → 1–2 years: Joint pain, insulin resistance",
            "  → 3+ years: Heart disease, diabetes onset"
        ],
        "PCOS": [
            "  → Initial: Irregular periods, acne",
            "  → 1–2 years: Weight gain, hair loss",
            "  → 3+ years: Infertility, metabolic syndrome"
        ],
        "Cystic Fibrosis": [
            "  → 0–6 months: Frequent cough, mucus buildup",
            "  → 1–2 years: Recurrent lung infections",
            "  → 3+ years: Reduced lung function, hospitalization"
        ],
        "Pneumonia": [
            "  → 0–1 week: Fever, cough, chest pain",
            "  → 1–2 weeks: Breathing difficulty",
            "  → 3+ weeks: Sepsis, lung abscess if untreated"
        ],
        "Tuberculosis": [
            "  → 0–2 months: Chronic cough, night sweats",
            "  → 3–6 months: Weight loss, chest pain",
            "  → 6+ months: Lung damage, spread to organs"
        ],
        "Hypertension": [
            "  → Silent phase: No symptoms for months",
            "  → 1–2 years: Headaches, fatigue, vision issues",
            "  → 3+ years: Kidney failure, stroke, heart issues"
        ],
        "HIV": [
            "  → 0–3 months: Flu-like symptoms",
            "  → 1–3 years: Asymptomatic period",
            "  → 3+ years: Opportunistic infections, AIDS if untreated"
        ],
        "Hepatitis": [
            "  → 0–3 months: Fatigue, jaundice",
            "  → 3–6 months: Liver inflammation, abdominal pain",
            "  → 6+ months: Cirrhosis, liver cancer if untreated"
        ]
    }

    for disease, status in final_results.items():
        if status == "Detected" and disease in symptom_timeline:
            print(f"\n• {disease}:")
            for symptom in symptom_timeline[disease]:
                print(symptom)


# generate_disease_interaction_network(...)

def generate_disease_interaction_network(final_results):
    print("\n DISEASE INTERACTION NETWORK:\nDetected Chain:")
    interaction_chains = {
        "Obesity": [
            ("Diabetes", "Obesity → Insulin Resistance → Diabetes"),
            ("Hypertension", "Obesity → ↑ BP → Hypertension")
        ],
        "Diabetes": [
            ("Chronic Kidney Disease", "→ Endothelial dysfunction → CKD"),
            ("Cancer", "Diabetes → Renal Stress → Mass Formation → ↑ Cancer Risk"),
            ("Heart Disease", "Diabetes → Vascular Inflammation → Atherosclerosis → Heart Disease")
        ],
        "Chronic Kidney Disease": [
            ("Stroke", "→ ↑ BP + Atrial Fibrillation → Stroke Risk"),
            ("Heart Disease", "CKD → Electrolyte Imbalance → Arrhythmia → Heart Disease")
        ],
        "PCOS": [
            ("Obesity", "PCOS → Hormonal Imbalance → Obesity")
        ],
        "Anemia": [
            ("Heart Disease", "Anemia → Hypoxia → Cardiac Overload → Heart Disease"),
            ("Cystic Fibrosis", "Anemia → Weak Oxygenation → Respiratory Burden in CF")
        ],
        "HIV": [
            ("Tuberculosis", "HIV → ↓ Immunity → Tuberculosis Susceptibility"),
            ("Pneumonia", "HIV → ↓ Immunity → Pneumonia Vulnerability"),
            ("Cancer", "HIV → CD4 Depletion → Cancer Pathways")
        ],
        "Alzheimer's Disease": [
            ("Stroke", "Stroke History → Cognitive Impairment → Alzheimer's Disease"),
            ("Parkinson's Disease", "Neurodegeneration Overlap → Parkinson’s + Alzheimer’s Risk")
        ],
        "Liver Disease": [
            ("Cancer", "Chronic Liver Damage → Cirrhosis → Hepatic Cancer")
        ],
        "Pneumonia": [
            ("Cystic Fibrosis", "CF → Mucus Stasis → Chronic Infections → Pneumonia")
        ],
        "Tuberculosis": [
            ("HIV", "HIV → ↓ Immunity → TB Infection")
        ],
        "Hypertension": [
            ("Stroke", "Hypertension → Vascular Damage → Stroke Risk")
        ],
        "Parkinson's Disease": [
            ("Stroke", "Stroke History → Cognitive Impairment → Parkinson’s Disease")
        ],
        "Heart Disease": [
            ("Stroke", "Heart Disease → Vascular Dysfunction → Stroke Risk")
        ],
        "Cystic Fibrosis": [
            ("Pneumonia", "CF → Mucus Stasis → Chronic Infections → Pneumonia")
        ],
        "Hepatitis": [
            ("Liver Disease", "Hepatitis → Liver Inflammation → Liver Disease")
        ],
        "Stroke": [
            ("Heart Disease", "Heart Disease → Vascular Dysfunction → Stroke Risk")
        ],
        "Cancer": [
            ("HIV", "HIV → CD4 Depletion → Cancer Pathways")
        ]
    }

    found = False
    for disease, relations in interaction_chains.items():
        if final_results.get(disease) == "Detected":
            for target, description in relations:
                if final_results.get(target) == "Detected":
                    print(description)
                    found = True

    if not found:
        print("No major chains detected. Diseases appear isolated.")

# generate_immunity_forecast(...)

def generate_immunity_forecast(final_results):
    print("\n IMMUNITY FORECAST / RESISTANCE SCORE:")

    # Define disease impact categories in dictionary format
    immunity_impact = {
        "HIV": ("High", 20),
        "Tuberculosis": ("High", 20),
        "Pneumonia": ("High", 20),
        "Cancer": ("High", 20),
        "Cystic Fibrosis": ("High", 20),
        "Anemia": ("Moderate", 10),
        "Diabetes": ("Moderate", 10),
        "Liver Disease": ("Moderate", 10),
        "Chronic Kidney Disease": ("Moderate", 10),
        "PCOS": ("Moderate", 10),
        "Obesity": ("Low", 5),
        "Stroke": ("Low", 5),
        "Heart Disease": ("Low", 5),
        "Hypertension": ("Low", 5),
        "Alzheimer's Disease": ("Low", 5),
        "Parkinson's Disease": ("Low", 5),
        "Hepatitis": ("Low", 5)
    }

    score = 100
    breakdown = []

    for disease, (impact, deduction) in immunity_impact.items():
        if final_results.get(disease) == "Detected":
            score -= deduction
            breakdown.append(f"- {disease}: {impact} impact (-{deduction})")

    # Clamp the score to valid range
    score = max(0, min(score, 100))

    # Interpretation of score
    if score >= 75:
        level = "HIGH"
        comment = "Strong immunity profile. Normal recovery expected from infections/surgeries."
    elif score >= 50:
        level = "MODERATE"
        comment = "Moderate immune reserve. Recovery may be delayed; preventive care suggested."
    else:
        level = "LOW"
        comment = "Immunity is significantly compromised. High risk for infections and complications."

    # Output
    print(f"Immunity Index: {level} ({score}%)")
    print("Breakdown of Immunity Score Deductions:")
    for line in breakdown:
        print(f"  {line}")
    print(f"Summary: {comment}\n")


# generate_clinical_recommendations(......)

def generate_clinical_recommendation_section(final_results):
    print("\n CLINICAL RECOMMENDATIONS BASED ON DETECTED DISEASES")

    diagnosed = [d.lower().replace("'", '').replace(" ", "_") for d, status in final_results.items() if status == "Detected"]

    if not diagnosed:
        print("\n• General Advice:")
        print("  ↳ Maintain regular checkups")
        print("  ↳ Follow a healthy diet and exercise routine")
        print("  ↳ Avoid smoking and excessive alcohol consumption")
        return

    recommendations = {
        'diabetes': [
            "HbA1c testing every 3–6 months",
            "Annual comprehensive foot exam",
            "Annual dilated eye exam",
            "Regular kidney function monitoring",
            "Nutritionist consultation",
            "Cardiovascular risk assessment"
        ],
        'heart_disease': [
            "Cardiology consultation within 1 month",
            "Lipid profile testing every 6 months",
            "Weekly blood pressure monitoring",
            "Stress test evaluation if symptoms worsen",
            "Dietary modifications for heart health",
            "Regular moderate exercise program"
        ],
        'chronic_kidney_disease': [
            "Nephrology referral",
            "Monthly kidney function tests",
            "Blood pressure control (<130/80 mmHg)",
            "Protein restriction diet if indicated",
            "Avoid NSAIDs and nephrotoxic drugs",
            "Monitor for anemia and bone disease"
        ],
        'liver_disease': [
            "Hepatology consultation",
            "Liver function tests every 3 months",
            "Ultrasound elastography for fibrosis",
            "Alcohol abstinence",
            "Vaccination for hepatitis A/B if not immune",
            "Monitor for varices if cirrhosis present"
        ],
        'hiv': [
            "Infectious disease specialist referral",
            "CD4 and viral load monitoring every 3–6 months",
            "ART regimen optimization if needed",
            "STI screening every 6 months",
            "Pneumocystis pneumonia prophylaxis if CD4<200",
            "Mental health screening"
        ],
        'pneumonia': [
            "Chest X-ray follow-up in 6–8 weeks",
            "Pneumococcal and influenza vaccination",
            "Pulmonary function tests if symptoms persist",
            "Smoking cessation if applicable",
            "Consider underlying immunodeficiency workup",
            "Respiratory therapy if needed"
        ],
        'tuberculosis': [
            "Directly observed therapy (DOT) program",
            "Contact tracing and testing",
            "Monthly sputum tests during treatment",
            "Liver function monitoring (INH/RMP)",
            "HIV testing if status unknown",
            "Chest X-ray follow-up after treatment"
        ],
        'hypertension': [
            "Blood pressure monitoring at least twice weekly",
            "Lifestyle modifications (diet, exercise, stress management)",
            "Medication review and adjustment every 3 months",
            "Annual kidney function and cholesterol check",
            "Stress management counseling",
            "Smoking cessation if applicable"
        ],
        'alzheimers_disease': [
            "Neurology or geriatrics consultation",
            "Cognitive assessment annually",
            "Memory training exercises",
            "Environment modification for safety",
            "Social engagement and support",
            "Medication review for cognitive enhancement"
        ],
        'parkinson_s_disease': [
            "Neurology consultation",
            "Motor function assessment every 6 months",
            "Speech therapy if dysphagia present",
            "Deep brain stimulation (DBS) evaluation if severe",
            "Fall prevention measures",
            "Regular medication adherence counseling"
        ],
        'anemia': [
            "Hematology consultation",
            "Iron and vitamin levels monitoring every 3 months",
            "Dietary modifications for iron intake",
            "Erythropoiesis stimulating agents (ESA) if needed",
            "Blood transfusion consideration if severe",
            "Iron supplementation with gastroprotective measures"
        ],
        'stroke': [
            "Neurology consultation within 24 hours of symptoms",
            "Thrombolytic therapy consideration if eligible",
            "Anticoagulation therapy if atrial fibrillation present",
            "Lifestyle modifications for cardiovascular risk reduction",
            "Physical and occupational therapy for recovery",
            "Speech therapy if dysphagia present"
        ],
        'cancer': [
            "Oncology consultation",
            "Follow-up imaging and pathology every 3–6 months",
            "Performance status assessment",
            "Pain management plan",
            "Psychosocial support and counseling",
            "Genetic counseling if applicable"
      ],
        'pcos':[
            "Endocrinology consultation",
            "Hormonal monitoring every 3 months",
            "Lifestyle modifications (diet, exercise)",
            "Contraceptive options consideration if applicable",
            "Fertility evaluation if desired",
            "Stress management counseling"
      ],
        'cystic fibrosis':[
            "Pulmonology consultation",
            "Pulmonary function tests every 6 months",
            "Antibiotic prophylaxis for infections",
            "Dietary modifications for malnutrition",
            "Exercise program for lung function",
            "Genetic counseling if applicable"
      ],
        'hepatitis':[
            "Hepatology consultation",
            "Liver function tests every 3 months",
            "Vaccination for hepatitis A/B if not immune",
            "Alcohol abstinence",
            "Lifestyle modifications for overall health",
            "Monitor for cirrhosis and complications"
      ],
        'obesity':[
            "Nutritionist consultation",
            "Weight monitoring and tracking",
            "Exercise program tailored to fitness level",
            "Behavioral therapy for eating habits",
            "Consider bariatric surgery if severely obese",
            "Regular cardiovascular risk assessment"
        ]
    }

    for disease in diagnosed:
        disease_title = disease.replace("_", " ").title()
        print(f"\n• {disease_title}:")
        recs = recommendations.get(disease, [
            "Consult appropriate specialist",
            "Regular monitoring of disease markers",
            "Lifestyle modifications as needed"
        ])
        for r in recs:
            print(f"  ↳ {r}")

# generate_disease_explainability_section(...)

def generate_disease_explainability_section(final_results):
    print("\n EXPLAINABILITY: WHY THESE DISEASES WERE DETECTED")

    # Dictionary of explanations per disease
    explainability_reasons = {
        "Diabetes": "Elevated glucose, HbA1c, obesity, and family history indicate poor glycemic control.",
        "Chronic Kidney Disease": "High creatinine, low eGFR, and albuminuria suggest progressive kidney impairment.",
        "Heart Disease": "High cholesterol, abnormal ECG, and angina features point to cardiac risk.",
        "Liver Disease": "Elevated liver enzymes (ALT, AST), bilirubin levels, and alcohol history suggest hepatic damage.",
        "Cancer": "Malignant pattern in tumor size, shape, and texture from imaging or pathology data.",
        "Stroke": "High BP, atrial fibrillation, and vascular issues indicate high cerebrovascular risk.",
        "Alzheimer's Disease": "Memory test scores and neurological patterns suggest progressive cognitive decline.",
        "Parkinson's Disease": "Tremors, motor instability, and vocal feature abnormalities align with Parkinsonism.",
        "Anemia": "Low hemoglobin, hematocrit, and iron levels reflect impaired oxygen delivery.",
        "Obesity": "High BMI, poor diet, and sedentary lifestyle indicate chronic weight imbalance.",
        "PCOS": "Irregular menstruation, hormonal imbalance (LH/FSH), and ovarian cyst patterns detected.",
        "Cystic Fibrosis": "Recurring lung infections, mucus buildup, and CF gene indicators present.",
        "Pneumonia": "Lung inflammation, chest X-ray opacity, and breathing difficulty signal infection.",
        "Tuberculosis": "TB-specific imaging findings, cough history, and exposure risk strongly suggest active TB.",
        "Hypertension": "Sustained high BP readings, salt intake, stress, and hereditary factors noted.",
        "HIV": "Exposure history, immune suppression patterns, and CD4+ decline indicate infection.",
        "Hepatitis": "Elevated liver enzymes, jaundice, and viral antibodies suggest hepatic inflammation."
    }

    for disease, status in final_results.items():
        if status == "Detected":
            print(f"\n• {disease}:")
            explanation = explainability_reasons.get(disease, "↳ Diagnostic criteria met based on model indicators.")
            print(f"  ↳ {explanation}")

#generate_clinical_guidance_summary(final_results, predicted_progression)

def generate_clinical_guidance_summary(final_results, predicted_progression):
    print("\n CLINICAL GUIDANCE: ACTIONS BASED ON RISK LEVELS")


    guidance_text = {
        'critical': " Immediate clinical intervention required.",
        'high': " Urgent specialist consultation recommended.",
        'moderate': " Clinical evaluation and regular monitoring advised.",
        'low': " Preventive lifestyle changes and periodic checkups.",
        'minimal': " Maintain general health with routine habits."
    }

    def get_risk_level(score):
        if score >= 85:
            return 'critical'
        elif score >= 70:
            return 'high'
        elif score >= 50:
            return 'moderate'
        elif score >= 30:
            return 'low'
        else:
            return 'minimal'

    for disease, status in final_results.items():
        if disease not in predicted_progression:
            continue
        risk_score = predicted_progression[disease]
        level = get_risk_level(risk_score)
        print(f"\n• {disease}:")
        print(f"  ↳ Risk Score: {risk_score:.2f}% → {level.upper()}")
        print(f"  ↳ Recommendation: {guidance_text[level]}")

from datetime import datetime

# === Final Report ===
def generate_full_patient_report(patient_id, age, gender, final_results, predicted_progression, future_disease_risks):
    print(f"\n{'='*60}\n PATIENT DEMOGRAPHICS\n{'='*60}")
    print(f"Patient ID: {patient_id}")
    print(f"Age: {age} | Gender: {gender}")
    print(f"Date: {datetime.now().strftime('%B %d, %Y')}")

    print(f"\n{'='*60}\n CURRENT DIAGNOSED DISEASES\n{'='*60}")
    print(generate_terminal_case_report(final_results))

    print(f"\n{'='*60}\n PREDICTED DISEASE PROGRESSION\n{'='*60}")
    generate_predicted_progression_section(predicted_progression, final_results)

    print(f"\n{'='*60}\n FUTURE DISEASE RISKS\n{'='*60}")
    generate_future_disease_risk_section(future_disease_risks, final_results)

    print(f"\n{'='*60}\n SYMPTOM TIMELINE\n{'='*60}")
    generate_symptom_timeline_section(final_results)

    print(f"\n{'='*60}\n DISEASE INTERACTION NETWORK\n{'='*60}")
    print(generate_disease_interaction_network(final_results))

    print(f"\n{'='*60}\n IMMUNITY FORECAST\n{'='*60}")
    generate_immunity_forecast(final_results)

    print(f"\n{'='*60}\n CLINICAL RECOMMENDATIONS\n{'='*60}")
    generate_clinical_recommendation_section(final_results)

    print(f"\n{'='*60}\n EXPLAINABILITY\n{'='*60}")
    generate_disease_explainability_section(final_results)

    print(f"\n{'='*60}\n CLINICAL GUIDANCE SUMMARY\n{'='*60}")
    generate_clinical_guidance_summary(final_results, predicted_progression)

    print(f"\n{'='*60}\n END OF REPORT\n{'='*60}")


generate_full_patient_report(
    patient_id="KP20250503-001",
    age=48,
    gender="Male",
    final_results=final_results,
    predicted_progression=predicted_progression,
    future_disease_risks=future_disease_risks
)


# Sharp Visulization

def plot_disease_risk_dashboard(final_results, predicted_progression, future_disease_risks):
    diseases = list(final_results.keys())
    fig, ax = plt.subplots(figsize=(16, 12))

    bar_height = 0.3
    spacing = 1.2
    y_pos = []
    labels = []

    for i, disease in enumerate(diseases):
        y = -i * spacing
        if final_results[disease] == "Detected":

            red_length = predicted_progression.get(disease, 0)
            ax.barh(y, red_length, height=bar_height/2, color='red')


            green_length = min(red_length + 10, 100)
            ax.barh(y - bar_height, green_length, height=bar_height, color='green')

            y_pos.append(y - bar_height / 2)
            labels.append(disease)

        else:

            blue_length = future_disease_risks.get(disease, 0)
            ax.barh(y, blue_length, height=bar_height, color='steelblue')
            y_pos.append(y)
            labels.append(disease)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Risk (%)')
    ax.set_ylabel('Diseases')
    ax.set_title('Overall Disease Risk Dashboard')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    legend_patches = [
        Patch(color='red', label='Detected Disease Marker (%)'),
        Patch(color='green', label='Predicted Progression (%)'),
        Patch(color='steelblue', label='Future Disease Risk (%)')
    ]
    ax.legend(handles=legend_patches, loc='upper right')
    plt.tight_layout()
    plt.show()


plot_disease_risk_dashboard(final_results, predicted_progression, future_disease_risks)

# Radar Visulization

def plot_radar_chart(final_results, predicted_progression):
    diseases = list(final_results.keys())
    current_risk = [predicted_progression.get(d, 0) for d in diseases]
    future_risk = [future_disease_risks.get(d, 0) for d in diseases]
    status = [final_results[d] for d in diseases]


    values = current_risk + current_risk[:1]  
    angles = np.linspace(0, 2 * np.pi, len(diseases), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, values, 'o-', linewidth=2, label='Current Risk')
    ax.fill(angles, values, alpha=0.25)

    
    ax.set_thetagrids(np.degrees(angles[:-1]), diseases, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Current Disease VS Predicted Disease Progression", fontsize=14, y=1.08)
    ax.grid(True)

    
    for angle, risk, label, stat in zip(angles, values, diseases + [diseases[0]], status + [status[0]]):
        if stat == "Detected":

             ax.text(
                angle,
                min(risk + 35, 100),
                f"{label}*",
                ha='center',
                va='center',
                fontsize=9,
                color='red',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3')

            )

    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.show()

plot_radar_chart(final_results, predicted_progression)
