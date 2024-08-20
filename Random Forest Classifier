# Import all the libraries required in the training

import ROOT
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract data
def extract_data(file_path, tree_path):
    file = ROOT.TFile.Open(file_path)
    tree = file.Get(tree_path)

    svPt = []
    svX = []
    svY = []
    svZ = []
    svNormalizedChi2 = []
    svChi2 = []
    svEta = []
    svMass = []
    svNumDaughters = []
    svDxy = []
    svNdf = []
    for event in tree:
        svPt.append(event.svPt)
        svX.append(event.svX)
        svY.append(event.svY)
        svZ.append(event.svZ)
        svNormalizedChi2.append(event.svNormalizedChi2)
        svChi2.append(event.svChi2)
        svEta.append(event.svEta)
        svMass.append(event.svMass)
        svNumDaughters.append(event.svNumDaughters)
        svDxy.append(event.svDxy)
        svNdf.append(event.svNdf)

    data = {
        'svPt': svPt,
        'svX': svX,
        'svY': svY,
        'svZ': svZ,
        'svNormalizedChi2': svNormalizedChi2,
        'svChi2': svChi2,
        'svEta': svEta,
        'svMass': svMass,
        'svNumDaughters': svNumDaughters,
        'svDxy': svDxy,
        'svNdf': svNdf
    }

    df = pd.DataFrame(data)
    df = df[df.svPt != 0]
    return df 

# Load the datasets
signal_df = extract_data("Grav_Combined.root", "demo/tree")
background_df = extract_data("QCD_Combined.root", "demo/tree")

# Balance the datasets
min_len = min(len(signal_df), len(background_df))
signal_df = signal_df.sample(min_len)
background_df = background_df.sample(min_len)

# Add labels and combine
signal_df['label'] = 1
background_df['label'] = 0

df = pd.concat([signal_df, background_df]) 

X = df.drop(['label'], axis=1)
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and probabilities
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# Classification Report
rf_report = classification_report(y_test, rf_preds)

# Confusion Matrix
rf_cm = confusion_matrix(y_test, rf_preds)

# ROC Curve
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

# Precision-Recall Curve
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_probs)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(rf_fpr, rf_tpr, label='Random Forest ROC', color='blue')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(rf_recall, rf_precision, label='Random Forest PRC', color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Calculate classification efficiency and signal/background proportion for each threshold
thresholds = np.linspace(0, 1, 101)
classification_efficiency = []
signal_proportion = []
background_proportion = []

for threshold in thresholds:
    # Apply threshold
    rf_preds = (rf_probs > threshold).astype(int)
    
    # Calculate efficiency
    rf_correct_classification = (rf_preds == y_test).sum()
    total_samples = len(y_test)
    rf_efficiency = (rf_correct_classification / total_samples) * 100

    rf_signal_count = 0 
    rf_background_count = 0 

    for i, value in enumerate(y_test): 
        if value == 1 and rf_preds[i] == 1: 
            rf_signal_count += 1
        if value == 0 and rf_preds[i] == 1: 
            rf_background_count += 1

    total_count = len(rf_preds)
    
    sig_og = y_test.value_counts()[1]
    back_og = y_test.value_counts()[0]
    
    rf_signal_eff = rf_signal_count * 100 / sig_og
    rf_background_eff = rf_background_count * 100 / back_og

    classification_efficiency.append(rf_efficiency)
    signal_proportion.append(rf_signal_eff)
    background_proportion.append(rf_background_eff)

# Plot classification efficiency for different thresholds
plt.figure(figsize=(10, 6))
plt.plot(thresholds, classification_efficiency, marker='o', label='Random Forest')
plt.title('Correct Classification Efficiency')
plt.xlabel('Threshold')
plt.ylabel('Classification Efficiency (%)')
plt.grid(True)
plt.legend()
plt.show()

# Plot background vs signal proportion for different thresholds
plt.figure(figsize=(10, 6))
plt.scatter(signal_proportion, background_proportion, marker='o', color='blue', label='Random Forest')

plt.title('Signal vs Background Proportion')
plt.xlabel('Signal Proportion (%)')
plt.ylabel('Background Proportion (%)')
plt.grid(True)
plt.legend()
plt.show()
