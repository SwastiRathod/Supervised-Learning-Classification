# Import all the libraries required in the training

import ROOT
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
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
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Predictions and probabilities
gb_preds = gb_model.predict(X_test)
gb_probs = gb_model.predict_proba(X_test)[:, 1]

# Classification Report
gb_report = classification_report(y_test, gb_preds)

# Confusion Matrix
gb_cm = confusion_matrix(y_test, gb_preds)

# ROC Curve
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_probs)

# Precision-Recall Curve
gb_precision, gb_recall, _ = precision_recall_curve(y_test, gb_probs)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(gb_fpr, gb_tpr, label='Gradient Boosting ROC', color='green')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(gb_recall, gb_precision, label='Gradient Boosting PRC', color='green')
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
    gb_preds = (gb_probs > threshold).astype(int)
    
    # Calculate efficiency
    gb_correct_classification = (gb_preds == y_test).sum()
    total_samples = len(y_test)
    gb_efficiency = (gb_correct_classification / total_samples) * 100

    gb_signal_count = 0 
    gb_background_count = 0 

    for i, value in enumerate(y_test): 
        if value == 1 and gb_preds[i] == 1: 
            gb_signal_count += 1
        if value == 0 and gb_preds[i] == 1: 
            gb_background_count += 1

    total_count = len(gb_preds)
    
    sig_og = y_test.value_counts()[1]
    back_og = y_test.value_counts()[0]
    
    gb_signal_eff = gb_signal_count * 100 / sig_og
    gb_background_eff = gb_background_count * 100 / back_og

    classification_efficiency.append(gb_efficiency)
    signal_proportion.append(gb_signal_eff)
    background_proportion.append(gb_background_eff)

# Plot classification efficiency for different thresholds
plt.figure(figsize=(10, 6))
plt.plot(thresholds, classification_efficiency, marker='o', label='Gradient Boosting')
plt.title('Correct Classification Efficiency')
plt.xlabel('Threshold')
plt.ylabel('Classification Efficiency (%)')
plt.grid(True)
plt.legend()
plt.show()

# Plot background vs signal proportion for different thresholds
plt.figure(figsize=(10, 6))
plt.scatter(signal_proportion, background_proportion, marker='o', color='green', label='Gradient Boosting')

plt.title('Signal vs Background Proportion')
plt.xlabel('Signal Proportion (%)')
plt.ylabel('Background Proportion (%)')
plt.grid(True)
plt.legend()
plt.show()
