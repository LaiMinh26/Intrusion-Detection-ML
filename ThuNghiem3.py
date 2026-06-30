import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
#chunks = pd.read_csv("phase2_NetworkData.csv", chunksize=100000)
#df = pd.concat([chunk for chunk in chunks])
df = pd.read_csv("phase2_NetworkData.csv", low_memory=False)
df = df.drop(columns=['subLabel', 'subLabelCat'])
df_sample = df.sample(n=100000, random_state=42)
print(df.head())
df.info()
df.describe()
print(df["label"].value_counts())
df.isnull().sum()
df = df.fillna(df.mean(numeric_only=True))
features = ['flow_duration', 'Rate', 'Srate', 'Tot size', 'IAT',
            'ack_count', 'syn_count', 'Variance', 'Magnitue']
#X = df[features]
#y = df['label']
X = df_sample[features]
y = df_sample['label']
results_adasyn = []
for seed in range(5):
    print(f"\n========== LOOP {seed + 1} ==========")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=seed,
        stratify=y
    )
    adasyn = ADASYN(sampling_strategy='auto',
                    random_state=seed,
                    n_neighbors=5
                    )
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    print(y_resampled.value_counts())
    # model = GaussianNB()
    model = DecisionTreeClassifier(max_depth=10, random_state=seed)
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results_adasyn.append([accuracy, precision, recall, f1])

    print("\nADASYN")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print(accuracy)
    #print(classification_report(y_test, y_pred))
#Tính trung bình
results_adasyn = np.array(results_adasyn)
print("\n==============================")
print("AVERAGE RESULTS")
print("==============================")
print("Accuracy :", results_adasyn[:,0].mean())
print("Precision:", results_adasyn[:,1].mean())
print("Recall   :", results_adasyn[:,2].mean())
print("F1-score :", results_adasyn[:,3].mean())
