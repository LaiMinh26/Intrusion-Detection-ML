import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
#chunks = pd.read_csv("phase2_NetworkData.csv", chunksize=100000)
#df = pd.concat([chunk for chunk in chunks])
df = pd.read_csv("phase2_NetworkData.csv", low_memory=False)
df = df.drop(columns=['subLabel', 'subLabelCat'])
df_sample = df.sample(n=100000, random_state=42)
print(df_sample.head())
df_sample.info()
df_sample.describe()
print(df_sample["label"].value_counts())
df_sample.isnull().sum()
df_sample = df_sample.fillna(df_sample.mean(numeric_only=True))
features = ['flow_duration', 'Rate', 'Srate', 'Tot size', 'IAT',
            'ack_count', 'syn_count', 'Variance', 'Magnitue']
#X = df[features]
#y = df['label']
X = df_sample[features]
y = df_sample['label']
X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
"""scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
X_test = pd.DataFrame(scaler.transform(X_test), columns=features)"""
adasyn = ADASYN(sampling_strategy='auto',
                random_state=42,
                n_neighbors=5
)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
print(y_resampled.value_counts())
comparison1 = pd.DataFrame({
    'Feature': features,
    'Mean_Original': [X_train[col].mean() for col in features],
    'Mean_ADASYN': [X_resampled[col].mean() for col in features],
    'Median_Original': [X_train[col].median() for col in features],
    'Median_ADASYN': [X_resampled[col].median() for col in features],
})
print(comparison1)
comparison2 = pd.DataFrame({
    'Feature': features,
    'Std_Original': [X_train[col].std() for col in features],
    'Std_ADASYN': [X_resampled[col].std() for col in features],
    'IQR_Original': [
        X_train[col].quantile(0.75) - X_train[col].quantile(0.25)
        for col in features
    ],
    'IQR_ADASYN': [
        X_resampled[col].quantile(0.75) - X_resampled[col].quantile(0.25)
        for col in features
    ],
})
print(comparison2)

print("Old distribution:")
print(y_train.value_counts(normalize=True))

print("\nNew distribution:")
print(y_resampled.value_counts(normalize=True))

print("\nCounts after ADASYN:")
print(y_resampled.value_counts())

plt.figure(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(3, 3, i + 1) 

    plt.boxplot(
        [X_train[col], X_resampled[col]],
        tick_labels=['Old', 'New']
    )

    plt.title(col)

plt.tight_layout()
plt.show()

X_train_sample = X_train.sample(n=10000, random_state=42)
X_resampled_sample = X_resampled.sample(n=10000, random_state=42)

plt.figure(figsize=(12, 8))
for i, col in enumerate(features):
    plt.subplot(3, 3, i+1)

    plt.hist(X_train_sample[col], bins=50, alpha=0.5, label='Old')
    plt.hist(X_resampled_sample[col], bins=50, alpha=0.5, label='ADASYN')

    plt.title(col)
    plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
for i, col in enumerate(features):
    plt.subplot(3, 3, i+1)

    X_train_sample[col].plot(kind='kde', label='Old')
    X_resampled_sample[col].plot(kind='kde', label='ADASYN')

    plt.title(col)
    plt.legend()

plt.tight_layout()
plt.show()
#model = GaussianNB()
model = DecisionTreeClassifier(max_depth=10)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(classification_report(y_test, y_pred))
