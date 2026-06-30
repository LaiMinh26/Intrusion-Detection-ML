import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
df = pd.read_csv("CIDDS-001-internal-week2.csv", low_memory=False)
df['class'] = df['class'].apply(
    lambda x: 'normal' if x == 'normal' else 'attack'
)
print(df.head())
df.info()
df.describe()
print(df["class"].value_counts())
df.isnull().sum()
le_proto = LabelEncoder()
le_flags = LabelEncoder()
le_class = LabelEncoder()
df['Proto'] = le_proto.fit_transform(df['Proto'])
df['Flags'] = le_flags.fit_transform(df['Flags'])
df['class'] = le_class.fit_transform(df['class'])
df['Bytes'] = pd.to_numeric(df['Bytes'], errors='coerce')
df['Bytes'] = df['Bytes'].fillna(df['Bytes'].mean())
features = ['Duration', 'Proto', 'Src Pt', 'Dst Pt', 'Packets', 'Bytes', 'Flows',
            'Flags', 'Tos']
X = df[features]
y = df['class']

"""smote = SMOTE(
    sampling_strategy='auto',
    random_state=42,
    k_neighbors=5
)"""
X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
adasyn = ADASYN(sampling_strategy='auto',
                random_state=42,
                n_neighbors=5
)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
print(y_resampled.value_counts())


"""for k in [3, 5, 10]:
    for ratio in [0.5, 0.8, 1.0]:
        print(f"\n===== k={k}, ratio={ratio} =====")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        adasyn = ADASYN(n_neighbors=k, sampling_strategy=ratio, random_state=42)

        try:
            X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

            model = DecisionTreeClassifier(max_depth=10)
            model.fit(X_resampled, y_resampled)

            y_pred = model.predict(X_test)

            print(classification_report(y_test, y_pred))

        except Exception as e:
            print("Lỗi:", e)"""

features = ['Duration', 'Bytes', 'Packets']

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

comparison3 = pd.DataFrame({
    'Feature': features,
    'Min_Original': [X_train[col].min() for col in features],
    'Min_ADASYN': [X_resampled[col].min() for col in features],
    'Max_Original': [X_train[col].max() for col in features],
    'Max_ADASYN': [X_resampled[col].max() for col in features],
})

print(comparison3)

for col in features:
    plt.figure()

    plt.boxplot([X_train[col], X_resampled[col]],
                labels=['Original', 'ADASYN'])

    plt.title(f"Boxplot of {col}")
    plt.show()

#model = GaussianNB()
model = DecisionTreeClassifier(max_depth=10)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(classification_report(y_test, y_pred))

"""skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
print(scores)
print(scores.mean())"""
