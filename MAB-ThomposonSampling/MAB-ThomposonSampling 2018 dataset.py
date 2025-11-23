--------------------Random Forest---------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_fscore_support

class ThompsonSamplingMultiArmedBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Initialize alpha parameters to 1
        self.beta = np.ones(n_arms)   # Initialize beta parameters to 1

    def choose_arm(self):
        samples = np.random.beta(self.alpha, self.beta)  # Thompson sampling
        return np.argmax(samples)

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

# Load the extracted features and labels from the CSV file
extracted_features_path = r"E:\train\encoded_features.csv"
df_extracted_features = pd.read_csv(extracted_features_path)

# Separate features and labels
X = df_extracted_features.drop(columns=['Label'])
y = df_extracted_features['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Multi-Armed Bandit
n_arms = len(np.unique(y_train))  # Number of unique classes
bandit = ThompsonSamplingMultiArmedBandit(n_arms)

# Train the Multi-Armed Bandit
for _ in range(len(X_train)):
    arm = bandit.choose_arm()
    reward = 1 if y_train.iloc[_] == arm else 0
    bandit.update(arm, reward)

# Initialize the Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier
random_forest_classifier.fit(X_train, y_train)

# Predict probabilities on the testing set
y_pred_proba = random_forest_classifier.predict_proba(X_test)[:, 1]

# Threshold the probabilities to get binary predictions
threshold = 0.5
y_pred = (y_pred_proba > threshold).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

# Parse classification report to get F1 score and detection rate
classification_dict = classification_report(y_test, y_pred, output_dict=True)
f1_score = classification_dict['1']['f1-score']
detection_rate = classification_dict['1']['recall']

# Calculate precision, recall, and F1 score
precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")
print(f"Detection Rate : {detection_rate}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"AUC Score: {auc_score}")
print("Classification Report:")
print(classification_rep)


----------------------------Logistic Regression---------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_fscore_support

class ThompsonSamplingMultiArmedBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Initialize alpha parameters to 1
        self.beta = np.ones(n_arms)   # Initialize beta parameters to 1

    def choose_arm(self):
        samples = np.random.beta(self.alpha, self.beta)  # Thompson sampling
        return np.argmax(samples)

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

# Load the extracted features and labels from the CSV file
extracted_features_path = r"E:\train\encoded_features.csv"
df_extracted_features = pd.read_csv(extracted_features_path)

# Separate features and labels
X = df_extracted_features.drop(columns=['Label'])
y = df_extracted_features['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Multi-Armed Bandit
n_arms = len(np.unique(y_train))  # Number of unique classes
bandit = ThompsonSamplingMultiArmedBandit(n_arms)

# Train the Multi-Armed Bandit
for _ in range(len(X_train)):
    arm = bandit.choose_arm()
    reward = 1 if y_train.iloc[_] == arm else 0
    bandit.update(arm, reward)

# Initialize the Logistic Regression classifier
logistic_regression_classifier = LogisticRegression(max_iter=1000, random_state=42)

# Train the Logistic Regression classifier
logistic_regression_classifier.fit(X_train, y_train)

# Predict probabilities on the testing set
y_pred_proba = logistic_regression_classifier.predict_proba(X_test)[:, 1]

# Threshold the probabilities to get binary predictions
threshold = 0.5
y_pred = (y_pred_proba > threshold).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

# Parse classification report to get F1 score and detection rate
classification_dict = classification_report(y_test, y_pred, output_dict=True)
f1_score = classification_dict['1']['f1-score']
detection_rate = classification_dict['1']['recall']

# Calculate precision, recall, and F1 score
precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")
print(f"Detection Rate : {detection_rate}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"AUC Score: {auc_score}")
print("Classification Report:")
print(classification_rep)

---------------------------Support Vectore Machine---------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_fscore_support

class ThompsonSamplingMultiArmedBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Initialize alpha parameters to 1
        self.beta = np.ones(n_arms)   # Initialize beta parameters to 1

    def choose_arm(self):
        samples = np.random.beta(self.alpha, self.beta)  # Thompson sampling
        return np.argmax(samples)

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

# Load the extracted features and labels from the CSV file
extracted_features_path = r"E:\train\encoded_features.csv"
df_extracted_features = pd.read_csv(extracted_features_path)

# Separate features and labels
X = df_extracted_features.drop(columns=['Label'])
y = df_extracted_features['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Multi-Armed Bandit
n_arms = len(np.unique(y_train))  # Number of unique classes
bandit = ThompsonSamplingMultiArmedBandit(n_arms)

# Train the Multi-Armed Bandit
for _ in range(len(X_train)):
    arm = bandit.choose_arm()
    reward = 1 if y_train.iloc[_] == arm else 0
    bandit.update(arm, reward)

# Initialize the SVM classifier
svm_classifier = SVC(kernel='rbf', probability=True, random_state=42)

# Train the SVM classifier
svm_classifier.fit(X_train, y_train)

# Predict probabilities on the testing set
y_pred_proba = svm_classifier.predict_proba(X_test)[:, 1]

# Threshold the probabilities to get binary predictions
threshold = 0.5
y_pred = (y_pred_proba > threshold).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

# Parse classification report to get F1 score and detection rate
classification_dict = classification_report(y_test, y_pred, output_dict=True)
f1_score = classification_dict['1']['f1-score']
detection_rate = classification_dict['1']['recall']

# Calculate precision, recall, and F1 score
precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")
print(f"Detection Rate : {detection_rate}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"AUC Score: {auc_score}")
print("Classification Report:")
print(classification_rep)

------------------------------DNN--------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class ThompsonSamplingMultiArmedBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Initialize alpha parameters to 1
        self.beta = np.ones(n_arms)   # Initialize beta parameters to 1

    def choose_arm(self):
        samples = np.random.beta(self.alpha, self.beta)  # Thompson sampling
        return np.argmax(samples)

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

# Load the extracted features and labels from the CSV file
extracted_features_path = r"E:\train\encoded_features.csv"
df_extracted_features = pd.read_csv(extracted_features_path)

# Separate features and labels
X = df_extracted_features.drop(columns=['Label'])
y = df_extracted_features['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Multi-Armed Bandit
n_arms = len(np.unique(y_train))  # Number of unique classes
bandit = ThompsonSamplingMultiArmedBandit(n_arms)

# Train the Multi-Armed Bandit
for _ in range(len(X_train)):
    arm = bandit.choose_arm()
    reward = 1 if y_train.iloc[_] == arm else 0
    bandit.update(arm, reward)

# Define and train a Deep Neural Network (DNN) using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(n_arms, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Predict on the testing set
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)

# Predict probabilities on the testing set
y_pred_proba = model.predict(X_test)

# Calculate precision, recall, F1 score, detection rate (recall), and AUC score
precision_recall_f1 = classification_report(y_test, np.argmax(y_pred_proba, axis=1), output_dict=True)
precision = precision_recall_f1['1']['precision']
recall = precision_recall_f1['1']['recall']
f1_score = precision_recall_f1['1']['f1-score']
detection_rate = recall

# Calculate AUC score for each class separately
auc_scores = []
for i in range(len(np.unique(y_train))):
    auc_scores.append(roc_auc_score((y_test == i).astype(int), y_pred_proba[:, i]))

# Average AUC scores across all classes
auc_score = np.mean(auc_scores)

# Print the evaluation metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print(f"Detection Rate (Recall): {detection_rate}")
print(f"AUC Score: {auc_score}")