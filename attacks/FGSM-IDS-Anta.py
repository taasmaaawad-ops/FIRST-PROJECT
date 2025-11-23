----------------------------------FGSM-IDS-Anta------------------------------------


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
""" #from sklearn.preprocessing import StandardScaler
#import os
 """
class MultiArmedBanditAntColonyOptimization:
    def __init__(self, n_arms, n_ants):
        self.n_arms = n_arms
        self.n_ants = n_ants
        self.arms = [SVC(probability=True), LogisticRegression(), RandomForestClassifier(), self.create_dnn_model()]
        self.pheromone = np.ones(n_arms)
        self.epsilon = 0.1

    def create_dnn_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def choose_arm(self):
        probabilities = self.pheromone / np.sum(self.pheromone)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms)
        else:
            return np.random.choice(self.n_arms, p=probabilities)

    def update(self, arm, reward):
        self.pheromone[arm] += reward

# Fast Gradient Sign Method (FGSM) for generating adversarial samples
def generate_adversarial_sample(model, X, y, epsilon=0.1):
    with tf.GradientTape() as tape:
        tape.watch(X)
        prediction = model(X)
        loss = tf.keras.losses.binary_crossentropy(y, prediction)
    gradient = tape.gradient(loss, X)
    signed_grad = tf.sign(gradient)
    adversarial_X = X + epsilon * signed_grad
    return adversarial_X

# Load the extracted features and labels from the CSV file
extracted_features_path = r"E:\train\encoded_features_2017_18.csv"
df_extracted_features = pd.read_csv(extracted_features_path)

# Separate features and labels
X = df_extracted_features.drop(columns=['Label'])
y = df_extracted_features['Label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Taining we use the 2017&18 Data and testing we take the 2017 adversial data
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Multi-Armed Bandit with Ant Colony Optimization
n_arms = 4  # Number of classifiers
n_ants = 10  # Number of ants
bandit = MultiArmedBanditAntColonyOptimization(n_arms, n_ants)

# Train the Multi-Armed Bandit
num_iterations = 10
for _ in range(num_iterations):
    for _ in range(n_ants):
        arm = bandit.choose_arm()
        classifier = bandit.arms[arm]
        if classifier.__class__.__name__ != 'Sequential':
            # For non-DNN classifiers
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_train)
            reward = accuracy_score(y_train, y_pred)
            bandit.update(arm, reward)
        else:
            # For DNN classifier using FGSM for adversarial training
            adversarial_X_train = generate_adversarial_sample(classifier, X_train, y_train)
            classifier.fit(adversarial_X_train, y_train)
            _, accuracy = classifier.evaluate(X_train, y_train, verbose=0)
            reward = accuracy
            bandit.update(arm, reward)

# Choose the best classifier
best_arm = np.argmax(bandit.pheromone)
best_classifier = bandit.arms[best_arm]

# Evaluate the best classifier on the test set
if best_classifier.__class__.__name__ != 'Sequential':
    y_pred = best_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
else:
    _, accuracy = best_classifier.evaluate(X_test, y_test, verbose=0)
    y_pred = (best_classifier.predict(X_test) > 0.5).astype("int32")
    classification_rep = classification_report(y_test, y_pred)

# Print accuracy and classification report for the original test set
print("Results for original test set:")
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)

# Generate and save adversarial examples for the test set using FGSM
adversarial_X_test = generate_adversarial_sample(best_classifier, X_test, y_test)
adversarial_samples_dir = "adversarial_samples"
os.makedirs(adversarial_samples_dir, exist_ok=True)
np.savetxt(os.path.join(adversarial_samples_dir, "adversarial_samples.csv"), adversarial_X_test, delimiter=",")

# Evaluate the best classifier on the adversarial test set
if best_classifier.__class__.__name__ != 'Sequential':
    y_pred_adv = best_classifier.predict(adversarial_X_test)
    accuracy_adv = accuracy_score(y_test, y_pred_adv)
    classification_rep_adv = classification_report(y_test, y_pred_adv)
else:
    _, accuracy_adv = best_classifier.evaluate(adversarial_X_test, y_test, verbose=0)
    y_pred_adv = (best_classifier.predict(adversarial_X_test) > 0.5).astype("int32")
    classification_rep_adv = classification_report(y_test, y_pred_adv)

# Print accuracy and classification report for the adversarial test set
print("\nResults for adversarial test set:")
print(f"Accuracy: {accuracy_adv}")
print("Classification Report:")
print(classification_rep_adv)



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier, TensorFlowClassifier

class MultiArmedBanditAntColonyOptimization:
    def __init__(self, n_arms, n_ants):
        self.n_arms = n_arms
        self.n_ants = n_ants
        self.arms = [SVC(probability=True), LogisticRegression(), RandomForestClassifier(), self.create_dnn_model()]
        self.pheromone = np.ones(n_arms)
        self.epsilon = 0.1

    def create_dnn_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def choose_arm(self):
        probabilities = self.pheromone / np.sum(self.pheromone)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms)
        else:
            return np.random.choice(self.n_arms, p=probabilities)

    def update(self, arm, reward):
        self.pheromone[arm] += reward

# Load the extracted features and labels from the CSV file
extracted_features_path = r"E:\train\encoded_features_2017_18.csv"
df_extracted_features = pd.read_csv(extracted_features_path)

# Separate features and labels
X = df_extracted_features.drop(columns=['Label'])
y = df_extracted_features['Label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Taining we use the 2017&18 Data and testing we take the 2017 adversial data
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Multi-Armed Bandit with Ant Colony Optimization
n_arms = 4  # Number of classifiers
n_ants = 10  # Number of ants
bandit = MultiArmedBanditAntColonyOptimization(n_arms, n_ants)

# Train the Multi-Armed Bandit
num_iterations = 10
for _ in range(num_iterations):
    for _ in range(n_ants):
        arm = bandit.choose_arm()
        classifier = bandit.arms[arm]
        classifier.fit(X_train, y_train)
        if classifier.__class__.__name__ != 'Sequential':
            y_pred = classifier.predict(X_train)
            reward = accuracy_score(y_train, y_pred)
        else:
            _, accuracy = classifier.evaluate(X_train, y_train, verbose=0)
            reward = accuracy
        bandit.update(arm, reward)

# Choose the best classifier
best_arm = np.argmax(bandit.pheromone)
best_classifier = bandit.arms[best_arm]

# Create ART classifier for the best classifier
if best_classifier.__class__.__name__ != 'Sequential':
    art_classifier = SklearnClassifier(model=best_classifier)
else:
    art_classifier = TensorFlowClassifier(model=self.create_dnn_model(), nb_classes=2, input_shape=X_train.shape[1])

# Generate adversarial samples using Fast Gradient Method (FGM)
fgm_attack = FastGradientMethod(estimator=art_classifier, eps=0.1)
X_test_adv = fgm_attack.generate(X_test)

# Evaluate the best classifier on the adversarial test set
y_pred_adv = best_classifier.predict(X_test_adv)
precision_adv, recall_adv, f1_score_adv, _ = precision_recall_fscore_support(y_test, y_pred_adv, average='binary')
auc_score_adv = roc_auc_score(y_test, y_pred_adv)

# Compute detection rate on adversarial samples (equivalent to recall)
detection_rate_adv = recall_adv

# Print precision, recall, detection rate, F1 score, and AUC score on adversarial samples
print("Precision on adversarial samples:", precision_adv)
print("Recall on adversarial samples:", recall_adv)
print("Detection Rate on adversarial samples:", detection_rate_adv)
print("F1 Score on adversarial samples:", f1_score_adv)
print("AUC Score on adversarial samples:", auc_score_adv)

# Evaluate the best classifier on the non-adversarial test set
if best_classifier.__class__.__name__ != 'Sequential':
    y_pred = best_classifier.predict(X_test)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    auc_score = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
else:
    _, accuracy = best_classifier.evaluate(X_test, y_test, verbose=0)
    y_pred = (best_classifier.predict(X_test) > 0.5).astype("int32")
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    auc_score = roc_auc_score(y_test, y_pred)
    
# Compute detection rate on non-adversarial samples (equivalent to recall)
detection_rate = recall

print("----------------------------------------------")
# Print precision, recall, detection rate, F1 score, AUC score, and accuracy on non-adversarial samples
print("Precision on non-adversarial samples:", precision)
print("Recall on non-adversarial samples:", recall)
print("Detection Rate on non-adversarial samples:", detection_rate)
print("F1 Score on non-adversarial samples:", f1_score)
print("AUC Score on non-adversarial samples:", auc_score)
print("Accuracy on non-adversarial samples:", accuracy)