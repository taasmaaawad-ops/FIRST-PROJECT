----------------------------ZOO IDS-Anta---------------------------


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from collections import defaultdict
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack

class MultiArmedBanditThompsonSampling:
    def __init__(self, num_classifiers):
        self.num_classifiers = num_classifiers
        self.successes = defaultdict(int)
        self.failures = defaultdict(int)
        self.selected_classifier = None

    def select_classifier(self):
        max_ucb = -float('inf')
        for clf in range(self.num_classifiers):
            beta_sample = np.random.beta(self.successes[clf] + 1, self.failures[clf] + 1)
            if beta_sample > max_ucb:
                max_ucb = beta_sample
                self.selected_classifier = clf
        return self.selected_classifier

    def update(self, clf_index, success):
        if success:
            self.successes[clf_index] += 1
        else:
            self.failures[clf_index] += 1

# Load the extracted features and labels from the CSV file
extracted_features_path = r"E:\train\encoded_features_2017_18.csv"
df_extracted_features = pd.read_csv(extracted_features_path)

# Separate features and labels
X = df_extracted_features.drop(columns=['Label'])
y = df_extracted_features['Label']

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Taining we use the 2017&18 Data and testing we take the 2017 adversial data
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Implement ACO 
class AntColony:
    def __init__(self, num_ants, num_iterations, num_features):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.num_features = num_features

    def select_features(self, X_train, y_train):
        # Implement feature selection using ACO
        # For simplicity, we'll randomly select features
        return np.random.choice(range(self.num_features), size=self.num_features // 2, replace=False)

# Initialize ACO
aco = AntColony(num_ants=10, num_iterations=50, num_features=X_train.shape[1])
# Select features using ACO
selected_features = aco.select_features(X_train, y_train)

# Extract selected features from the training and testing sets
X_train_aco = X_train.iloc[:, selected_features]
X_test_aco = X_test.iloc[:, selected_features]

# Initialize classifiers
classifiers = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    LogisticRegression(max_iter=1000, random_state=42),
    SVC(kernel='linear', random_state=42),
    tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_aco.shape[1],)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
]

# Initialize Thompson Sampling Multi-Armed Bandit
bandit = MultiArmedBanditThompsonSampling(num_classifiers=len(classifiers))

# Perform Thompson Sampling for a fixed number of rounds
num_rounds = 1000
for round in range(num_rounds):
    selected_clf_index = bandit.select_classifier()
    selected_clf = classifiers[selected_clf_index]
    
    if isinstance(selected_clf, tf.keras.Sequential):
        # Compile and train the DNN model
        selected_clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        selected_clf.fit(X_train_aco, y_train, epochs=10, batch_size=32, validation_data=(X_test_aco, y_test), verbose=0)
        y_pred_probs = selected_clf.predict(X_test_aco)
        y_pred = np.argmax(y_pred_probs, axis=1)
    else:
        # Train the classifier
        selected_clf.fit(X_train_aco, y_train)
        y_pred = selected_clf.predict(X_test_aco)

    # Evaluate the selected classifier and update the bandit
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    accuracy = report['accuracy']
    bandit.update(selected_clf_index, accuracy)

# Generate a classification report for the classifier
best_clf_index = max(bandit.successes, key=bandit.successes.get)
best_clf = classifiers[best_clf_index]

if isinstance(best_clf, tf.keras.Sequential):
    best_clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    best_clf.fit(X_train_aco, y_train, epochs=10, batch_size=32, validation_data=(X_test_aco, y_test), verbose=0)
    y_pred_probs = best_clf.predict(X_test_aco)
    y_pred = np.argmax(y_pred_probs, axis=1)
else:
    best_clf.fit(X_train_aco, y_train)
    y_pred = best_clf.predict(X_test_aco)

# Generate classification report for the classifier
target_names = [str(class_name) for class_name in label_encoder.classes_]
report = classification_report(y_test, y_pred, target_names=target_names)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)
print("Classification Report:")
print(report)

# Adversarial attack using ZooAttack (Zeroth-order Optimization Attack)
best_clf_art = SklearnClassifier(model=best_clf, clip_values=(0, 1))
attack = ZooAttack(classifier=best_clf_art, max_iter=100, learning_rate=1e-2, targeted=False, use_resize=False, nb_parallel=5)

# Convert data to NumPy array
X_test_np = X_test_aco.to_numpy()

X_test_adv = attack.generate(X_test_np)
y_pred_adv = best_clf_art.predict(X_test_adv)

# Flatten the y_test array 
y_test_flat = y_test if len(y_test.shape) == 1 else np.argmax(y_test, axis=1)

# Flatten the y_pred_adv array
y_pred_adv_flat = np.argmax(y_pred_adv, axis=1)

# Generate the classification report for the adversarial attack
report_adv = classification_report(y_test_flat, y_pred_adv_flat, target_names=target_names)

# Calculate the accuracy for the adversarial attack
test_accuracy_adv = accuracy_score(y_test_flat, y_pred_adv_flat)
print("Test Accuracy (Adversarial):", test_accuracy_adv)

# Print the classification report
print("Classification Report (Adversarial):\n", report_adv)

# Save the generated adversarial samples to a CSV file
output_directory = r"E:\train"
np.savetxt(output_directory + "/adversarial_samplles.csv", X_test_adv, delimiter=",")


from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def evaluate_model(y_true, y_pred):
    # Calculate Precision, Recall, F1-score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Calculate Detection Rate (True Positive Rate)
    conf_matrix = confusion_matrix(y_true, y_pred)
    detection_rate = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # Calculate AUC Score
    auc_score = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')

    return precision, recall, f1, detection_rate, auc_score

# Evaluate non-adversarial case
precision_non_adv, recall_non_adv, f1_non_adv, detection_rate_non_adv, auc_score_non_adv = evaluate_model(y_test_flat, y_pred)

# Evaluate adversarial case
precision_adv, recall_adv, f1_adv, detection_rate_adv, auc_score_adv = evaluate_model(y_test_flat, y_pred_adv_flat)

# Print non-adversarial metrics
print("Non-Adversarial Metrics:")
print("Precision:", precision_non_adv)
print("Recall:", recall_non_adv)
print("F1-score:", f1_non_adv)
print("Detection Rate :", detection_rate_non_adv)
print("AUC Score:", auc_score_non_adv)
print()

# Print adversarial metrics
print("Adversarial Metrics:")
print("Precision:", precision_adv)
print("Recall:", recall_adv)
print("F1-score:", f1_adv)
print("Detection Rate :", detection_rate_adv)
print("AUC Score:", auc_score_adv)