------------------------------------------2017-------------------------------------------------


import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Load the normalized data from the CSV file
normalized_data_path = r"E:\train\normalized_data_2017.csv"
df1_normalized = pd.read_csv(normalized_data_path)

# Separate labels from features
labels = df1_normalized['Label']
features = df1_normalized.drop(columns=['Label'])

# Perform SVD on the feature matrix
n_components = 10  # Number of components to keep
svd = TruncatedSVD(n_components=n_components)
extracted_features = svd.fit_transform(features)

# Create a DataFrame for the extracted features
df1_extracted_features = pd.DataFrame(extracted_features, columns=[f'SVD_Component_{i+1}' for i in range(n_components)])

# Add the labels to the DataFrame
df1_extracted_features['Label'] = labels

# Save the extracted features with labels to a new CSV file
extracted_features_output_path = r"E:\train\extracted_features_2017.csv"
df1_extracted_features.to_csv(extracted_features_output_path, index=False)

# Print the values of the extracted features with labels
print("Extracted Features with Labels:")
print(df1_extracted_features.head())






-----------------------------------------------2018---------------------------------------------------------


import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Load the normalized data from the CSV file
normalized_data_path = r"E:\train\normalized_data.csv"
df_normalized = pd.read_csv(normalized_data_path)

# Separate labels from features
labels = df_normalized['Label']
features = df_normalized.drop(columns=['Label','Timestamp'])

# Perform SVD on the feature matrix
n_components = 10  # Number of components to keep
svd = TruncatedSVD(n_components=n_components)
extracted_features = svd.fit_transform(features)

# Create a DataFrame for the extracted features
df_extracted_features = pd.DataFrame(extracted_features, columns=[f'SVD_Component_{i+1}' for i in range(n_components)])

# Add the labels to the DataFrame
df_extracted_features['Label'] = labels

# Save the extracted features with labels to a new CSV file
extracted_features_output_path = r"E:\train\extracted_features.csv"
df_extracted_features.to_csv(extracted_features_output_path, index=False)

# Print the values of the extracted features with labels
print("Extracted Features with Labels:")
print(df_extracted_features.head())



------------------------------------------------2019---------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Load the normalized data from the CSV file
normalized_data_path = r"E:\train\normalized_data_2019.csv"
df2_normalized = pd.read_csv(normalized_data_path)

# Separate labels from features
labels = df2_normalized[' Label']
# Drop non-numeric columns and other columns you don't need for SVD
features = df2_normalized.drop(columns=[' Label',' Timestamp','Flow ID',' Source IP',' Destination IP'])

# Check for non-numeric columns
non_numeric_columns = features.select_dtypes(exclude=[np.number]).columns
print("Non-numeric columns:", non_numeric_columns)

# Drop non-numeric columns
features = features.select_dtypes(include=[np.number])

# Perform SVD on the feature matrix
n_components = 10  # Number of components to keep
svd = TruncatedSVD(n_components=n_components)
extracted_features = svd.fit_transform(features)

# Create a DataFrame for the extracted features
df2_extracted_features = pd.DataFrame(extracted_features, columns=[f'SVD_Component_{i+1}' for i in range(n_components)])

# Add the labels to the DataFrame
df2_extracted_features[' Label'] = labels

# Save the extracted features with labels to a new CSV file
extracted_features_output_path = r"E:\train\extracted_features_2019.csv"
df2_extracted_features.to_csv(extracted_features_output_path, index=False)

# Print the values of the extracted features with labels
print("Extracted Features with Labels:")
print(df2_extracted_features.head())