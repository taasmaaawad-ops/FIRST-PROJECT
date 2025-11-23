------------------------------------------2017-------------------------------------------------


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read the CSV file into a DataFrame
df1 = pd.read_csv(r"E:\train\cleaned_dataset_2017.csv")

# Check for missing values and handle them (replace with mean or drop, depending on your preference)
df1 = df1.dropna()  # Drop rows with missing values

# Check for infinite values and replace them with NaN
df1.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values after handling missing and infinite values
df1 = df1.dropna()

# Select only the numerical columns for normalization
numerical_columns = df1.select_dtypes(include=['float64', 'int64']).columns

# Create a copy of the DataFrame for printing the values before normalization
original_df1 = df1.copy()

# Create a StandardScaler object
scaler = StandardScaler()

# Apply Z-score normalization to the numerical columns
df1[numerical_columns] = scaler.fit_transform(df1[numerical_columns])

# Save the normalized DataFrame to a new CSV file
normalized_output_file_path = r"E:\train\normalized_data_2017.csv"
df1.to_csv(normalized_output_file_path, index=False)

# Index=False is used to prevent pandas from writing row indices to the CSV file

# Print the original and normalized values
print("Original Data:")
print(original_df1.head())  # Print the first few rows of the original DataFrame
print("\nNormalized Data:")
print(df1.head())  # Print the first few rows of the normalized DataFrame






-----------------------------------------------2018---------------------------------------------------------


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read the CSV file into a DataFrame
df = pd.read_csv(r"E:\train\sampled_data_2018.csv")

# Check for missing values and handle them (replace with mean or drop, depending on your preference)
df = df.dropna()  # Drop rows with missing values

# Check for infinite values and replace them with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values after handling missing and infinite values
df = df.dropna()

# Select only the numerical columns for normalization
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Create a copy of the DataFrame for printing the values before normalization
original_df = df.copy()

# Create a StandardScaler object
scaler = StandardScaler()

# Apply Z-score normalization to the numerical columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Save the normalized DataFrame to a new CSV file
normalized_output_file_path = r"E:\train\normalized_data.csv"
df.to_csv(normalized_output_file_path, index=False)

# Index=False is used to prevent pandas from writing row indices to the CSV file

# Print the original and normalized values
print("Original Data:")
print(original_df.head())  # Print the first few rows of the original DataFrame
print("\nNormalized Data:")
print(df.head())  # Print the first few rows of the normalized DataFrame



------------------------------------------------2019---------------------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read the CSV file into a DataFrame
df2 = pd.read_csv(r"E:\train\sampled_data_2019.csv")

# Check for missing values and handle them (replace with mean or drop, depending on your preference)
df2 = df2.dropna()  # Drop rows with missing values

# Check for infinite values and replace them with NaN
df2.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values after handling missing and infinite values
df2 = df2.dropna()

# Select only the numerical columns for normalization
numerical_columns = df2.select_dtypes(include=['float64', 'int64']).columns

# Create a copy of the DataFrame for printing the values before normalization
original_df2 = df2.copy()

# Create a StandardScaler object
scaler = StandardScaler()

# Apply Z-score normalization to the numerical columns
df2[numerical_columns] = scaler.fit_transform(df2[numerical_columns])

# Save the normalized DataFrame to a new CSV file
normalized_output_file_path = r"E:\train\normalized_data_2019.csv"
df2.to_csv(normalized_output_file_path, index=False)

# Index=False is used to prevent pandas from writing row indices to the CSV file

# Print the original and normalized values
print("Original Data:")
print(original_df2.head())  # Print the first few rows of the original DataFrame
print("\nNormalized Data:")
print(df2.head())  # Print the first few rows of the normalized DataFrame
