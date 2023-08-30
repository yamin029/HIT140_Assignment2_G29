import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns

# Step 1: Data Loading and Exploration:
# Load the dataset
data_path = "/Users/yaminhossain/Desktop/CDU/Data_Science/Assignment2/po1_data.txt"
column_names = [
    "Subject_ID", "Jitter%", "Jitter_Abs", "Jitter_RAP", "Jitter_PPQ5",
    "Jitter_DDP", "Shimmer%", "Shimmer_Abs", "Shimmer_APQ3", "Shimmer_APQ5",
    "Shimmer_APQ11", "Shimmer_DDA", "Harmonicity_AC", "Harmonicity_NHR",
    "Harmonicity_HNR", "Pitch_Median", "Pitch_Mean", "Pitch_Std",
    "Pitch_Min", "Pitch_Max", "Pulse_Num", "Pulse_Period", "Pulse_Mean",
    "Pulse_Std", "Voice_Unvoiced", "Voice_Num_Breaks","Voice_Deg_Breaks", "UPDRS", "PD_Indicator"
]
df = pd.read_csv(data_path, names=column_names)

# Display the first few rows of the dataset
print(df.head())

# Step 2: Data Wrangling:
# Check for missing values
print(df.isnull().sum())

# Step 3: Group Comparison:
# Create separate DataFrames for PD patients and healthy individuals
pd_patients = df[df['PD_Indicator'] == 1]
healthy_individuals = df[df['PD_Indicator'] == 0]

# Step 4: Descriptive Analysis:
# Descriptive statistics for PD patients
pd_stats = pd_patients.describe()

# Descriptive statistics for healthy individuals
healthy_stats = healthy_individuals.describe()

# Display the statistics
print(pd_stats)
print(healthy_stats)

# Salma Fariha Eera
# Step 5: Inferential Analysis:
# Perform t-tests for each variable
p_values = {}
# Exclude Subject identifier and PD indicator columns
for column in df.columns[1:-2]: 
    print(column)
    p_value = st.ttest_ind(pd_patients[column], healthy_individuals[column]).pvalue
    p_values[column] = p_value

# Display p-values
print(p_values)

# Guldeep 
# Step 6: Feature Selection:
# Set the significance level (alpha)
significance_level = 0.05

# Select features with low p-values (below significance level)
selected_features = [feature for feature, p_value in p_values.items() if p_value < significance_level]

# Zainab Fatima
# Display selected features
print("Selected features:", selected_features)
for feature in selected_features:
    plt.figure(figsize=(8, 6))
    plt.hist(pd_patients[feature], bins=20, alpha=0.5, label='PD Patients')
    plt.hist(healthy_individuals[feature], bins=20, alpha=0.5, label='Healthy Individuals')
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()