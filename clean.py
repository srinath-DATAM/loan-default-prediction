import pandas as pd 
import os 

def load_csv(file_path):
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    print(f"Total Rows and Columns: {df.shape}")
    return df

def clean_data(df):
    """Handle missing values and clean the dataset."""
    print("\n--- Starting Data Cleaning ---")
    print("Missing values before cleaning:")
    print(df.isnull().sum())
    
    # getting missing values for numeric columns 
    numeric_columns = df.select_dtypes(include=["float64", "int64"])
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
            
    categorical_columns = df.select_dtypes(include=["object"])
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    return df 

if __name__ == "__main__":
    file_path = r"C:\Users\sushm\AppData\Local\Temp\Loan Eligibility Prediction.csv"
    df = load_csv(file_path)
    
    # Inga 'file_path' irukkakoodathu, kandippa 'df' thaan irukkanum!
    df = clean_data(df)
    print("\nData cleaning completed successfully!")
    if not os.path.exists("data"):
        os.makedirs("data")
    
    output_file_path="data/cleaned_loan_data.csv"
    df.to_csv(output_file_path, index=False)
    print(f"Cleaned data saved to {output_file_path}")