import pandas as pd

def create_mini_csv(file_path):
    # Read the original CSV file
    data = pd.read_csv(file_path)
    
    # Extract the first 250 rows
    mini_data = data.head(250)
    
    # Generate the new filename
    new_file_path = file_path.replace('.csv', '_mini.csv')
    
    # Save the new CSV file
    mini_data.to_csv(new_file_path, index=False)
    
    print(f"Mini CSV file created: {new_file_path}")

# Example usage
file_path = "Salary_Data.csv"  # Replace with the actual path to your CSV file
create_mini_csv(file_path)
 