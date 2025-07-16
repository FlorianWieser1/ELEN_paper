#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
import argparse
import pandas as pd

def process_csv(input_file):
    # Read the CSV file into a pandas DataFrame.
    df = pd.read_csv(input_file)
    
    # Drop duplicate rows based on 'fname_pdb'
    # This keeps the first occurrence of each 'fname_pdb'
    df_unique = df.drop_duplicates(subset='fname_pdb', keep='first')
    
    # Select only the 'fname_pdb' and 'avg_per_chain' columns
    df_filtered = df_unique[['fname_pdb', 'avg_per_chain']]
    
    # Sort the DataFrame by 'avg_per_chain'
    df_sorted = df_filtered.sort_values(by='avg_per_chain')
    
    return df_sorted

def main():
    parser = argparse.ArgumentParser(description='Filter CSV by fname_pdb and avg_per_chain.')
    parser.add_argument('input_csv', help='Path to the input CSV file.')
    args = parser.parse_args()
    
    sorted_df = process_csv(args.input_csv)
    
    # Print the result to standard output
    print(sorted_df.to_csv(index=False))

if __name__ == '__main__':
    main()
