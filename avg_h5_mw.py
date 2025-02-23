import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

# Function to read the data from a text file
def read_data(file_path):
    data = pd.read_csv(file_path, sep='|', header=None, names=[
        'bottom_orcid', 'bottom_h5_index', 'bottom_subject', 'bottom_avg_cited_journal_hindex', 
        'random_orcid', 'random_h5_index', 'random_subject', 'random_avg_cited_journal_hindex'
    ])
    return data

# Function to extract the relevant columns and perform calculations
def process_data(data):
    bottom_avg_hindex = data['bottom_avg_cited_journal_hindex'].dropna()  # 4th column
    random_avg_hindex = data['random_avg_cited_journal_hindex'].dropna()  # 8th column

    # Calculate medians
    bottom_median = np.median(bottom_avg_hindex)
    random_median = np.median(random_avg_hindex)

    # Perform Mann-Whitney U test
    stat, p_value = mannwhitneyu(bottom_avg_hindex, random_avg_hindex)

    return bottom_median, random_median, stat, p_value

# Function to process data grouped by subject and perform Mann-Whitney U test
def process_data_by_subject(data):
    results = []
    grouped = data.groupby('bottom_subject')
    for subject, group in grouped:
        bottom_avg_hindex = group['bottom_avg_cited_journal_hindex'].dropna()
        random_avg_hindex = group['random_avg_cited_journal_hindex'].dropna()
        
        # Calculate medians
        bottom_median = np.median(bottom_avg_hindex)
        random_median = np.median(random_avg_hindex)
        
        # Perform Mann-Whitney U test
        stat, p_value = mannwhitneyu(bottom_avg_hindex, random_avg_hindex)
        
        results.append({
            'subject': subject,
            'bottom_median': bottom_median,
            'random_median': random_median,
            'stat': stat,
            'p_value': p_value
        })
        
    return results

# Main function
def main(file_path):
    data = read_data(file_path)
    bottom_median, random_median, stat, p_value = process_data(data)

    print(f'Overall Bottom Average H-Index Median: {bottom_median}')
    print(f'Overall Random Average H-Index Median: {random_median}')
    print(f'Overall Mann-Whitney U Test Statistic: {stat}')
    print(f'Overall P-Value: {p_value}')

    # Process data by subject
    subject_results = process_data_by_subject(data)
    print("\nResults by Subject:")
    for result in subject_results:
        print(f"\nSubject: {result['subject']}")
        print(f"Bottom Median: {result['bottom_median']}")
        print(f"Random Median: {result['random_median']}")
        print(f"Mann-Whitney U Test Statistic: {result['stat']}")
        print(f"P-Value: {result['p_value']}")
        
    # Write results in reports/avg_h5_mw.txt
    with open("reports/avg_h5_mw.txt", "w") as f:
        f.write(f'Overall Bottom Average H-Index Median: {bottom_median}\n')
        f.write(f'Overall Random Average H-Index Median: {random_median}\n')
        f.write(f'Overall Mann-Whitney U Test Statistic: {stat}\n')
        f.write(f'Overall P-Value: {p_value}\n\n')
        
        f.write("Results by Subject:\n")
        for result in subject_results:
            f.write(f"\nSubject: {result['subject']}\n")
            f.write(f"Bottom Median: {result['bottom_median']}\n")
            f.write(f"Random Median: {result['random_median']}\n")
            f.write(f"Mann-Whitney U Test Statistic: {result['stat']}\n")
            f.write(f"P-Value: {result['p_value']}\n")

# Replace 'data.txt' with the path to your text file
if __name__ == "__main__":
    file_path = 'reports/avg_h5_cited_issn.txt'
    main(file_path)
