import numpy as np
from scipy.stats import mannwhitneyu

def calculate_medians_and_mann_whitney(file_top, file_other):
    # Load data with number of cliques and max clique size
    top_data = np.genfromtxt(file_top, delimiter='\t', dtype=None, encoding=None, names=('subject', 'num_cliques', 'max_clique_size'))
    other_data = np.genfromtxt(file_other, delimiter='\t', dtype=None, encoding=None, names=('subject', 'num_cliques', 'max_clique_size'))
    
    # Calculate medians for number of cliques and max clique size
    top_num_cliques_median = np.median(top_data['num_cliques'])
    other_num_cliques_median = np.median(other_data['num_cliques'])
    
    top_max_clique_size_mean = np.mean(top_data['max_clique_size'])
    other_max_clique_size_mean = np.mean(other_data['max_clique_size'])
    
    # Perform Mann-Whitney U test for both number of cliques and max clique size
    u_stat_num_cliques, p_value_num_cliques = mannwhitneyu(top_data['num_cliques'], other_data['num_cliques'], alternative='two-sided')
    u_stat_max_clique_size, p_value_max_clique_size = mannwhitneyu(top_data['max_clique_size'], other_data['max_clique_size'], alternative='two-sided')
    
    print(f"Median Number of Cliques (Top): {top_num_cliques_median}")
    print(f"Median Number of Cliques (Other): {other_num_cliques_median}")
    print(f"U-Statistic (Number of Cliques): {u_stat_num_cliques}")
    print(f"P-Value (Number of Cliques): {p_value_num_cliques}")
    
    print(f"Mean Max Clique Size (Top): {top_max_clique_size_mean}")
    print(f"Mean Max Clique Size (Other): {other_max_clique_size_mean}")
    print(f"U-Statistic (Max Clique Size): {u_stat_max_clique_size}")
    print(f"P-Value (Max Clique Size): {p_value_max_clique_size}")
    
    # Write the results to a file
    with open('reports/mann_whitney_results.txt', 'w') as f:
        f.write(f"Median Number of Cliques (Top): {top_num_cliques_median}\n")
        f.write(f"Median Number of Cliques (Bottom): {other_num_cliques_median}\n")
        f.write(f"U-Statistic (Number of Cliques): {u_stat_num_cliques}\n")
        f.write(f"P-Value (Number of Cliques): {p_value_num_cliques}\n")
        
        f.write(f"Median Max Clique Size (Top): {top_max_clique_size_mean}\n")
        f.write(f"Median Max Clique Size (Bottom): {other_max_clique_size_mean}\n")
        f.write(f"U-Statistic (Max Clique Size): {u_stat_max_clique_size}\n")
        f.write(f"P-Value (Max Clique Size): {p_value_max_clique_size}\n")
    
    # Analyze by subject
    subjects = set(top_data['subject']).union(other_data['subject'])
    results = []
    for subject in subjects:
        top_subject_num_cliques = top_data[top_data['subject'] == subject]['num_cliques']
        other_subject_num_cliques = other_data[other_data['subject'] == subject]['num_cliques']
        
        top_subject_max_clique_size = top_data[top_data['subject'] == subject]['max_clique_size']
        other_subject_max_clique_size = other_data[other_data['subject'] == subject]['max_clique_size']
        
        median_top_num_cliques = np.median(top_subject_num_cliques) if len(top_subject_num_cliques) > 0 else float('nan')
        median_other_num_cliques = np.median(other_subject_num_cliques) if len(other_subject_num_cliques) > 0 else float('nan')
        
        mean_top_max_clique_size = np.mean(top_subject_max_clique_size) if len(top_subject_max_clique_size) > 0 else float('nan')
        mean_other_max_clique_size = np.mean(other_subject_max_clique_size) if len(other_subject_max_clique_size) > 0 else float('nan')
        
        if len(top_subject_num_cliques) > 0 and len(other_subject_num_cliques) > 0:
            u_stat_num_cliques, p_value_num_cliques = mannwhitneyu(top_subject_num_cliques, other_subject_num_cliques, alternative='two-sided')
        else:
            u_stat_num_cliques, p_value_num_cliques = float('nan'), float('nan')
        
        if len(top_subject_max_clique_size) > 0 and len(other_subject_max_clique_size) > 0:
            u_stat_max_clique_size, p_value_max_clique_size = mannwhitneyu(top_subject_max_clique_size, other_subject_max_clique_size, alternative='two-sided')
        else:
            u_stat_max_clique_size, p_value_max_clique_size = float('nan'), float('nan')
        
        results.append((subject, median_top_num_cliques, median_other_num_cliques, u_stat_num_cliques, p_value_num_cliques,
                        mean_top_max_clique_size, mean_other_max_clique_size, u_stat_max_clique_size, p_value_max_clique_size))
    
    # Print and save the results by subject
    with open('reports/cliques_mann_whitney_results.txt', 'a') as f:
        f.write("\nSubject\tMedian Num Cliques (Top)\tMedian Num Cliques (Bottom)\tU-Statistic (Num Cliques)\tP-Value (Num Cliques)\t")
        f.write("Mean Max Clique Size (Top)\Mean Max Clique Size (Bottom)\tU-Statistic (Max Clique Size)\tP-Value (Max Clique Size)\n")
        for result in results:
            print(f"Subject: {result[0]}, Median Num Cliques (Top): {result[1]}, Median Num Cliques (Bottom): {result[2]}, U-Statistic (Num Cliques): {result[3]}, P-Value (Num Cliques): {result[4]}")
            print(f"Subject: {result[0]}, Mean Max Clique Size (Top): {result[5]}, Mean Max Clique Size (Bottom): {result[6]}, U-Statistic (Max Clique Size): {result[7]}, P-Value (Max Clique Size): {result[8]}")
            if result[4] < 0.001 or result[8] < 0.001:
                print("The difference is statistically significant with P < 0.001")
            else:
                print("The difference is not statistically significant with P < 0.001")
            
            f.write(f"{result[0]}\t{result[1]}\t{result[2]}\t{result[3]}\t{result[4]}\t{result[5]}\t{result[6]}\t{result[7]}\t{result[8]}\n")
            
calculate_medians_and_mann_whitney(
    "reports/cliques-top.txt",
    "reports/cliques-bottom.txt"
)