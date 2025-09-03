import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_top_journals_by_work_percentage(db_path_rolap='rolap.db', db_path_impact='impact.db', 
                                        work_percentage=0.25, save_to_db=True):
    """
    Select journals that collectively account for the top X% of works sorted by Eigenfactor score.
    
    Args:
        db_path_rolap: Path to ROLAP SQLite database with works tables
        db_path_impact: Path to IMPACT SQLite database with Eigenfactor scores
        work_percentage: Percentage of total works to include (e.g., 0.25 for top 25%)
        save_to_db: Whether to save results to the database
        
    Returns:
        DataFrame with selected journals
    """
    print(f"Selecting journals for top {work_percentage*100:.1f}% of works...")
    
    # Connect to databases
    conn_rolap = sqlite3.connect(db_path_rolap)
    conn_impact = sqlite3.connect(db_path_impact)
    
    # Step 1: Get work counts per journal
    print("Getting work counts per journal...")
    journal_work_counts = pd.read_sql_query("""
        SELECT 
            issn,
            subject,
            COUNT(DISTINCT doi) AS work_count
        FROM 
            works_issn_subject
        GROUP BY 
            issn, subject
    """, conn_rolap)
    
    # Calculate total works
    total_works = journal_work_counts['work_count'].sum()
    print(f"Total works: {total_works}")
    
    # Step 2: Get Eigenfactor scores
    print("Getting Eigenfactor scores...")
    eigenfactor_scores = pd.read_sql_query("""
        SELECT issn, subject, eigenfactor_score
        FROM eigenfactor_scores
    """, conn_impact)
    
    # Check and fix data types for merge
    print("Checking data types...")
    print(f"Journal work counts 'subject' type: {journal_work_counts['subject'].dtype}")
    print(f"Eigenfactor scores 'subject' type: {eigenfactor_scores['subject'].dtype}")
    
    # Convert both to string to ensure compatibility
    journal_work_counts['subject'] = journal_work_counts['subject'].astype(str)
    eigenfactor_scores['subject'] = eigenfactor_scores['subject'].astype(str)
    
    print("After conversion:")
    print(f"Journal work counts 'subject' type: {journal_work_counts['subject'].dtype}")
    print(f"Eigenfactor scores 'subject' type: {eigenfactor_scores['subject'].dtype}")
    
    # Step 3: Merge data
    print("Merging datasets...")
    merged_data = pd.merge(
        journal_work_counts, 
        eigenfactor_scores,
        on=['issn', 'subject'], 
        how='inner'
    )
    
    # Check how many journals have Eigenfactor scores
    journal_count = len(journal_work_counts)
    matched_count = len(merged_data)
    print(f"Journals with work counts: {journal_count}")
    print(f"Journals with Eigenfactor scores: {matched_count}")
    print(f"Match rate: {matched_count/journal_count*100:.1f}%")
    
    # If match rate is very low, that might indicate issues with the join
    if matched_count/journal_count < 0.1:  # Less than 10% match
        print("WARNING: Very low match rate. Check if ISSN and subject formats match between databases!")
        
        # Let's examine some examples for debugging
        print("\nExample records from journal_work_counts:")
        print(journal_work_counts.head())
        
        print("\nExample records from eigenfactor_scores:")
        print(eigenfactor_scores.head())
        
        # Let's also try a more flexible join on just ISSN
        print("\nTrying a join on ISSN only...")
        issn_only_merge = pd.merge(
            journal_work_counts, 
            eigenfactor_scores,
            on=['issn'], 
            how='inner'
        )
        print(f"Match count with ISSN-only join: {len(issn_only_merge)}")
    
    # Step 4: Sort by Eigenfactor score and calculate cumulative work counts
    print("Sorting and calculating cumulative works...")
    merged_data = merged_data.sort_values('eigenfactor_score', ascending=False)
    merged_data['cumulative_works'] = merged_data['work_count'].cumsum()
    
    # Step 5: Select journals that make up the top X% of works
    threshold = total_works * work_percentage
    top_journals = merged_data[merged_data['cumulative_works'] <= threshold].copy()
    
    # If we didn't get enough journals to reach our threshold, explain why
    if len(top_journals) == 0:
        print("ERROR: No journals selected. This could be due to data matching issues.")
        return pd.DataFrame()
    
    if top_journals['work_count'].sum() < threshold and len(top_journals) < len(merged_data):
        print("\nWARNING: Selected journals don't reach the target threshold.")
        print(f"Target: {threshold} works, Selected: {top_journals['work_count'].sum()} works")
        print("This could be due to large journals with many works.")
    
    # Add percentage of total works
    top_journals['percentage_of_total'] = top_journals['work_count'] / total_works * 100
    
    # Print summary statistics
    journal_count = len(top_journals)
    work_count = top_journals['work_count'].sum()
    actual_percentage = work_count / total_works
    min_ef = top_journals['eigenfactor_score'].min()
    
    print(f"\nResults:")
    print(f"Selected {journal_count} journals ({journal_count/len(merged_data)*100:.1f}% of all journals)")
    print(f"These journals account for {work_count} works ({actual_percentage*100:.1f}% of all works)")
    print(f"Minimum Eigenfactor score in selection: {min_ef}")
    
    # Save to database if requested
    if save_to_db:
        print("Saving to database...")
        try:
            # Create table for top journals
            cursor = conn_rolap.cursor()
            cursor.execute("DROP TABLE IF EXISTS top_issn_by_subject")
            cursor.execute("""
                CREATE TABLE top_issn_by_subject (
                    issn TEXT,
                    subject TEXT,
                    PRIMARY KEY (issn, subject)
                )
            """)
            
            # Insert data
            for _, row in tqdm(top_journals.iterrows(), total=len(top_journals)):
                cursor.execute(
                    "INSERT INTO top_issn_by_subject (issn, subject) VALUES (?, ?)",
                    (row['issn'], row['subject'])
                )
            
            conn_rolap.commit()
            print(f"Saved {journal_count} journals to top_issn_by_subject table")
            
            # Similarly, create bottom journals table (remaining journals)
            bottom_journals = merged_data[merged_data['cumulative_works'] > threshold].copy()
            cursor.execute("DROP TABLE IF EXISTS bottom_issn_by_subject")
            cursor.execute("""
                CREATE TABLE bottom_issn_by_subject (
                    issn TEXT,
                    subject TEXT,
                    PRIMARY KEY (issn, subject)
                )
            """)
            
            # Insert data
            for _, row in tqdm(bottom_journals.iterrows(), total=len(bottom_journals)):
                cursor.execute(
                    "INSERT INTO bottom_issn_by_subject (issn, subject) VALUES (?, ?)",
                    (row['issn'], row['subject'])
                )
            
            conn_rolap.commit()
            print(f"Saved {len(bottom_journals)} journals to bottom_issn_by_subject table")
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    # Close connections
    conn_rolap.close()
    conn_impact.close()
    
    # Save summary to CSV
    print("Saving summary to CSV...")
    top_journals.to_csv('top_journals_by_work_percentage.csv', index=False)
    
    # Return selected journals
    return top_journals

def get_subject_specific_top_journals(db_path_rolap='rolap.db', db_path_impact='impact.db', 
                                     work_percentage=0.25, save_to_db=True):
    """
    Select journals that collectively account for the top X% of works within each subject.
    
    Args:
        db_path_rolap: Path to ROLAP SQLite database with works tables
        db_path_impact: Path to IMPACT SQLite database with Eigenfactor scores
        work_percentage: Percentage of total works to include per subject (e.g., 0.25 for top 25%)
        save_to_db: Whether to save results to the database
        
    Returns:
        DataFrame with selected journals
    """
    print(f"Selecting journals for top {work_percentage*100:.1f}% of works within each subject...")
    
    # Connect to databases
    conn_rolap = sqlite3.connect(db_path_rolap)
    conn_impact = sqlite3.connect(db_path_impact)
    
    # Step 1: Get work counts per journal
    print("Getting work counts per journal...")
    journal_work_counts = pd.read_sql_query("""
        SELECT 
            issn,
            subject,
            COUNT(DISTINCT doi) AS work_count
        FROM 
            works_issn_subject
        GROUP BY 
            issn, subject
    """, conn_rolap)
    
    # Step 2: Get Eigenfactor scores
    print("Getting Eigenfactor scores...")
    eigenfactor_scores = pd.read_sql_query("""
        SELECT issn, subject, eigenfactor_score
        FROM eigenfactor_scores
    """, conn_impact)
    
    # Convert both to string to ensure compatibility
    journal_work_counts['subject'] = journal_work_counts['subject'].astype(str)
    eigenfactor_scores['subject'] = eigenfactor_scores['subject'].astype(str)
    
    # Step 3: Merge data
    print("Merging datasets...")
    merged_data = pd.merge(
        journal_work_counts, 
        eigenfactor_scores,
        on=['issn', 'subject'], 
        how='inner'
    )
    
    # Check how many journals have Eigenfactor scores
    journal_count = len(journal_work_counts)
    matched_count = len(merged_data)
    print(f"Journals with work counts: {journal_count}")
    print(f"Journals with Eigenfactor scores: {matched_count}")
    print(f"Match rate: {matched_count/journal_count*100:.1f}%")
    
    # If match rate is very low, that might indicate issues with the join
    if matched_count/journal_count < 0.1:  # Less than 10% match
        print("WARNING: Very low match rate. Check if ISSN and subject formats match between databases!")
    
    # Step 4: Process each subject separately
    print("Processing each subject...")
    all_top_journals = []
    all_bottom_journals = []
    
    for subject, group in tqdm(merged_data.groupby('subject')):
        # Sort by Eigenfactor score
        group = group.sort_values('eigenfactor_score', ascending=False)
        
        # Calculate cumulative work counts within this subject
        group['cumulative_works'] = group['work_count'].cumsum()
        
        # Get total works for this subject
        subject_total_works = group['work_count'].sum()
        
        # Select journals that make up the top X% of works in this subject
        threshold = subject_total_works * work_percentage
        subject_top_journals = group[group['cumulative_works'] <= threshold].copy()
        subject_bottom_journals = group[group['cumulative_works'] > threshold].copy()
        
        # Add to overall lists
        all_top_journals.append(subject_top_journals)
        all_bottom_journals.append(subject_bottom_journals)
    
    # Combine results from all subjects
    top_journals = pd.concat(all_top_journals, ignore_index=True) if all_top_journals else pd.DataFrame()
    bottom_journals = pd.concat(all_bottom_journals, ignore_index=True) if all_bottom_journals else pd.DataFrame()
    
    if len(top_journals) == 0:
        print("ERROR: No journals selected. This could be due to data matching issues.")
        return pd.DataFrame()
    
    # Calculate overall statistics
    total_journals = len(merged_data)
    top_journal_count = len(top_journals)
    top_work_count = top_journals['work_count'].sum()
    total_works = merged_data['work_count'].sum()
    
    print(f"\nResults:")
    print(f"Selected {top_journal_count} journals ({top_journal_count/total_journals*100:.1f}% of all journals)")
    print(f"These journals account for {top_work_count} works ({top_work_count/total_works*100:.1f}% of all works)")
    
    # Save to database if requested
    if save_to_db:
        print("Saving to database...")
        try:
            # Create table for top journals
            cursor = conn_rolap.cursor()
            cursor.execute("DROP TABLE IF EXISTS top_issn_by_subject")
            cursor.execute("""
                CREATE TABLE top_issn_by_subject (
                    issn TEXT,
                    subject TEXT,
                    PRIMARY KEY (issn, subject)
                )
            """)
            
            # Insert data
            for _, row in tqdm(top_journals.iterrows(), total=len(top_journals)):
                cursor.execute(
                    "INSERT INTO top_issn_by_subject (issn, subject) VALUES (?, ?)",
                    (row['issn'], row['subject'])
                )
            
            conn_rolap.commit()
            print(f"Saved {top_journal_count} journals to top_issn_by_subject table")
            
            # Similarly, create bottom journals table
            cursor.execute("DROP TABLE IF EXISTS bottom_issn_by_subject")
            cursor.execute("""
                CREATE TABLE bottom_issn_by_subject (
                    issn TEXT,
                    subject TEXT,
                    PRIMARY KEY (issn, subject)
                )
            """)
            
            # Insert data
            for _, row in tqdm(bottom_journals.iterrows(), total=len(bottom_journals)):
                cursor.execute(
                    "INSERT INTO bottom_issn_by_subject (issn, subject) VALUES (?, ?)",
                    (row['issn'], row['subject'])
                )
            
            conn_rolap.commit()
            print(f"Saved {len(bottom_journals)} journals to bottom_issn_by_subject table")
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    # Close connections
    conn_rolap.close()
    conn_impact.close()
    
    # Save summary to CSV
    print("Saving summary to CSV...")
    top_journals.to_csv('top_journals_by_subject_specific.csv', index=False)
    
    # Return selected journals
    return top_journals

if __name__ == "__main__":
    try:
        # Option 1: Select journals for global top 25% of works
        print("\n--- Option 1: Global Top 25% of Works ---")
        top_journals_global = get_top_journals_by_work_percentage(
            db_path_rolap='rolap.db', 
            db_path_impact='impact.db',
            work_percentage=0.25, 
            save_to_db=False  # Set to True to save to database
        )
        
        # Option 2: Select journals for subject-specific top 25% of works
        print("\n--- Option 2: Subject-Specific Top 25% of Works ---")
        top_journals_subject = get_subject_specific_top_journals(
            db_path_rolap='rolap.db', 
            db_path_impact='impact.db',
            work_percentage=0.25, 
            save_to_db=True  # Save to database
        )
        
        print("\nAnalysis complete! Review the CSV files for detailed results.")
        print("The subject-specific top 25% approach has been saved to the database.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()