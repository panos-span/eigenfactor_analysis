import sqlite3
import os
import sys
from pathlib import Path

def print_db_schema(db_path):
    """
    Connect to a SQLite database and print the schema for all tables.
    
    Args:
        db_path: Path to the SQLite database file
    """
    # Check if the database file exists
    if not os.path.isfile(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        sys.exit(1)
    
    # Print header info
    print(f"\nDatabase: {db_path}")
    print("=" * 80)
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in the database.")
            conn.close()
            return
        
        # Iterate through each table and print its schema
        for table in tables:
            table_name = table[0]
            
            # Skip internal SQLite tables if any
            if table_name.startswith('sqlite_'):
                continue
                
            print(f"\nTable: {table_name}")
            print("-" * 80)
            
            # Get the table schema details
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            # Create a dictionary of column names by position
            column_dict = {}
            for col in columns:
                column_dict[col[0]] = col[1]  # Map cid to name
            
            # Print column information
            print(f"{'Column Name':<20} {'Type':<15} {'Not Null':<10} {'Default Value':<15} {'Primary Key'}")
            print(f"{'-'*20:<20} {'-'*15:<15} {'-'*10:<10} {'-'*15:<15} {'-'*11}")
            
            for col in columns:
                col_id, name, col_type, not_null, default_val, pk = col
                not_null_text = "YES" if not_null else "NO"
                pk_text = "YES" if pk else "NO"
                default_text = str(default_val) if default_val is not None else "NULL"
                
                print(f"{name:<20} {col_type:<15} {not_null_text:<10} {default_text:<15} {pk_text}")
            
            # Check for indexes
            cursor.execute(f"PRAGMA index_list({table_name});")
            indexes = cursor.fetchall()
            
            if indexes:
                print("\nIndexes:")
                print(f"{'Index Name':<30} {'Unique':<10} {'Columns'}")
                print(f"{'-'*30:<30} {'-'*10:<10} {'-'*30}")
                
                for idx in indexes:
                    # Structure changed in newer SQLite versions
                    if len(idx) >= 3:
                        idx_name = idx[1]
                        is_unique = "YES" if idx[2] else "NO"
                    
                        # Get the columns in this index
                        cursor.execute(f"PRAGMA index_info({idx_name});")
                        idx_columns = cursor.fetchall()
                        
                        # Build column list carefully
                        column_names = []
                        for idx_col in idx_columns:
                            col_pos = idx_col[2]  # Position of the column
                            if col_pos in column_dict:
                                column_names.append(column_dict[col_pos])
                            else:
                                column_names.append(f"col_{col_pos}")
                        
                        print(f"{idx_name:<30} {is_unique:<10} {', '.join(column_names)}")
            
            # Check for foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            foreign_keys = cursor.fetchall()
            
            if foreign_keys:
                print("\nForeign Keys:")
                print(f"{'Column':<20} {'References':<30} {'On Update':<15} {'On Delete'}")
                print(f"{'-'*20:<20} {'-'*30:<30} {'-'*15:<15} {'-'*15}")
                
                for fk in foreign_keys:
                    if len(fk) >= 8:
                        fk_id, _, ref_table, from_col, to_col, on_update, on_delete, _ = fk
                        ref_info = f"{ref_table}({to_col})"
                        print(f"{from_col:<20} {ref_info:<30} {on_update:<15} {on_delete}")
        
        conn.close()
        print("\nSchema inspection complete.")
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Use the current directory to look for impact.db
    import argparse
    
    parser = argparse.ArgumentParser(description="Print the schema of a SQLite database.")
    # Add argument for database path
    parser.add_argument("db_path", nargs='?', default="impact.db", help="Path to the SQLite database file (default: impact.db)")
    args = parser.parse_args()
    
    
    db_path = args.db_path
    
    # Allow specifying a different path as command-line argument
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])
    
    print_db_schema(db_path)