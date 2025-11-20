import csv
import sys
import os

def detect_csv_delimiter(file_path):
    """
    Detects the delimiter of a CSV file.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    try:
        # Read a sample of the file to help the sniffer
        # Use 'latin-1' as a fallback encoding if UTF-8 fails, common for some CSVs
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            sample = f.read(4096) # Read first 4KB
        
        # Try sniffing with common delimiters
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=',;\t|')
        return f"Detected delimiter: '{dialect.delimiter}'"
    except UnicodeDecodeError:
        try:
            # Fallback to latin-1 if UTF-8 fails
            with open(file_path, 'r', newline='', encoding='latin-1') as f:
                sample = f.read(4096)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=',;\t|')
            return f"Detected delimiter (using latin-1 encoding): '{dialect.delimiter}'"
        except Exception as e:
            return f"Could not decode file with UTF-8 or latin-1. An unexpected error occurred: {e}"
    except csv.Error:
        return "Could not detect delimiter. The file might not be a standard CSV or the sample is too small."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_delimiter.py <path_to_csv_file>")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    print(detect_csv_delimiter(csv_file_path))
