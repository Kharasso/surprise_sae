import pandas as pd
import os
import logging
import argparse
import json
from datetime import datetime

# ----- Parse command-line arguments -----
parser = argparse.ArgumentParser(description="Process transcript CSV files.")
parser.add_argument("--input_csv", required=True, help="Path to the large input CSV file.")
parser.add_argument("--tickers_csv", default="tickers_big_mid_cap_with_companyid.csv", 
                    help="Path to the tickers CSV file with allowed company IDs.")
parser.add_argument("--start_index", type=int, default=0, 
                    help="Row index (starting at 0 for header) to start processing from.")
args = parser.parse_args()

# ----- Setup file names and logging -----
input_basename = os.path.basename(args.input_csv).split('.')[0]
run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"processing_{input_basename}_start{args.start_index}_{run_datetime}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logging.info("Starting processing...")

# ----- Define paths and parameters -----
metadata_csv = f"transcript_metadata_{input_basename}.csv"   # output CSV for simplified metadata
output_json = f"transcripts_{input_basename}.jsonl"            # JSON Lines file for text records
chunksize = 1000000                          # number of rows per chunk
flush_interval = 100                         # flush metadata CSV every 100 rows

# Create the directory for metadata (if needed) or ensure the JSON Lines file is empty
with open(output_json, "w", encoding="utf-8") as out_file:
    pass  # create/empty the JSON Lines file

# ----- Read allowed company IDs from tickers CSV -----
try:
    allowed_companyids = pd.read_csv(args.tickers_csv, usecols=["COMPANYID"])["COMPANYID"].unique().tolist()
    logging.info(f"Loaded {len(allowed_companyids)} allowed company IDs from {args.tickers_csv}.")
except Exception as e:
    logging.error(f"Error reading tickers CSV: {e}")
    raise

# ----- Define the metadata columns -----
metadata_cols = [
    "companyid",
    "keydevid",
    "transcriptid",
    "componentorder",
    "transcriptcomponentid",
    "transcriptcomponenttypeid",
    "mostimportantdateutc"  # transcript date column
]

# Write header to the metadata CSV
try:
    with open(metadata_csv, "w", encoding="utf-8") as meta_file:
        meta_file.write(",".join(metadata_cols) + "\n")
    logging.info("Metadata CSV header written.")
except Exception as e:
    logging.error(f"Error writing header to metadata CSV: {e}")
    raise

# ----- Initialize metadata buffer -----
metadata_buffer = []

def flush_metadata_buffer(buffer):
    """Flush the metadata buffer to the CSV file and clear the buffer."""
    if buffer:
        try:
            df_buffer = pd.DataFrame(buffer, columns=metadata_cols)
            df_buffer.to_csv(metadata_csv, mode="a", header=False, index=False)
            buffer.clear()
            logging.info("Flushed metadata buffer to file.")
        except Exception as e:
            logging.error(f"Error flushing metadata buffer: {e}")

# ----- Determine skiprows for resuming processing -----
skiprows = None
if args.start_index > 0:
    skiprows = range(1, args.start_index + 1)
    logging.info(f"Starting processing from row index {args.start_index}; skipping rows 1 to {args.start_index}.")

# ----- Process the input CSV in chunks -----
chunk_number = 0
try:
    for chunk in pd.read_csv(args.input_csv, chunksize=chunksize, skiprows=skiprows):
        chunk_number += 1
        logging.info(f"Processing chunk {chunk_number} with {len(chunk)} rows.")
        
        # Filter rows to only include allowed company IDs
        filtered_chunk = chunk[chunk["companyid"].isin(allowed_companyids)]
        logging.info(f"Chunk {chunk_number}: {len(filtered_chunk)} rows after filtering by allowed company IDs.")
    
        # Process each row in the filtered chunk
        for _, row in filtered_chunk.iterrows():
            try:
                # Add metadata row to buffer
                metadata_row = [row[col] for col in metadata_cols]
                metadata_buffer.append(metadata_row)
                if len(metadata_buffer) >= flush_interval:
                    flush_metadata_buffer(metadata_buffer)
            except Exception as e:
                logging.error(f"Error processing metadata for row: {e}")
            
            try:
                # Parse transcript date (format: YYYY-mm-dd)
                transcript_date = datetime.strptime(row["mostimportantdateutc"], "%Y-%m-%d")
                date_str = transcript_date.strftime("%Y%m%d")
            except Exception as e:
                logging.error(f"Error parsing date for companyid {row.get('companyid')}: {e}")
                date_str = "unknown_date"
            
            try:
                # Convert numeric fields to int
                companyid = int(row["companyid"])
                transcriptid = int(row["transcriptid"])
                componentorder = int(row["componentorder"])
                transcriptcomponentid = int(row["transcriptcomponentid"])
                # Construct the key string (without .txt suffix)
                key_str = f"{companyid}_{transcriptid}_{componentorder}_{transcriptcomponentid}_{date_str}"
            except Exception as e:
                logging.error(f"Error creating key for row: {e}")
                continue

            try:
                # Write the component text to the JSON Lines file
                component_text = str(row["componenttext"])
                record = {key_str: component_text}
                with open(output_json, "a", encoding="utf-8") as out_file:
                    out_file.write(json.dumps(record) + "\n")
            except Exception as e:
                logging.error(f"Error writing record for key {key_str}: {e}")
        
        # Flush any remaining metadata in the buffer after processing each chunk
        flush_metadata_buffer(metadata_buffer)
        
        # After processing the first chunk, set skiprows to None to avoid skipping further rows
        skiprows = None

    logging.info("All chunks processed successfully.")
except Exception as e:
    logging.error(f"Error processing input CSV: {e}")

logging.info("Processing complete. Simplified metadata CSV and JSON Lines file for allowed company IDs have been created.")
