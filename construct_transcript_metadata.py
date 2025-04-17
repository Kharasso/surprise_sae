import pandas as pd
import json
import logging
from datetime import timedelta, datetime

# Setup logging: adjust the level as needed (DEBUG, INFO, WARNING, etc.)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def processing_transcript_year_order(processing_year, order):
    try:
        # Read the transcript CSV file
        transcript_file = f"./transcript/{processing_year}_{order}.csv"
        year_transcript_df = pd.read_csv(transcript_file)
        logging.info(f"Successfully loaded transcript file: {transcript_file}")
    except Exception as e:
        logging.error(f"Error loading transcript file: {e}")
        return

    try:
        # Build metadata column list, then append additional columns.
        transcript_meta_cols = list(year_transcript_df.columns[1:])  # excluding the first column
        transcript_meta_cols.extend(['PERMNO', 'CUSIP', 'TICKER', 'GVKEY'])
        logging.info("Metadata columns list created.")
    except Exception as e:
        logging.error(f"Error constructing transcript metadata columns: {e}")
        return

    try:
        # Define the columns that uniquely identify a conference event.
        conference_cols = ['companyid', 'mostimportantdateutc', 'mostimportanttimeutc']

        # For each conference, determine the transcriptid to keep (keep the minimum transcriptid).
        chosen_transcripts = (
            year_transcript_df.groupby(conference_cols)['transcriptid']
            .min()
            .reset_index()
            .rename(columns={'transcriptid': 'chosen_transcriptid'})
        )
        logging.info("Chosen transcripts computed using minimum transcriptid.")
    except Exception as e:
        logging.error(f"Error computing chosen transcripts: {e}")
        return

    try:
        # Merge the chosen transcriptid back onto the original DataFrame.
        merged_df = pd.merge(year_transcript_df, chosen_transcripts, on=conference_cols, how='left')

        # Filter the DataFrame to keep only rows where transcriptid matches the chosen_transcriptid.
        year_transcript_df_filtered = merged_df[merged_df['transcriptid'] == merged_df['chosen_transcriptid']].copy()
        
        # Cast columns to int and drop unwanted columns
        year_transcript_df_filtered['companyid'] = year_transcript_df_filtered['companyid'].astype(int)
        year_transcript_df_filtered['transcriptid'] = year_transcript_df_filtered['transcriptid'].astype(int)
        year_transcript_df_filtered['keydevid'] = year_transcript_df_filtered['keydevid'].astype(int)
        year_transcript_df_filtered.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
        year_transcript_df_filtered.rename(columns={'companyid': "COMPANYID"}, inplace=True)
        
        # Convert dates to datetime.
        year_transcript_df_filtered['mostimportantdateutc'] = pd.to_datetime(year_transcript_df_filtered['mostimportantdateutc'])
        logging.info("Transcript DataFrame merged and filtered successfully.")
    except Exception as e:
        logging.error(f"Error processing transcript DataFrame: {e}")
        return

    try:
        # These two dataframes should be defined previously or loaded from file.
        # For example, aggregated_merged_sp500_monthly_meta_df might be loaded externally.
        aggregated_merged_sp500_monthly_meta_df = pd.read_csv("./sp500_metadata.csv")
        aggregated_merged_sp500_monthly_meta_df['startdate'] = pd.to_datetime(aggregated_merged_sp500_monthly_meta_df['startdate'])
        aggregated_merged_sp500_monthly_meta_df['enddate'] = pd.to_datetime(aggregated_merged_sp500_monthly_meta_df['enddate'])
        logging.info("Loaded SP500 metadata and converted dates.")
    except Exception as e:
        logging.error(f"Error loading or processing SP500 metadata: {e}")
        return

    try:
        # Merge transcripts with metadata based on company id.
        merged = pd.merge(year_transcript_df_filtered, aggregated_merged_sp500_monthly_meta_df, on='COMPANYID', how='inner')
        
        # Define current date for rows where LINKENDDT is NaN (assuming column 'enddate' here).
        current_date = pd.Timestamp.today()
        merged['effective_linkend'] = merged['enddate'].fillna(current_date)
        
        # Filter rows where the transcript date is between startdate and effective_linkend.
        mask = (merged['mostimportantdateutc'] >= merged['startdate']) & (merged['mostimportantdateutc'] <= merged['effective_linkend'])
        merged_filtered = merged.loc[mask].copy()
        
        # Remove duplicates from the transcript side
        transcript_cols = year_transcript_df_filtered.columns.tolist()
        result = merged_filtered.drop_duplicates(subset=transcript_cols, keep='first')
        result['companyid'] = result['COMPANYID']
        result.reset_index(drop=True, inplace=True)
        result = result[transcript_meta_cols]
        logging.info("Merged and filtered transcripts with SP500 metadata.")
    except Exception as e:
        logging.error(f"Error merging and filtering with SP500 metadata: {e}")
        return

    try:
        # Define the date range for the surprise data filter.
        start_of_year = pd.Timestamp(f'{processing_year}-01-01')
        end_of_year_plus_90 = pd.Timestamp(f'{processing_year}-12-31') + pd.Timedelta(days=90)
        
        # Assuming surprise_w_compid_df is loaded from file or defined previously.
        surprise_w_compid_df = pd.read_csv("./surprise_data.csv")
        # Convert 'EPSDATS' to datetime.
        surprise_w_compid_df['EPSDATS'] = pd.to_datetime(surprise_w_compid_df['EPSDATS'])
        
        # Prefilter the surprise dataset.
        surprise_filtered = surprise_w_compid_df[
            (surprise_w_compid_df['EPSDATS'] >= start_of_year) &
            (surprise_w_compid_df['EPSDATS'] < end_of_year_plus_90)
        ].copy()
        logging.info("Filtered surprise dataset based on processing year.")
    except Exception as e:
        logging.error(f"Error processing surprise dataset: {e}")
        return

    try:
        # Ensure that mostimportantdateutc is in datetime format.
        result['mostimportantdateutc'] = pd.to_datetime(result['mostimportantdateutc'])
        
        # Compute matching boundaries.
        result['lower_bound'] = result['mostimportantdateutc'] + pd.DateOffset(months=1)
        result['upper_bound'] = result['mostimportantdateutc'] + pd.Timedelta(days=90)
        
        # Merge the result with the surprise_filtered DataFrame on company id.
        merged = result.merge(surprise_filtered, left_on='companyid', right_on='COMPANYID', how='left')
        
        # Filter merged DataFrame by the matching date condition.
        mask = (merged['EPSDATS'] >= merged['lower_bound']) & (merged['EPSDATS'] < merged['upper_bound'])
        merged_filtered = merged[mask].copy()
        
        # Select the desired result columns.
        # Note: Added a missing comma between 'word_count' and 'ANNDATS'
        res_cols = ['companyid', 'keydevid', 'transcriptid', 'componentorder', 
                    'transcriptcomponenttypeid', 'mostimportantdateutc', 'word_count',
                    'ANNDATS', 'OFTIC', 'EPSDATS', 'ACTUAL', 'SUESCORE', 'SURPMEAN', 
                    'SURPSTDEV', 'componenttext']
        output_df = merged_filtered[res_cols].drop_duplicates(subset=['transcriptid', 'componentorder'])
        logging.info("Successfully merged and filtered all data for transcripts.")
    except Exception as e:
        logging.error(f"Error during final merging and filtering process: {e}")
        return

    try:
        # Write metadata CSV (all columns except 'componenttext')
        metadata_df = output_df.drop(columns=['componenttext'])
        csv_filename = f"./data/train_test_data/transcript_metadata_{processing_year}_{order}.csv"
        metadata_df.to_csv(csv_filename, index=False)
        logging.info(f"Metadata CSV exported as {csv_filename}")
    except Exception as e:
        logging.error(f"Error exporting metadata CSV: {e}")
        return

    try:
        # Write JSONL file for component texts
        jsonl_filename = f"./data/train_test_data/transcript_componenttext_{processing_year}_{order}.jsonl"
        with open(jsonl_filename, 'w', encoding='utf-8') as jsonl_file:
            for _, row in output_df.iterrows():
                # Create key from specified columns using underscore as delimiter.
                key = f"{row['companyid']}_{row['keydevid']}_{row['transcriptid']}_{row['componentorder']}_{row['transcriptcomponenttypeid']}"
                entry = {key: row['componenttext']}
                jsonl_file.write(json.dumps(entry) + "\n")
        logging.info(f"Component text JSONL exported as {jsonl_filename}")
    except Exception as e:
        logging.error(f"Error exporting component text JSONL: {e}")
        return

    logging.info("Processing complete.")