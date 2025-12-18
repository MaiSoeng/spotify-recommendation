import boto3
import time

# Configuration
REGION = 'us-east-1'
PROJECT_NAME = 'music-recommendation-v2'  # Update if different
ACCOUNT_ID = boto3.client('sts').get_caller_identity()['Account']

# Bucket Names (Must match CloudFormation)
RAW_DATA_BUCKET = f"{PROJECT_NAME}-raw-data-{ACCOUNT_ID}"
PROCESSED_DATA_BUCKET = f"{PROJECT_NAME}-processed-data-{ACCOUNT_ID}"

# Athena Configuration
DATABASE_NAME = f"{PROJECT_NAME}_analytics".replace('-', '_')
OUTPUT_LOCATION = f"s3://{PROJECT_NAME}-glue-scripts-{ACCOUNT_ID}/athena-results/"

athena = boto3.client('athena', region_name=REGION)

def execute_query(query, database=None):
    context = {}
    if database:
        context['Database'] = database
        
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext=context,
        ResultConfiguration={'OutputLocation': OUTPUT_LOCATION}
    )
    query_execution_id = response['QueryExecutionId']
    print(f"Execution ID: {query_execution_id}")
    
    # Wait for completion
    while True:
        stats = athena.get_query_execution(QueryExecutionId=query_execution_id)
        status = stats['QueryExecution']['Status']['State']
        if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            if status == 'FAILED':
                print(f"Query Failed: {stats['QueryExecution']['Status']['StateChangeReason']}")
            else:
                print(f"Query Succeeded: {status}")
            break
        time.sleep(1)

def setup_analytics():
    print(f"Setting up Athena Analytics for {PROJECT_NAME}...")
    
    # 1. Create Database
    create_db_query = f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}"
    execute_query(create_db_query)
    
    # 2. Create Table for Raw User Events
    # Adjust schema based on your actual generic-events structure
    raw_table_query = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {DATABASE_NAME}.user_events (
        user_id STRING,
        ts STRING,
        artist_id STRING,
        artist STRING,
        track_id STRING,
        track STRING
    )
    ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
    LOCATION 's3://{RAW_DATA_BUCKET}/'
    """
    execute_query(raw_table_query)
    
    # 3. Create Table for Training Data (Matrix Factorization format)
    # Assuming CSV/TSV format in processed bucket
    training_table_query = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {DATABASE_NAME}.training_data (
        user_id STRING,
        item_id STRING,
        rating FLOAT,
        timestamp BIGINT
    )
    ROW FORMAT DELIMITED
    FIELDS TERMINATED BY '\\t'
    STORED AS TEXTFILE
    LOCATION 's3://{PROCESSED_DATA_BUCKET}/neumf-training-data/train/'
    LOCATION 's3://{PROCESSED_DATA_BUCKET}/neumf-training-data/train/'
    """
    execute_query(training_table_query)

    # 4. Training Metrics Table
    # Points to SageMaker Model Artifacts (Output Data)
    # Note: We need to find the specific model output path dynamically, or user updates this
    # For simplicity, we assume a known common prefix or use a crawler. 
    # HERE, we assume the user will manually point QuickSight or we use a broad path
    metrics_query = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {DATABASE_NAME}.training_metrics (
        epoch INT,
        loss FLOAT,
        hr_10 FLOAT,
        ndcg_10 FLOAT,
        timestamp STRING
    )
    ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
    LOCATION 's3://{PROJECT_NAME}-model-artifacts-{ACCOUNT_ID}/models/' 
    """
    # Note: S3 structure is models/<job-name>/output/data/training_metrics.json
    # Athena recursive search might pick up other files if not careful, but JSONSerDe handles ignores well usually
    execute_query(metrics_query)

    # 5. Item Genres (Table Removed per User Request)
    # The raw data does not contain genre info, and we avoid simulated data.
    
    # 6. Sample Predictions
    preds_query = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {DATABASE_NAME}.predictions (
        user_id STRING,
        track STRING,
        rank INT,
        score FLOAT,
        prediction_date STRING
    )
    ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
    LOCATION 's3://{PROJECT_NAME}-model-artifacts-{ACCOUNT_ID}/models/'
    """
    execute_query(preds_query)
    
    print("Athena Setup Complete.")
    print(f"Database: {DATABASE_NAME}")
    print("You can now connect Amazon QuickSight to this Athena database.")

if __name__ == "__main__":
    setup_analytics()
