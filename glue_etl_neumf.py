"""
Glue ETL script for NeuMF training data preparation.
Processes Last.fm/Spotify data and outputs Parquet for SageMaker.
"""

import sys
import boto3
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
import pyspark.sql.functions as f
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, 
    IntegerType, LongType, ArrayType, FloatType
)
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get job arguments - database_secret_arn is optional
required_args = ['JOB_NAME', 'raw_data_bucket', 'processed_data_bucket', 'database_name']
optional_args = ['database_secret_arn']

args = getResolvedOptions(sys.argv, required_args)

# Handle optional arguments
for opt_arg in optional_args:
    try:
        args.update(getResolvedOptions(sys.argv, [opt_arg]))
    except:
        args[opt_arg] = ''


def get_database_credentials():
    """Retrieve database credentials from Secrets Manager."""
    secret_arn = args.get('database_secret_arn', '')
    if not secret_arn:
        logger.info("No database secret configured, skipping credential retrieval")
        return None
    
    try:
        secrets_client = boto3.client('secretsmanager')
        response = secrets_client.get_secret_value(SecretId=secret_arn)
        secret = json.loads(response['SecretString'])
        logger.info(f"Successfully retrieved database credentials for user: {secret.get('username')}")
        return secret
    except Exception as e:
        logger.warning(f"Failed to retrieve database credentials: {e}")
        return None


# Retrieve credentials at startup
db_credentials = get_database_credentials()
if db_credentials:
    logger.info(f"Database user: {db_credentials.get('username')}")

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger.info(f"Starting job: {args['JOB_NAME']}")
logger.info(f"Input: {args['raw_data_bucket']}, Output: {args['processed_data_bucket']}")


# Schema definitions

# Last.fm TSV schema
LASTFM_SCHEMA = StructType([
    StructField("user_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("artist_id", StringType(), True),
    StructField("artist_name", StringType(), True),
    StructField("track_id", StringType(), True),
    StructField("track_name", StringType(), True)
])

# Spotify playlist track schema
TRACK_SCHEMA = StructType([
    StructField("pos", IntegerType(), True),
    StructField("artist_name", StringType(), True),
    StructField("track_uri", StringType(), True),
    StructField("artist_uri", StringType(), True),
    StructField("track_name", StringType(), True),
    StructField("album_uri", StringType(), True),
    StructField("duration_ms", IntegerType(), True),
    StructField("album_name", StringType(), True)
])

PLAYLIST_SCHEMA = StructType([
    StructField("name", StringType(), True),
    StructField("collaborative", StringType(), True),
    StructField("pid", IntegerType(), True),
    StructField("modified_at", LongType(), True),
    StructField("num_tracks", IntegerType(), True),
    StructField("num_albums", IntegerType(), True),
    StructField("num_followers", IntegerType(), True),
    StructField("tracks", ArrayType(TRACK_SCHEMA), True),
    StructField("num_edits", IntegerType(), True),
    StructField("duration_ms", LongType(), True),
    StructField("num_artists", IntegerType(), True),
    StructField("user_id", StringType(), True)  # For Last.fm converted data
])


try:
    # ===============================
    # Load Raw Data (Last.fm or Spotify format)
    # ===============================
    logger.info("Loading raw data...")
    
    interactions_df = None
    data_source = "unknown"
    
    # Try Last.fm format first (preferred - larger dataset)
    lastfm_paths = [
        f"s3://{args['raw_data_bucket']}/lastfm/raw/interactions.tsv",
        f"s3://{args['raw_data_bucket']}/lastfm/playlists.json",
    ]
    
    spotify_paths = [
        f"s3://{args['raw_data_bucket']}/spotify-sample/playlists.json",
        f"s3://{args['raw_data_bucket']}/spotify-sample/",
    ]
    
    # Try Last.fm TSV format
    try:
        logger.info("Trying Last.fm TSV format...")
        # Point to directory containing chunks
        tsv_path = f"s3://{args['raw_data_bucket']}/lastfm/raw/chunks/"
        
        interactions_df = spark.read.csv(
            tsv_path,
            sep='\t',
            header=True,
            schema=LASTFM_SCHEMA
        )
        
        # Rename columns to match expected format
        interactions_df = interactions_df.select(
            f.col('user_id').alias('playlist_id'),
            f.lit(None).alias('playlist_name'),
            f.lit(0).alias('num_followers'),
            f.lit(0).alias('position'),
            f.col('track_id').alias('track_uri'),
            f.col('track_name'),
            f.col('artist_name'),
            f.col('artist_id').alias('artist_uri'),
            f.lit('Unknown').alias('album_name'),
            f.lit('').alias('album_uri'),
            f.lit(200000).alias('duration_ms')
        )
        
        row_count = interactions_df.count()
        logger.info(f"Loaded {row_count} interactions from Last.fm TSV")
        data_source = "lastfm_tsv"
        
    except Exception as e:
        logger.warning(f"Could not load Last.fm TSV: {e}")
        
        # Try Last.fm JSON (playlist format)
        try:
            logger.info("Trying Last.fm JSON playlist format...")
            # Point to directory containing chunks
            json_path = f"s3://{args['raw_data_bucket']}/lastfm/playlists/"
            
            raw_df = spark.read.json(json_path, multiLine=True)
            
            if 'playlists' in raw_df.columns:
                playlists_df = raw_df.select(f.explode(f.col('playlists')).alias('playlist'))
                playlists_df = playlists_df.select('playlist.*')
            else:
                playlists_df = raw_df
            
            # Explode tracks
            if 'tracks' in playlists_df.columns:
                interactions_df = playlists_df.select(
                    f.col('pid').alias('playlist_id'),
                    f.col('name').alias('playlist_name'),
                    f.col('user_id'),
                    f.explode(f.col('tracks')).alias('track')
                ).select(
                    'playlist_id',
                    'playlist_name',
                    f.lit(0).alias('num_followers'),
                    f.col('track.pos').alias('position'),
                    f.col('track.track_uri').alias('track_uri'),
                    f.col('track.track_name').alias('track_name'),
                    f.col('track.artist_name').alias('artist_name'),
                    f.col('track.artist_uri').alias('artist_uri'),
                    f.col('track.album_name').alias('album_name'),
                    f.col('track.album_uri').alias('album_uri'),
                    f.col('track.duration_ms').alias('duration_ms')
                )
            
            row_count = interactions_df.count()
            logger.info(f"Loaded {row_count} interactions from Last.fm JSON")
            data_source = "lastfm_json"
            
        except Exception as e2:
            logger.warning(f"Could not load Last.fm JSON: {e2}")
            
            # Fall back to Spotify format
            try:
                logger.info("Trying Spotify format...")
                raw_data_path = f"s3://{args['raw_data_bucket']}/spotify-sample/"
                
                raw_df = spark.read.json(raw_data_path, multiLine=True)
                
                if 'playlists' in raw_df.columns:
                    playlists_df = raw_df.select(f.explode(f.col('playlists')).alias('playlist'))
                    playlists_df = playlists_df.select('playlist.*')
                else:
                    playlists_df = raw_df
                
                interactions_df = playlists_df.select(
                    f.col('pid').alias('playlist_id'),
                    f.col('name').alias('playlist_name'),
                    f.col('num_followers'),
                    f.explode(f.col('tracks')).alias('track')
                ).select(
                    'playlist_id',
                    'playlist_name',
                    'num_followers',
                    f.col('track.pos').alias('position'),
                    f.col('track.track_uri').alias('track_uri'),
                    f.col('track.track_name').alias('track_name'),
                    f.col('track.artist_name').alias('artist_name'),
                    f.col('track.artist_uri').alias('artist_uri'),
                    f.col('track.album_name').alias('album_name'),
                    f.col('track.album_uri').alias('album_uri'),
                    f.col('track.duration_ms').alias('duration_ms')
                )
                
                row_count = interactions_df.count()
                logger.info(f"Loaded {row_count} interactions from Spotify format")
                data_source = "spotify"
                
            except Exception as e3:
                logger.error(f"Could not load any data format: {e3}")
                raise
    
    logger.info(f"Data source: {data_source}")
    logger.info("Data schema:")
    interactions_df.printSchema()
    
    total_interactions = interactions_df.count()
    logger.info(f"Total interactions: {total_interactions}")

    
    # ===============================
    # Data Quality Checks
    # ===============================
    logger.info("Performing data quality checks...")
    
    # Check for nulls in critical columns
    null_counts = {}
    for col_name in ['playlist_id', 'track_uri']:
        if col_name in interactions_df.columns:
            null_count = interactions_df.filter(f.col(col_name).isNull()).count()
            null_counts[col_name] = null_count
            if null_count > 0:
                logger.warning(f"Column '{col_name}' has {null_count} null values")
    
    # Remove invalid records
    valid_interactions_df = interactions_df.filter(
        f.col('playlist_id').isNotNull() & 
        f.col('track_uri').isNotNull()
    )
    
    valid_count = valid_interactions_df.count()
    logger.info(f"Valid interactions: {valid_count} ({valid_count/total_interactions*100:.1f}%)")
    
    # ===============================
    # Create Encodings
    # ===============================
    logger.info("Creating user and item encodings...")
    
    # Create dense user (playlist) IDs
    user_window = Window.orderBy('playlist_id')
    unique_users = valid_interactions_df.select('playlist_id').distinct() \
        .withColumn('user_idx', f.dense_rank().over(user_window) - 1)
    
    # Create dense item (track) IDs
    item_window = Window.orderBy('track_uri')
    unique_items = valid_interactions_df.select('track_uri', 'track_name', 'artist_name').distinct() \
        .withColumn('item_idx', f.dense_rank().over(item_window) - 1)
    
    num_users = unique_users.count()
    num_items = unique_items.count()
    logger.info(f"Unique users (playlists): {num_users}")
    logger.info(f"Unique items (tracks): {num_items}")
    
    # Join encodings back to interactions
    encoded_df = valid_interactions_df \
        .join(unique_users, 'playlist_id', 'left') \
        .join(unique_items, 'track_uri', 'left')
    
    # ===============================
    # Compute Item Popularity
    # ===============================
    logger.info("Computing item popularity statistics...")
    
    item_stats = encoded_df.groupBy('item_idx', 'track_uri', 'track_name', 'artist_name').agg(
        f.count('*').alias('play_count'),
        f.avg('position').alias('avg_position')
    )
    
    # Add popularity percentile
    popularity_window = Window.orderBy(f.col('play_count').desc())
    item_stats = item_stats.withColumn(
        'popularity_rank', 
        f.row_number().over(popularity_window)
    ).withColumn(
        'popularity_percentile',
        f.col('popularity_rank') / num_items
    )
    
    # ===============================
    # Create Train/Test Split
    # ===============================
    logger.info("Creating train/test split...")
    
    # Use 80/20 split based on random sampling
    train_df = encoded_df.sample(fraction=0.8, seed=42)
    test_df = encoded_df.subtract(train_df)
    
    train_count = train_df.count()
    test_count = test_df.count()
    logger.info(f"Train set: {train_count} interactions")
    logger.info(f"Test set: {test_count} interactions")
    
    # ===============================
    # Prepare Training Data
    # ===============================
    logger.info("Preparing final training data...")
    
    # Select columns needed for NeuMF training
    train_output = train_df.select(
        f.col('user_idx').cast('int'),
        f.col('item_idx').cast('int'),
        f.col('position').cast('int'),
        f.col('playlist_id'),
        f.col('track_uri'),
        f.col('track_name'),
        f.col('artist_name'),
        (f.col('user_idx').cast('int') % 20).alias('part_id')
    )
    
    test_output = test_df.select(
        f.col('user_idx').cast('int'),
        f.col('item_idx').cast('int'),
        f.col('position').cast('int'),
        f.col('playlist_id'),
        f.col('track_uri'),
        f.col('track_name'),
        f.col('artist_name'),
        (f.col('user_idx').cast('int') % 20).alias('part_id')
    )
    
    # ===============================
    # Write Outputs
    # ===============================
    output_bucket = args['processed_data_bucket']
    
    # Write training data
    train_path = f"s3://{output_bucket}/neumf-training-data/train/"
    logger.info(f"Writing training data to {train_path}")
    
    train_output.write \
        .mode('overwrite') \
        .partitionBy('part_id') \
        .parquet(train_path)
    
    # Write test data
    test_path = f"s3://{output_bucket}/neumf-training-data/test/"
    logger.info(f"Writing test data to {test_path}")
    
    test_output.write \
        .mode('overwrite') \
        .partitionBy('part_id') \
        .parquet(test_path)
    
    # Write user encodings
    user_encoding_path = f"s3://{output_bucket}/neumf-training-data/encodings/users/"
    logger.info(f"Writing user encodings to {user_encoding_path}")
    
    unique_users.write \
        .mode('overwrite') \
        .parquet(user_encoding_path)
    
    # Write item encodings with stats
    item_encoding_path = f"s3://{output_bucket}/neumf-training-data/encodings/items/"
    logger.info(f"Writing item encodings to {item_encoding_path}")
    
    item_stats.write \
        .mode('overwrite') \
        .parquet(item_encoding_path)
    
    # Write metadata
    metadata = {
        'num_users': num_users,
        'num_items': num_items,
        'train_interactions': train_count,
        'test_interactions': test_count,
        'total_interactions': valid_count,
        'null_counts': null_counts
    }
    
    metadata_path = f"s3://{output_bucket}/neumf-training-data/metadata/"
    metadata_df = spark.createDataFrame([metadata])
    metadata_df.write.mode('overwrite').json(metadata_path)
    
    logger.info("ETL job completed successfully")
    
    # Log final metrics
    logger.info("=" * 50)
    logger.info("ETL Job Summary:")
    logger.info(f"  Total playlists processed: {playlists_df.count()}")
    logger.info(f"  Total interactions: {total_interactions}")
    logger.info(f"  Valid interactions: {valid_count}")
    logger.info(f"  Unique users: {num_users}")
    logger.info(f"  Unique items: {num_items}")
    logger.info(f"  Train set size: {train_count}")
    logger.info(f"  Test set size: {test_count}")
    logger.info(f"  Output location: s3://{output_bucket}/neumf-training-data/")
    logger.info("=" * 50)

except Exception as e:
    logger.error(f"ETL job failed with error: {str(e)}")
    import traceback
    traceback.print_exc()
    raise e

finally:
    job.commit()
