"""
Glue Python Shell Job: Download Last.fm 1K Dataset

Professional data engineering approach for downloading large datasets:
- No Lambda time/memory limits
- Native S3 integration
- 16GB memory with 1 DPU

Usage:
    Triggered by CodePipeline via CodeBuild or directly via Glue API
    
Arguments:
    --raw_data_bucket: S3 bucket for raw data
"""

import sys
import os
import json
import urllib.request
import tarfile
import logging
from datetime import datetime

import boto3
from awsglue.utils import getResolvedOptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get job arguments
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'raw_data_bucket'])

# Dataset configuration
DATASET_URL = 'http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz'
TSV_FILE = 'userid-timestamp-artid-artname-traid-traname.tsv'
DESCRIPTION = 'Last.fm 1K users with listening history (~19M records)'
CHUNK_SIZE = 50000

# Initialize S3 client
s3 = boto3.client('s3')
bucket = args['raw_data_bucket']


def upload_chunk(records, chunk_num):
    """Upload a chunk of records to S3 as TSV"""
    header = 'user_id\tts\tartist_id\tartist\ttrack_id\ttrack'
    lines = [header]
    
    for r in records:
        row = '\t'.join([
            str(r.get('user_id', '')),
            str(r.get('ts', '')),
            str(r.get('artist_id', '')),
            str(r.get('artist', '')),
            str(r.get('track_id', '')),
            str(r.get('track', ''))
        ])
        lines.append(row)
    
    key = f'lastfm/raw/chunks/chunk_{chunk_num:05d}.tsv'
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body='\n'.join(lines),
        ContentType='text/tab-separated-values'
    )
    logger.info(f'Uploaded {key} ({len(records)} records)')


def upload_playlists(user_tracks, chunk_num):
    """Upload user playlists as JSON for NeuMF training"""
    playlists = []
    
    for i, (user_id, tracks) in enumerate(user_tracks.items()):
        playlist = {
            'name': f'User {user_id}',
            'pid': chunk_num * 1000 + i,
            'user_id': user_id,
            'tracks': [
                {
                    'pos': j,
                    'track_uri': t['track_id'],
                    'track_name': t['track'],
                    'artist_name': t['artist']
                }
                for j, t in enumerate(tracks)
            ]
        }
        playlists.append(playlist)
    
    key = f'lastfm/playlists/chunk_{chunk_num:05d}.json'
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps({'playlists': playlists}),
        ContentType='application/json'
    )
    logger.info(f'Uploaded {key} ({len(playlists)} playlists)')


def download_and_process():
    """
    Download and process Last.fm 1K dataset.
    Format: user_id, timestamp, artist_id, artist_name, track_id, track_name
    """
    tmp_file = '/tmp/lastfm-1k.tar.gz'
    
    logger.info(f"Downloading {DESCRIPTION}...")
    logger.info(f"URL: {DATASET_URL}")
    
    # Download
    start_time = datetime.now()
    urllib.request.urlretrieve(DATASET_URL, tmp_file)
    download_time = (datetime.now() - start_time).seconds
    logger.info(f"Download completed in {download_time} seconds")
    
    # Process
    logger.info("Processing dataset...")
    stats = {
        'total': 0,
        'chunks': 0,
        'users': set(),
        'tracks': set(),
        'artists': set()
    }
    
    chunk = []
    chunk_num = 0
    user_tracks = {}
    
    with tarfile.open(tmp_file, 'r:gz') as tar:
        for member in tar.getmembers():
            if TSV_FILE in member.name:
                logger.info(f"Processing {member.name}...")
                
                file_obj = tar.extractfile(member)
                for line in file_obj:
                    try:
                        parts = line.decode('utf-8', errors='ignore').strip().split('\t')
                        if len(parts) < 6:
                            continue
                        
                        user_id, ts, artist_id, artist_name, track_id, track_name = parts[:6]
                        
                        record = {
                            'user_id': user_id,
                            'ts': ts,
                            'artist_id': artist_id,
                            'artist': artist_name,
                            'track_id': track_id,
                            'track': track_name
                        }
                        
                        chunk.append(record)
                        stats['total'] += 1
                        stats['users'].add(user_id)
                        stats['tracks'].add(track_id)
                        stats['artists'].add(artist_id)
                        
                        # Build user history for playlists
                        if user_id not in user_tracks:
                            user_tracks[user_id] = []
                        user_tracks[user_id].append(record)
                        
                        # Upload chunk when full
                        if len(chunk) >= CHUNK_SIZE:
                            upload_chunk(chunk, chunk_num)
                            chunk_num += 1
                            stats['chunks'] += 1
                            chunk = []
                            
                            # Upload playlists periodically
                            if len(user_tracks) > 500:
                                upload_playlists(user_tracks, chunk_num)
                                user_tracks = {}
                            
                            # Progress logging
                            if stats['total'] % 500000 == 0:
                                logger.info(f"Processed {stats['total']:,} records...")
                    
                    except Exception:
                        continue
                
                break  # Only process the main TSV file
    
    # Upload remaining data
    if chunk:
        upload_chunk(chunk, chunk_num)
        stats['chunks'] += 1
    
    if user_tracks:
        upload_playlists(user_tracks, chunk_num + 1)
    
    # Cleanup
    os.remove(tmp_file)
    
    return stats


def upload_metadata(stats):
    """Upload job metadata and statistics"""
    metadata = {
        'source': '1k',
        'dataset': DESCRIPTION,
        'total_records': stats['total'],
        'total_chunks': stats['chunks'],
        'unique_users': len(stats['users']),
        'unique_tracks': len(stats['tracks']),
        'unique_artists': len(stats['artists']),
        'processed_at': datetime.now().isoformat(),
        'job_name': args['JOB_NAME']
    }
    
    s3.put_object(
        Bucket=bucket,
        Key='lastfm/metadata.json',
        Body=json.dumps(metadata, indent=2),
        ContentType='application/json'
    )
    
    logger.info(f"Metadata uploaded: {json.dumps(metadata)}")
    return metadata


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info(f"Glue Python Shell Job: {args['JOB_NAME']}")
    logger.info(f"Target bucket: {bucket}")
    logger.info(f"Dataset: Last.fm 1K")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    # Download and process
    stats = download_and_process()
    
    # Upload metadata
    metadata = upload_metadata(stats)
    
    # Summary
    elapsed = (datetime.now() - start_time).seconds
    logger.info("=" * 60)
    logger.info("Job completed successfully!")
    logger.info(f"Total records: {stats['total']:,}")
    logger.info(f"Total chunks: {stats['chunks']}")
    logger.info(f"Unique users: {len(stats['users']):,}")
    logger.info(f"Unique tracks: {len(stats['tracks']):,}")
    logger.info(f"Elapsed time: {elapsed} seconds")
    logger.info("=" * 60)
    
    return metadata


if __name__ == '__main__':
    main()
