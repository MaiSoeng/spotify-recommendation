import streamlit as st
import boto3
import json
import os
import pandas as pd
from decimal import Decimal

# Configuration
REGION = 'us-east-1'  # Update if different
PROJECT_NAME = st.sidebar.text_input("Project Name", value="music-recommendation-v2")

# Initialize Clients
@st.cache_resource
def get_aws_clients():
    session = boto3.Session(region_name=REGION)
    return {
        'dynamodb': session.resource('dynamodb'),
        'sagemaker': session.client('sagemaker-runtime'),
        'cloudformation': session.client('cloudformation')
    }

clients = get_aws_clients()

def get_resource_names(project_name):
    # Construct names based on convention
    # In a real app, you might fetch these from SSM or CloudFormation outputs
    return {
        'user_table': f"{project_name}-user-features",
        'track_table': f"{project_name}-track-metadata",
        # Endpoint name might have a timestamp, so we need to find the active one or use a specific one
        # For simplicity, we assume a deterministic name or search for it
        'endpoint_prefix': f"{project_name}-neumf-endpoint" 
    }

def find_active_endpoint(prefix):
    sagemaker = clients['sagemaker']
    # This is a runtime client, it doesn't have list_endpoints. 
    # We need a standard sagemaker client for management tasks.
    sm_client = boto3.client('sagemaker', region_name=REGION)
    
    try:
        # List endpoints filtering by name
        response = sm_client.list_endpoints(NameContains=prefix, StatusEquals='InService')
        endpoints = response['Endpoints']
        if endpoints:
            # Sort by creation time desc
            endpoints.sort(key=lambda x: x['CreationTime'], reverse=True)
            return endpoints[0]['EndpointName']
    except Exception as e:
        st.error(f"Error listing endpoints: {e}")
    return None

def get_user_history(user_id, table_name):
    table = clients['dynamodb'].Table(table_name)
    try:
        response = table.get_item(Key={'user_id': user_id})
        return response.get('Item')
    except Exception as e:
        st.error(f"Error fetching user history: {e}")
        return None

def get_track_info(track_ids, table_name):
    table = clients['dynamodb'].Table(table_name)
    tracks = {}
    # DynamoDB batch_get_item could be used here for efficiency
    for tid in track_ids:
        try:
            resp = table.get_item(Key={'track_id': tid})
            if 'Item' in resp:
                tracks[tid] = resp['Item']
        except:
            pass
    return tracks

def get_recommendations(endpoint_name, user_id, top_n=10):
    client = clients['sagemaker']
    payload = {
        'user_id': user_id,
        'n_recommendations': top_n,
        'include_scores': True
    }
    
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        result = json.loads(response['Body'].read().decode())
        return result
    except Exception as e:
        st.error(f"Inference error: {e}")
        return []

# UI Layout
st.title("🎵 NeuMF Music Recommender")

resource_names = get_resource_names(PROJECT_NAME)

# Sidebar Control
user_id = st.sidebar.text_input("User ID", value="user_000001")
top_n = st.sidebar.slider("Top N Recommendations", 5, 20, 10)

if st.sidebar.button("Refresh"):
    st.rerun()

# Main Content
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 User Profile")
    if user_id:
        user_data = get_user_history(user_id, resource_names['user_table'])
        if user_data:
            st.json(user_data)
            last_track = user_data.get('last_track')
            if last_track:
                st.info(f"Last Track ID: {last_track}")
        else:
            st.warning("User not found in Feature Store.")

with col2:
    st.subheader("🚀 Recommendations")
    
    # Auto-discover endpoint
    endpoint_name = find_active_endpoint(PROJECT_NAME)
    
    if endpoint_name:
        st.success(f"Connected to: `{endpoint_name}`")
        
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Invoking SageMaker Endpoint..."):
                recs = get_recommendations(endpoint_name, user_id, top_n)
                
                if recs:
                    # Parse recommendations
                    # Format is usually {'recommendations': [{'track_id':..., 'score':...}]} or list
                    if isinstance(recs, dict):
                        rec_list = recs.get('recommendations', [])
                    else:
                        rec_list = recs
                    
                    if rec_list:
                        # Enhance with metadata
                        track_ids = [r['track_id'] for r in rec_list]
                        track_info = get_track_info(track_ids, resource_names['track_table'])
                        
                        display_data = []
                        for r in rec_list:
                            tid = r['track_id']
                            info = track_info.get(tid, {})
                            display_data.append({
                                'Track ID': tid,
                                'Score': f"{r['score']:.4f}",
                                'Artist': info.get('artist_name', 'Unknown'),
                                'Track Name': info.get('track_name', tid) # Fallback if no metadata
                            })
                        
                        st.table(pd.DataFrame(display_data))
                    else:
                        st.info("No recommendations returned.")
    else:
        st.error("No active SageMaker Endpoint found. Please check your pipeline.")

with st.expander("Debugging Info"):
    st.write("Project:", PROJECT_NAME)
    st.write("Resources:", resource_names)
