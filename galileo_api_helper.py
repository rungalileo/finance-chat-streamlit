import requests
import logging
import time
import streamlit as st

def get_galileo_app_url() -> str:
    """
    Get the Galileo web console URL from secrets.
    
    Returns:
        str: The Galileo web console URL without trailing slash
    """
    galileo_url = st.secrets.get("galileo_console_url", "https://app.galileo.ai")
    # Remove trailing slash if present
    return galileo_url.rstrip('/')

def get_galileo_api_url() -> str:
    """
    Get the Galileo API URL from secrets.
    
    Returns:
        str: The Galileo API URL
    """
    galileo_url = get_galileo_app_url()
    
    # Extract domain without protocol
    if galileo_url.startswith("https://"):
        domain = galileo_url[8:]  # Remove https://
    elif galileo_url.startswith("http://"):
        domain = galileo_url[7:]  # Remove http://
    else:
        domain = galileo_url
    
    # Remove app. prefix if exists and any path components
    if domain.startswith("app."):
        domain = domain[4:]
    domain = domain.split('/')[0]  # Get just the domain part
    
    return f"https://api.{domain}"

def get_galileo_project_id(api_key: str, project_name: str, starting_token: int = 0, limit: int = 10) -> str:
    """
    Fetches the Galileo project ID for a given project name.

    Args:
        api_key (str): Your Galileo API key.
        project_name (str): The name of the project to search for.
        starting_token (int): The starting token for pagination.
        limit (int): The number of projects to fetch.

    Returns:
        str: The project ID if found, else None.
    """
    # Get the base URL from secrets
    galileo_url = get_galileo_app_url()
    
    url = f"{galileo_url}/api/galileo/public/v2/projects/paginated?starting_token={starting_token}&limit={limit}"
    headers = {
        "accept": "*/*",
        "galileo-api-key": api_key,
        "content-type": "application/json",
        "origin": galileo_url,
        "referer": f"{galileo_url}/",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    }
    data = {
        "sort": {
            "name": "updated_at",
            "ascending": False
        },
        "filters": []
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    for project in result.get("projects", []):
        if project.get("name") == project_name:
            return project.get("id")
    return None

def get_galileo_log_stream_id(api_key: str, project_id: str, log_stream_name: str) -> str:
    """
    Fetches the Galileo log stream ID for a given project ID and log stream name.

    Args:
        api_key (str): Your Galileo API key.
        project_id (str): The ID of the project.
        log_stream_name (str): The name of the log stream to search for.

    Returns:
        str: The log stream ID if found, else None.
    """
    # Get the base URL from secrets
    galileo_url = get_galileo_app_url()
    
    url = f"{galileo_url}/api/galileo/v2/projects/{project_id}/log_streams"
    headers = {
        "accept": "*/*",
        "galileo-api-key": api_key,
        "content-type": "application/json",
        "origin": galileo_url,
        "referer": f"{galileo_url}/",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    log_streams = response.json()  # This is now a list of log streams
    
    for stream in log_streams:  # Iterate directly over the list
        if stream.get("name") == log_stream_name:
            return stream.get("id")
    return None 

def list_galileo_experiments(api_key: str, project_id: str) -> list:
    """
    Lists all experiments for a given project.
    
    Args:
        api_key (str): Your Galileo API key.
        project_id (str): The ID of the project.
        
    Returns:
        list: A list of experiment objects with details.
    """
    # Get URLs
    api_url = get_galileo_api_url()
    app_url = get_galileo_app_url()
    
    url = f"{api_url}/v2/projects/{project_id}/experiments"
    headers = {
        "accept": "*/*",
        "galileo-api-key": api_key,
        "content-type": "application/json",
        "origin": app_url,
        "referer": f"{app_url}/",
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        experiments = response.json()  # API directly returns a list of experiments
        
        logging.info(f"Successfully listed {len(experiments)} experiments")
        return experiments
    except Exception as e:
        logging.error(f"Error listing experiments: {str(e)}")
        return []

def delete_galileo_experiment(api_key: str, project_id: str, experiment_id: str) -> bool:
    """
    Deletes a specific experiment.
    
    Args:
        api_key (str): Your Galileo API key.
        project_id (str): The ID of the project.
        experiment_id (str): The ID of the experiment to delete.
        
    Returns:
        bool: True if deletion was successful, False otherwise.
    """
    # Get URLs
    api_url = get_galileo_api_url()
    app_url = get_galileo_app_url()
    
    url = f"{api_url}/v2/projects/{project_id}/experiments/{experiment_id}"
    headers = {
        "accept": "*/*",
        "galileo-api-key": api_key,
        "content-type": "application/json",
        "origin": app_url,
        "referer": f"{app_url}/",
    }
    
    try:
        response = requests.delete(url, headers=headers)
        response.raise_for_status()
        # 204 status code indicates successful deletion
        return response.status_code == 204
    except Exception as e:
        logging.error(f"Error deleting experiment {experiment_id}: {str(e)}")
        return False

def delete_all_galileo_experiments(api_key: str, project_id: str) -> dict:
    """
    Deletes all experiments for a given project.
    
    Args:
        api_key (str): Your Galileo API key.
        project_id (str): The ID of the project.
        
    Returns:
        dict: A summary of deletion results.
    """
    experiments = list_galileo_experiments(api_key, project_id)
    
    if not experiments:
        return {"success": True, "total": 0, "deleted": 0, "failed": 0, "message": "No experiments found to delete"}
    
    success_count = 0
    fail_count = 0
    
    for experiment in experiments:
        experiment_id = experiment.get('id')
        if experiment_id:
            # Add a small delay to avoid rate limiting
            time.sleep(0.2)
            if delete_galileo_experiment(api_key, project_id, experiment_id):
                success_count += 1
                logging.info(f"Successfully deleted experiment {experiment_id}")
            else:
                fail_count += 1
                logging.error(f"Failed to delete experiment {experiment_id}")
    
    result = {
        "success": fail_count == 0,
        "total": len(experiments),
        "deleted": success_count,
        "failed": fail_count,
        "message": f"Deleted {success_count} of {len(experiments)} experiments"
    }
    
    return result 

