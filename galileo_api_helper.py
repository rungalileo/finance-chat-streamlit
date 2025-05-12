import requests

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
    url = f"https://app.galileo.ai/api/galileo/v2/projects?starting_token={starting_token}&limit={limit}&actions=delete"
    headers = {
        "accept": "*/*",
        "galileo-api-key": api_key,
        "content-type": "application/json",
        "origin": "https://app.galileo.ai",
        "referer": "https://app.galileo.ai/",
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
    url = f"https://app.galileo.ai/api/galileo/v2/projects/{project_id}/log_streams"
    headers = {
        "accept": "*/*",
        "galileo-api-key": api_key,
        "content-type": "application/json",
        "origin": "https://app.galileo.ai",
        "referer": "https://app.galileo.ai/",
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