import os
import bs4
import jinja2
import redis
import requests
import gnupg
import zipfile
import io
import pandas as pd
import json
import logging
import yaml
from deepdiff import DeepDiff  # Ensure deepdiff is installed
from jwt import encode as jwt_encode
from datetime import datetime, timedelta, timezone

# Initialize Redis connection
def get_redis_client():
    try:
        redis_client = redis.StrictRedis(
            host=os.environ.get('REDIS_HOST',"localhost"),
            port=int(os.environ.get('REDIS_PORT', 28665)),
            db=0,
            password=os.environ.get('REDIS_PWD', "password"),
            decode_responses=True,
            socket_timeout=30,
            retry_on_timeout=True
        )
        redis_client.ping()
        return redis_client
    except redis.ConnectionError:
        logging.warning("Redis connection failed. Proceeding without caching.")
        return None

# Fetch file mappings from API
def generate_jwt_token(wallet_address: str, secret_key: str, expiration_time: int) -> str:
    """Generate a JWT token for a given wallet address."""
    exp = datetime.now(timezone.utc) + timedelta(seconds=expiration_time)

    payload = {
        'exp': exp,
        'walletAddress': wallet_address  # Send wallet address in the payload
    }
    
    # Encode the JWT
    token = jwt_encode(payload, secret_key, algorithm='HS256')
    return token

def get_file_mappings(wallet_address: str):
    """Fetch file mappings for a given wallet address with JWT authentication."""
    validator_base_api_url = os.environ.get('VALIDATOR_BASE_API_URL')
    secret_key = os.environ.get('JWT_SECRET_KEY')  # Retrieve the secret key from environment variables
    expiration_time = 600  # JWT expiration time in seconds (10 minutes)

    if not validator_base_api_url or not secret_key:
        raise ValueError("VALIDATOR_BASE_API_URL and JWT_SECRET_KEY must be set in environment variables.")

    jwt_token = generate_jwt_token(wallet_address, secret_key, expiration_time)

    endpoint = "/api/userinfo"
    url = f"{validator_base_api_url.rstrip('/')}{endpoint}"

    payload = {"walletAddress": wallet_address}  # Send walletAddress in the body
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {jwt_token}"  # Attach JWT token
    }

    response = requests.post(url, json=payload, headers=headers)  # Make POST request

    if response.status_code == 200:
        return response.json()  # Return JSON response
    else:
        return []  # Return empty list in case of an error
    # return [{"fileId":999, "fileUrl":"https://drive.google.com/uc?export=download&id=1niGj_CmVap_rJotn-BQ3YrF7AxNIJBED"},]
    #         # {"fileId":1607848, "fileUrl":"https://drive.google.com/uc?export=download&id=16xQSjQ1KGNwSJZTA84Ex2v6Z2IGAEDyo"}]

# Download and decrypt file
def download_and_decrypt(file_url, gpg_signature):
    response = requests.get(file_url)
    if response.status_code == 200:
        gpg = gnupg.GPG()
        decrypted_data = gpg.decrypt(response.content, passphrase=gpg_signature)
        if decrypted_data.ok:
            return decrypted_data.data
        else:
            logging.error("Decryption failed.")
            return None
    else:
        logging.error(f"Failed to download file: {response.status_code}")
        return None

yaml_template = """
bookmarks:
{% for dt in bookmarks %}
  - name: "{{ dt.name }}"
    add_date: "{{ dt.add_date }}"
    last_modified: "{{ dt.last_modified | default('') }}"
    personal_toolbar_folder: "{{ dt.personal_toolbar_folder | default('false') }}"
    {% if dt.children %}
    children:
      {% for child in dt.children %}
      - title: "{{ child.title }}"
        url: "{{ child.url }}"
        add_date: "{{ child.add_date }}"
      {% endfor %}
    {% endif %}
{% endfor %}
"""

def parse_bookmarks(html_content):
    soup = bs4.BeautifulSoup(html_content, "html.parser")
    urls = []

    for link in soup.find_all("a"):  # Extract all bookmark links
        urls.append(link["href"])
    
    print(f"unique urls", urls)
    return urls

def convert_to_yaml(bookmarks):
    template = jinja2.Template(yaml_template)
    return template.render(bookmarks=bookmarks)

# Extract files from ZIP data
def extract_files_from_zip(zip_data):
    csv_data_frames = []
    json_data_list = []
    yaml_data_list = []
    
    with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            with zip_ref.open(file_name) as file:
                content = file.read().decode("utf-8")
                if file_name.endswith('.csv'):
                    df = pd.read_csv(io.StringIO(content))
                    csv_data_frames.append(df)
                elif file_name.endswith('.json'):
                    json_data = json.loads(content)
                    json_data_list.append(json_data)
                elif file_name.endswith('.html'):
                    bookmarks = parse_bookmarks(content)
                    # yaml_data = convert_to_yaml(bookmarks)
                    yaml_data_list.extend(bookmarks)
    
    combined_csv_data = pd.concat(csv_data_frames, ignore_index=True) if csv_data_frames else pd.DataFrame()
    return combined_csv_data, json_data_list, yaml_data_list

# Process HTML files in input directory
def process_html_files(input_dir):
    yaml_data_list = []
    local_html_files = [f for f in os.listdir(input_dir) if f.endswith('.html')]
    for html_file in local_html_files:
        file_path = os.path.join(input_dir, html_file)
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            bookmarks = parse_bookmarks(content) # store only array format no yaml
            # yaml_data = convert_to_yaml(bookmarks)
            yaml_data_list.extend(bookmarks)
            print(f"yaml data is ", yaml_data_list)
    return yaml_data_list

# Convert CSV to required format
def convert_csv_to_required_format(df):
    required_columns = ["DateTime", "NavigatedToUrl", "PageTitle"]
    if not all(col in df.columns for col in required_columns):
        # Check for alternative columns
        alternative_columns = ["url", "url_clean", "url_domain", "title", "time", "hour", "day_of_week", "is_weekend", "day_of_month", "week_of_month", "month_of_year", "total_history_days", "seconds_until_next_visit_url", "seconds_until_next_visit_url_clean", "seconds_until_next_visit_domain", "seconds_until_next_visit", "page_transition", "id", "ref_id", "is_local", "client_id", "updated_at"]
        if all(col in df.columns for col in alternative_columns):
            df["DateTime"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            df["NavigatedToUrl"] = df["url"]
            df["PageTitle"] = df["title"]
            df = df[["DateTime", "NavigatedToUrl", "PageTitle"]]
        else:
            logging.error("CSV does not have the required columns or alternative columns.")
            return pd.DataFrame()
    return df

def calculate_unique_url_percentage(curr_yaml_data, combined_yaml_data):
    curr_urls = set()
    combined_urls = set()

    # Extract URLs from curr_yaml_data
    for curr_yaml in curr_yaml_data:
        for bookmark in curr_yaml.get("bookmarks", []):
            for child in bookmark.get("children", []):
                curr_urls.add(child.get("url"))

    # Extract URLs from combined_yaml_data
    for combined_yaml in combined_yaml_data:
        for bookmark in combined_yaml.get("bookmarks", []):
            for child in bookmark.get("children", []):
                combined_urls.add(child.get("url"))

    # Find unique URLs
    unique_urls = curr_urls - combined_urls  # URLs in curr_yaml_data but not in combined_yaml_data
    total_urls = len(curr_urls)
    
    uniqueness_percentage = (len(unique_urls) / total_urls) if total_urls > 0 else 0
    return uniqueness_percentage, len(unique_urls) , total_urls

# Main processing function
def process_files_for_uniqueness(curr_file_id, input_dir, wallet_address):
    gpg_signature = os.environ.get("SIGNATURE")
    redis_client = get_redis_client()
    combined_csv_data = pd.DataFrame()
    combined_json_data = []
    combined_yaml_data = []
    
    logging.info(f"Processing files for wallet address {wallet_address}")
    # Retrieve file mappings from API
    file_mappings = get_file_mappings(wallet_address)

    if redis_client:
        # Check Redis for cached data
        for file_info in file_mappings:
            file_id = file_info.get("fileId")
            if redis_client.exists(file_id):
                stored_csv_data = redis_client.hget(file_id, "browser_history_csv_data")
                stored_json_data = redis_client.hget(file_id, "location_history_json_data")
                stored_html_data = redis_client.hget(file_id, "bookmarks_yaml_data")
                if stored_csv_data:
                    df = pd.read_json(io.StringIO(stored_csv_data))
                    combined_csv_data = pd.concat([combined_csv_data, df], ignore_index=True)
                if stored_json_data:
                    json_data = json.loads(stored_json_data)
                    combined_json_data.extend(json_data)
                if stored_html_data:
                    bookmarks = json.loads(stored_html_data)
                    combined_yaml_data.extend(bookmarks)
                    # yaml_data = convert_to_yaml(bookmarks)
                    # combined_yaml_data.extend(bookmarks)
                    print(f"starting of combined yaml data",combined_yaml_data, "with bookmarks url",  bookmarks)

                if not stored_csv_data and not stored_json_data and not stored_html_data:
                    file_url = file_info.get("fileUrl")
                    if not file_url:
                        logging.warning(f"Skipping invalid fileUrl for fileId {file_id}")
                        continue

                    decrypted_data = download_and_decrypt(file_url, gpg_signature)

                    if not decrypted_data:  # Skip if download failed
                        logging.warning(f"Skipping file {file_url} due to download error.")
                        continue  # Move to the next file
                    if decrypted_data:
                        df, json_data_list, yaml_data_list = extract_files_from_zip(decrypted_data) # returns pd, json_list, yaml_list
                        if df is not None:
                            combined_csv_data = pd.concat([combined_csv_data, df], ignore_index=True) 
                        if json_data_list:
                            combined_json_data.extend(json_data_list)
                        if yaml_data_list:
                            combined_yaml_data.extend(yaml_data_list)
            else:
                file_url = file_info.get("fileUrl")
                if not file_url:
                    logging.warning(f"Skipping invalid fileUrl for fileId {file_id}")
                    continue
                decrypted_data = download_and_decrypt(file_url, gpg_signature)
                if not decrypted_data:  # Skip if download failed
                    logging.warning(f"Skipping file {file_url} due to download error.")
                    continue  # Move to the next file
                if decrypted_data:
                    df, json_data_list, yaml_data_list = extract_files_from_zip(decrypted_data)
                    if df is not None:
                        combined_csv_data = pd.concat([combined_csv_data, df], ignore_index=True) 
                    if json_data_list:
                        combined_json_data.extend(json_data_list)
                    if yaml_data_list:
                        combined_yaml_data.extend(yaml_data_list)
    else:
        # Download, decrypt, and extract files
        for file_info in file_mappings:
            file_url = file_info.get("fileUrl")
            if not file_url:
                logging.warning(f"Skipping invalid fileUrl for fileId {file_info.get('fileId')}")
                continue
            decrypted_data = download_and_decrypt(file_url, gpg_signature)
            if not decrypted_data:  # Skip if download failed
                logging.warning(f"Skipping file {file_url} due to download error.")
                continue  # Move to the next file
            if decrypted_data:
                df, json_data_list, yaml_data_list = extract_files_from_zip(decrypted_data)
                if df is not None:
                    combined_csv_data = pd.concat([combined_csv_data, df], ignore_index=True) 
                if json_data_list:
                    combined_json_data.extend(json_data_list)
                if yaml_data_list:
                    combined_yaml_data.extend(yaml_data_list)

    # Process current input directory CSVs
    curr_file_csv_data = pd.DataFrame()
    local_csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    for csv_file in local_csv_files:
        file_path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(file_path)
        df = convert_csv_to_required_format(df)
        curr_file_csv_data = pd.concat([curr_file_csv_data, df], ignore_index=True)

    # Process current input directory JSONs
    curr_file_json_data = []
    local_json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    for json_file in local_json_files:
        file_path = os.path.join(input_dir, json_file)
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            
            # If the JSON data is a list, wrap it in an object with "semanticSegments" as the key
            if isinstance(json_data, list):
                json_data = {"semanticSegments": json_data}
            
            curr_file_json_data.append(json_data)
        
        # Overwrite the file with the new format if it was a list
        with open(file_path, 'w') as file:
            json.dump(json_data, file, indent=4)

    print("JSON files processed and formatted successfully.")
    
    # Process current input directory HTMLs
    curr_yaml_data = process_html_files(input_dir)

    # Ensure both have the same datetime format
    if not curr_file_csv_data.empty:
        curr_file_csv_data["DateTime"] = pd.to_datetime(curr_file_csv_data["DateTime"], utc=True)
    if not combined_csv_data.empty:
        combined_csv_data["DateTime"] = pd.to_datetime(combined_csv_data["DateTime"], utc=True)

    # Find unique entries in curr_file_csv_data that are not in combined_csv_data and curr_file_csv_data exists
    unique_curr_csv_data = curr_file_csv_data
    if not combined_csv_data.empty:
        common_columns = curr_file_csv_data.columns.intersection(combined_csv_data.columns)
        if not common_columns.empty:
            unique_curr_csv_data = curr_file_csv_data.merge(combined_csv_data[common_columns], how="left", indicator=True).query('_merge == "left_only"').drop(columns=["_merge"])
        else:
            logging.warning("No common columns found between current and combined CSV data. Skipping merge operation.")

    # Identify unique JSON entries
    unique_curr_json_data = []
    for curr_json in curr_file_json_data:
        is_unique = True
        for combined_json in combined_json_data:
            if not DeepDiff(curr_json, combined_json, ignore_order=True):
                is_unique = False
                break
        if is_unique:
            unique_curr_json_data.append(curr_json)

    # for curr_yaml in curr_yaml_data:
    #     is_unique = True
    #     for combined_yaml in combined_yaml_data:
    #         if not DeepDiff(curr_yaml, combined_yaml, ignore_order=True):
    #             is_unique = False
    #             break
    #     if is_unique:
    #         unique_curr_yaml_data.append(curr_yaml)

    # Calculate uniqueness scores
    total_csv_entries = curr_file_csv_data.drop_duplicates().shape[0]
    unique_csv_entries = unique_curr_csv_data.drop_duplicates().shape[0]
    csv_uniqueness_score = unique_csv_entries / total_csv_entries if total_csv_entries > 0 else 0.0

    total_json_entries = len({json.dumps(entry, sort_keys=True) for entry in curr_file_json_data})
    unique_json_entries = len({json.dumps(entry, sort_keys=True) for entry in unique_curr_json_data})
    json_uniqueness_score = unique_json_entries / total_json_entries if total_json_entries > 0 else 0.0

     # Identify unique YAML entries
    # unique_curr_yaml_data = []
    unique_curr_yaml_data = list(set(curr_yaml_data) - set(combined_yaml_data))
    unique_yaml_entries = len(unique_curr_yaml_data)
    total_yaml_entries = len(set(curr_yaml_data))
    yaml_uniqueness_score = unique_yaml_entries / total_yaml_entries if total_yaml_entries > 0 else 0.0
    print(f"Unique yaml list data",unique_curr_yaml_data, "combined yaml list data", combined_yaml_data)
    # yaml_uniqueness_score, unique_yaml_entries, total_yaml_entries = calculate_unique_url_percentage(curr_yaml_data, combined_yaml_data)
    # unique_yaml_entries / total_yaml_entries if total_yaml_entries > 0 else 0.0

    # Determine final uniqueness score
    final_uniqueness_score = 0.0
    if csv_uniqueness_score != 0.0 and json_uniqueness_score != 0.0 and yaml_uniqueness_score != 0.0:
        # Normalize based on data volume
        final_uniqueness_score = (
            (csv_uniqueness_score * total_csv_entries) + 
            (json_uniqueness_score * total_json_entries) + 
            (yaml_uniqueness_score * total_yaml_entries)
        ) / (total_csv_entries + total_json_entries + total_yaml_entries)
    elif csv_uniqueness_score != 0.0:
        final_uniqueness_score = csv_uniqueness_score
    elif json_uniqueness_score != 0.0:
        final_uniqueness_score = json_uniqueness_score
    elif yaml_uniqueness_score != 0.0:
        final_uniqueness_score = yaml_uniqueness_score

    # Round final uniqueness score to 3 decimal places
    if final_uniqueness_score != 0.0:
        final_uniqueness_score = round(final_uniqueness_score, 3)

    # Cache current file data in Redis
    if redis_client:
        redis_client.hset(curr_file_id, mapping={
            "browser_history_csv_data": curr_file_csv_data.to_json(),
            "location_history_json_data": json.dumps(curr_file_json_data),
            "bookmarks_yaml_data": json.dumps(curr_yaml_data)
        })
    logging.info(f"Current file data stored in Redis under key {curr_file_id}")
    logging.info(f"Unique CSV data: {unique_curr_csv_data}")
    logging.info(f"Unique JSON data: {unique_curr_json_data}")
    logging.info(f"Unique YAML data: {unique_curr_yaml_data}")
    logging.info(f"Final Uniqueness Score: {final_uniqueness_score}")

    data_types_provided = []
    if not curr_file_csv_data.empty:
        data_types_provided.append("csv")
    if curr_file_json_data:
        data_types_provided.append("json")
    if curr_yaml_data:
        data_types_provided.append("yaml")

    # Return unique data and scores
    return {
        "unique_csv_data": unique_curr_csv_data,
        "unique_json_data": unique_curr_json_data,
        "unique_yaml_data": unique_curr_yaml_data,
        "unique_yaml_entries": unique_yaml_entries,   # count of unique urls
        "curr_csv_data": curr_file_csv_data,
        "curr_json_data": curr_file_json_data,
        "curr_yaml_data": curr_yaml_data,
        "uniqueness_score": final_uniqueness_score,
        "csv_uniqueness_score": csv_uniqueness_score,
        "json_uniqueness_score": json_uniqueness_score,
        "yaml_uniqueness_score": yaml_uniqueness_score,
        "data_types_provided": data_types_provided
    }