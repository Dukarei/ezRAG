import concurrent.futures
import requests
from bs4 import BeautifulSoup
import pathlib

def file_to_list(filename):
    """Reads a file and returns its contents as a list of lines."""
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
            return lines
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return []

def process_html(html):
    """Extracts relevant text from HTML content."""
    soup = BeautifulSoup(html, "lxml")
    info_div = soup.get_text()
    start_index = info_div.find("article")
    end_index = info_div.find("Was this page helpful")
    if start_index != -1 and end_index != -1:
        return info_div[start_index:end_index]
    else:
        return ""

def download_url(url, file_path):
    """Downloads content from a URL and saves it to a file."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        html = response.content
        text = process_html(html)
        with open(file_path, 'w') as file:
            file.write(text)
        print(f"Downloaded {url} successfully")
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")

def parallel_download(urls, num_workers, output_dir):
    """Downloads content from multiple URLs in parallel."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_url, url, output_dir / f"{url.split('/')[-1]}.txt"): url for url in urls}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error downloading {url}: {e}")

# Example usage
urls = file_to_list('winURLs.txt')
num_workers = 25
output_dir = 'downloaded_files'
parallel_download(urls, num_workers, output_dir)