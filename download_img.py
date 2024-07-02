import json
import os

import os
import requests
import concurrent.futures

import json
import gzip
from tqdm import tqdm
import torch

######## we only handle train splits here first  ############
patton_sports_dir = "./Patton/sports"
pairs_to_parse = torch.load(os.path.join(patton_sports_dir, 'splits.pt'))['train']
raw_data = torch.load('amazon_products_metadata.pt')

data_ls = []
seen_nodes = set()
for pair in pairs_to_parse:
    if pair[0] not in seen_nodes and pair[0] in raw_data and len(raw_data[pair[0]]['image_url']) > 0:
        node = {}
        node['asin'] = pair[0]
        node['image_url'] = raw_data[pair[0]]['image_url']
        data_ls.append(node)
        seen_nodes.add(pair[0])
    if pair[1] not in seen_nodes and pair[1] in raw_data and len(raw_data[pair[1]]['image_url']) > 0:
        node = {}
        node['asin'] = pair[1]
        node['image_url'] = raw_data[pair[1]]['image_url']
        data_ls.append(node)
        seen_nodes.add(pair[1])

path = './amazon-sports-images'
max_threads = 32

print('Start')
print(f'{len(data_ls)}')

completed_count = 0

def download_image(entry):
    global completed_count
    try:
        asin = entry["asin"]
        img_suffix = entry["image_url"][-3:]
        image_url = entry["image_url"].split("_")[0]+img_suffix
        file_path = os.path.join(path, f"{asin}.jpg")
    
        if os.path.exists(file_path):
            completed_count += 1
            return        
        
        response = requests.get(image_url)
        status = response.status_code
        if status == 200:
            with open(file_path, "wb") as file:
                file.write(response.content)
        completed_count += 1
        if completed_count % (len(data_ls) // 10000) == 0:
            with open("./log.txt", "a") as log_file:
                log_file.write(f"Downloaded {completed_count}/{len(data_ls)} - Book ID: {book_id} Status: {status}\n")
    except:
        return 

with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
    futures = [executor.submit(download_image, entry) for entry in data_ls]

    for future in futures:
        future.result()