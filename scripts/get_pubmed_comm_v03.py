import pandas as pd
import os
import gzip
import requests
from io import BytesIO
import tarfile
from shutil import copyfileobj
from zipfile import ZipFile
from joblib import Parallel, delayed
from tqdm.auto import tqdm as tqdm_auto

BASEURL = 'https://ftp.ncbi.nlm.nih.gov/pub/pmc/'
fdf = pd.read_csv(f'{BASEURL}/oa_comm_use_file_list.csv')
destination_directory = "datamount/pubmed/all_images"

def process_row(row):
    try:
        url = f'{BASEURL}/{row.File}'
        # Download the .gz file
        response = requests.get(url)
        response.raise_for_status()
        # Extract the contents of the .gz file
        with gzip.open(BytesIO(response.content), "rb") as gz_file:
            with tarfile.open(fileobj=gz_file) as tar_file:
                for file_info in tar_file.getmembers():
                    if file_info.name.endswith(".jpg"):
                        if '.g' not in file_info.name:
                            continue
                        # Save the .jpg file to the destination directory
                        with tar_file.extractfile(file_info) as input_file:
                            source_name = (row.File.split('.tar')[0]).replace('/', '_')
                            out_name_base = os.path.basename(file_info.name)
                            out_name_base = source_name + '__' + out_name_base
                            out_name= os.path.join(destination_directory, out_name_base)
                            with open(out_name, "wb") as output_file:
                                copyfileobj(input_file, output_file)
    except:
        pass

# Parallelize the loop using joblib
Parallel(n_jobs=-1)(delayed(process_row)(row) for _, row in tqdm_auto(fdf.iterrows(), total=len(fdf)))

