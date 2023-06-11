from config import car_config as config
import csv
import requests
import pandas as pd
import os
import concurrent.futures


csv_file = config.USED_CARS_DATSET_LINKS

df = pd.read_csv(csv_file)
output_folder = config.RAW_DATA_DIR


def save_image_from_url(url, output_folder):
    # print(url.main_picture_url)
    image = requests.get(url.main_picture_url, timeout=10)
    # print(image, url.main_picture_url)
    output_path = os.path.join(output_folder, url.main_picture_url.split("/")[-1])
    print(output_path)
    with open(output_path, "wb") as f:
        f.write(image.content)


def load(df, output_folder):    
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=5
    ) as executor:
        future_to_url = {
            executor.submit(save_image_from_url, url, output_folder): url
            for _, url in df.iterrows()
        }
        for future in concurrent.futures.as_completed(
            future_to_url
        ):
            url = future_to_url[future]
            try:
                future.result()
            except Exception as exc:
                print(
                    "%r generated an exception: %s" % (url, exc)
                )


load(df, output_folder)