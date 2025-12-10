"""
Need to first download urls in parquet format using HF CLI
"""

from img2dataset import download
import shutil
import os

resume = True

if __name__ == "__main__":
    output_dir = os.path.abspath("/mnt/proj3/open-35-39/datasets/coyo300m/")

    if not resume:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    else:
        pass


    download(
        processes_count=96,
        thread_count=256,
        input_format="parquet",
        url_list="/mnt/proj3/open-35-39/hf_caches/hub/datasets--kakaobrain--coyo-labeled-300m/snapshots/8d62a7d805261fc2ffd233a4f31e33049d87eec4/data/",
        url_col="url",
        image_size=512,
        retries=1,
        min_image_size=200,
        max_aspect_ratio=3,
        resize_only_if_bigger=True,
        resize_mode="keep_ratio",
        skip_reencode=True,
        save_additional_columns=["labels", "label_probs"],
        enable_wandb=False,
        output_folder=output_dir,
        output_format="webdataset",
        distributor="multiprocessing",
    )
