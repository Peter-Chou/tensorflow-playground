# -*- coding: utf-8 -*-
import zipfile
from os.path import isdir, isfile
from urllib.request import urlretrieve

from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def download(url, folder_path, filename, pbar_desc="Dataset"):
    if not isfile(filename):
        with DLProgress(
                unit='B', unit_scale=True, miniters=1, desc=pbar_desc) as pbar:
            urlretrieve(
                url,
                filename,
                pbar.hook)

    if not isdir(folder_path):
        with zipfile.ZipFile(filename) as zip_ref:
            zip_ref.extractall(folder_path)

    with open("data/text8") as f:
        text = f.read()

    return text
