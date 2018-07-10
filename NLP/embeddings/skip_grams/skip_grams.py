# -*- coding: utf-8 -*-

"""implementation of embeddings using skip-grams"""

from utils import download

DATASET_FOLDER_PATH = 'data'
DATASET_FILENAME = 'text8.zip'
DATASET_NAME = 'Text8 Dataset'
URL = "http://mattmahoney.net/dc/text8.zip"


def main():
    text = download(URL, DATASET_FOLDER_PATH, DATASET_FILENAME, DATASET_NAME)
    print(text[:100])


if __name__ == '__main__':
    main()
