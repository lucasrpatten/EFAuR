"""
Gets the metadata from the EFAuR repo.

Author: Lucas Patten
"""

import os
import tarfile
import requests


def get_tarball():
    url = "http://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2"
    with requests.get(url, timeout=15, stream=True) as r:
        r.raise_for_status()
        with open("/tmp/rdf-files.tar.bz2", "wb") as f:
            for chunk in r.iter_content(chunk_size=16384):
                f.write(chunk)


def extract_tarball(extract_dir="/home/lucasrp/compute/tmp/"):
    with tarfile.open('/tmp/rdf-files.tar.bz2', 'r:bz2') as tar:
        for member in tar.getmembers():
            target_path = os.path.join(extract_dir, member.name)
            if not os.path.exists(target_path):
                tar.extract(member, extract_dir, filter=tarfile.tar_filter)

get_tarball()
extract_tarball()
