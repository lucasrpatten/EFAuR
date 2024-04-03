"""
Gets the data for EFAuR from the Project Gutenberg Corpus

Author: Lucas Patten
"""

import csv
import logging
import re
import subprocess
import os

from urllib.parse import quote

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def mirror_data(data_dir: str = "/home/lucasrp/compute/tmp"):
    """Mirrors the data from the Project Gutenberg Corpus

    Args:
        data_dir (str, optional): Base path of data directory. Defaults to "/home/lucasrp/compute/tmp".
    """
    mirror_dir = os.path.join(data_dir, ".mirror")
    if not os.path.exists(mirror_dir):
        os.makedirs(mirror_dir)

    vstring = "v"
    sp_args = [
        "rsync",
        f"-am{vstring}",
        "--include",
        "*/",
        "--include",
        "[p123456789][g0123456789]*[.-][t0][x.]t[x.]*[t8]",
        "--exclude",
        "*",
        "aleph.gutenberg.org::gutenberg",
        mirror_dir,
    ]

    subprocess.call(sp_args)


def clean_book(book: str, book_id: str):

    start_pattern = r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG.*\*\*\*"
    end_pattern = r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG"

    split_book = re.split(start_pattern, book, flags=re.IGNORECASE)
    if len(split_book) < 2:
        logging.warning("Failed to parse book %s", book_id)
        return None
    book = split_book[1]
    split_book = re.split(end_pattern, book, flags=re.IGNORECASE)
    if len(split_book) < 2:
        logging.warning("Failed to parse book %s", book_id)
        return None
    book = split_book[0]

    return book


def by_author(data_dir: str = "/home/lucasrp/compute/tmp/"):
    mirror_dir = os.path.join(data_dir, ".mirror", "cache", "epub")
    author_dir = os.path.join(data_dir, "by_author")
    metadata_dir = os.path.join(data_dir, "metadata")
    if not os.path.exists(author_dir):
        os.makedirs(author_dir)
    with open(os.path.join(metadata_dir, "metadata.csv"), "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter=",")
        next(csv_reader) # Skip header
        authors = set()
        books_processed = 0
        for book_id, author in csv_reader:

            book_path = os.path.join(mirror_dir, book_id, f"pg{book_id}.txt.utf8")

            if not os.path.exists(book_path):
                book_path = "/".join([book_id[i] for i in range(len(book_id) - 1)])
                book_path = os.path.join(mirror_dir, book_path, book_id, f"{book_id}-0.txt")
                if not os.path.exists(book_path):
                    logging.info("Skipping %s", book_id)
                    continue

            logging.debug("Processing %s", book_id)
            with open(book_path, "r", encoding="latin-1") as f:
                book = f.read()
            cleaned_book = clean_book(book, book_id)
            if cleaned_book is None:
                continue
            author = quote(author)
            books_processed += 1
            if author not in authors:
                authors.add(author)
                with open(
                    os.path.join(author_dir, f"{author}.txt"), "w", encoding="utf-8"
                ) as f:
                    f.write(cleaned_book + "\n")
            else:
                with open(
                    os.path.join(author_dir, f"{author}.txt"), "a", encoding="utf-8"
                ) as f:
                    f.write(cleaned_book + "\n")
        logging.info("Successfully processed %d books", books_processed)


by_author()
