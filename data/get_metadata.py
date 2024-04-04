"""
Gets the metadata from the EFAuR repo.

Author: Lucas Patten
"""

import csv
import logging
import os
import tarfile
from xml.etree import ElementTree
import requests

XML_NAMESPACES = dict(
    cc="http://web.resource.org/cc/",
    pgterms="http://www.gutenberg.org/2009/pgterms/",
    dcterms="http://purl.org/dc/terms/",
    rdfs="http://www.w3.org/2000/01/rdf-schema#",
    rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    dcam="http://purl.org/dc/dcam/",
)

FIELDS = (
    "id",
    "author",
)


def get_tarball():
    """Downloads the RDF Metadata tarball to /tmp/rdf-files.tar.bz2"""
    logging.info("Downloading EFAuR metadata tarball...")
    url = "http://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2"
    with requests.get(url, timeout=15, stream=True) as r:
        r.raise_for_status()
        with open("/tmp/rdf-files.tar.bz2", "wb") as f:
            for chunk in r.iter_content(chunk_size=16384):
                f.write(chunk)


def parse_rdf(path: str, metadata_dir: str):
    """Parses the RDF Metadata and gets the valid books and writes them to metadata.csv

    Args:
        path (str): Path of rdf file inside tarball
        metadata_dir (str): Base path of metadata directory
    """
    metadata = dict.fromkeys(FIELDS)
    tree = ElementTree.parse(path)
    root = tree.getroot()
    ebook = root.find("{%(pgterms)s}ebook" % XML_NAMESPACES)
    print(ebook)
    if ebook is None:
        return

    # ensure it's public domain
    rights = ebook.find("{%(dcterms)s}rights" % XML_NAMESPACES)
    if rights is None:
        return
    assert isinstance(rights.text, str)
    print(rights.text)

    if "public domain" not in rights.text.lower():
        return

    # id
    about = ebook.get("{%(rdf)s}about" % XML_NAMESPACES)
    assert isinstance(about, str)
    id_number = int(os.path.basename(about))
    if id_number == 17:  # skip the Book of Mormon (I'm doing projects with it later)
        return
    metadata["id"] = id_number

    # author
    creator = ebook.find(".//{%(dcterms)s}creator" % XML_NAMESPACES)
    if creator is None:  # Not interested in books with unknown authors
        return
    author = creator.find(".//{%(pgterms)s}name" % XML_NAMESPACES)
    if author is None:  # Not interested in books with unknown authors
        return
    assert isinstance(author.text, str)
    if any(
        a in author.text.lower() for a in ("anonymous", "various", "unknown")
    ):  # Not interested in anonymous authors
        return
    metadata["author"] = author.text

    # type
    booktype = ebook.find(".//{%(dcterms)s}type//{%(rdf)s}value" % XML_NAMESPACES)
    if booktype is None:  # Not interested in books with unknown types
        return
    assert isinstance(booktype.text, str)
    if booktype.text != "Text":  # Not interested in other types
        return

    # languages
    language = ebook.find(".//{%(dcterms)s}language//{%(rdf)s}value" % XML_NAMESPACES)
    if language is None:  # Not interested in books with unknown languages
        return
    assert isinstance(language.text, str)
    if language.text.lower() not in (
        "english",
        "en",
    ):  # Not interested in other languages
        return

    fields = list(metadata.keys())
    with open(os.path.join(metadata_dir, "metadata.csv"), "a", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerow(metadata)


def get_metadata(data_dir: str):
    """Extracts metadata tarball and updates metadata information in metadata.csv

    Args:
        data_dir (str): Base path of data directory.
    """
    metadata_dir = os.path.join(data_dir, "metadata")

    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)
    if not os.path.exists(os.path.join(metadata_dir, "metadata.csv")):
        with open(
            os.path.join(metadata_dir, "metadata.csv"), "w", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()

    logging.info("Extracting EFAuR metadata tarball...")
    with tarfile.open("/tmp/rdf-files.tar.bz2", "r:bz2") as tar:
        for member in tar.getmembers():
            target_path = os.path.join(metadata_dir, member.name)
            # If the file is already extracted, skip, otherwise extract and parse rdf
            if not os.path.exists(target_path):
                tar.extract(member, metadata_dir, filter=tarfile.tar_filter)
                logging.debug("Extracted %s", member.name)
                if member.isfile() and member.name.endswith(".rdf"):
                    parse_rdf(target_path, metadata_dir)
            else:
                logging.debug("Skipping %s, already exists", member.name)
