from pathlib import Path
from shutil import unpack_archive

from wget import download

from model import train


def download_data() -> Path:
    data = "data"
    data_zip = f"{data}.zip"

    data, data_zip = Path(data), Path(data_zip)

    if not data.exists() and not data_zip.exists():
        download(f"https://www.dropbox.com/s/gqdo90vhli893e0/{data_zip}?dl=1")

    if not data.exists():
        unpack_archive(data_zip, "data")

    return data


if __name__ == "__main__":
    download_data()
    train()
