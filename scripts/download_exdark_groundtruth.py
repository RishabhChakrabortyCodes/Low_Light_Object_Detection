from pathlib import Path
import gdown
import zipfile

DATA_DIR = Path("data/raw")
ARCHIVE_PATH = DATA_DIR / "annotations.zip"
EXTRACT_DIR = DATA_DIR / "annotations"

FILE_ID = "1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i"


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print("Annotations already extracted.")
        return

    if not ARCHIVE_PATH.exists():
        print("Downloading ExDark Annotations...")

        result = gdown.download(
            id=FILE_ID,
            output=str(ARCHIVE_PATH),
            quiet=False,
        )

        if result is None:
            raise RuntimeError("Annotations download failed.")

    print("Extracting Annotations...")

    if not zipfile.is_zipfile(ARCHIVE_PATH):
        raise RuntimeError("Downloaded file is not a valid ZIP archive.")

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        archive.extractall(EXTRACT_DIR)

    print("Annotations successfully downloaded and extracted.")


if __name__ == "__main__":
    download_dataset()