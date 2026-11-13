from pathlib import Path
import gdown
import zipfile

DATA_DIR = Path("data/raw")
ARCHIVE_PATH = DATA_DIR / "exdark.zip"
EXTRACT_DIR = DATA_DIR / "images"

FILE_ID = "1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx"


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print("Dataset already extracted.")
        return

    if not ARCHIVE_PATH.exists():
        print("Downloading ExDark dataset...")

        result = gdown.download(
            id=FILE_ID,
            output=str(ARCHIVE_PATH),
            quiet=False,
        )

        if result is None:
            raise RuntimeError("Dataset download failed.")

    print("Extracting dataset...")

    if not zipfile.is_zipfile(ARCHIVE_PATH):
        raise RuntimeError("Downloaded file is not a valid ZIP archive.")

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        archive.extractall(EXTRACT_DIR)

    print("Dataset successfully downloaded and extracted.")


if __name__ == "__main__":
    download_dataset()