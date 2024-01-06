from os import listdir, makedirs, remove
from os.path import exists, join
from shutil import rmtree, unpack_archive

from wget import download

from datautils import ContinuousImageArray


def download_row_imgs(row_data_dir):
    link = "https://www.dropbox.com/s/gqdo90vhli893e0/data.zip?dl=1"
    zip_path = download(url=link, out=".data.zip")
    unpack_archive(zip_path, row_data_dir)
    remove(zip_path)


def serialize_row_data(row_data_dir, ser_data_dir, check_serialization: bool = True):
    # 0: test, train, val
    # 1: cat, dog
    # 2: *.jpg

    if exists(ser_data_dir):
        msg = f"Directory ser_data_dir={ser_data_dir!r} already exists!"
        raise RuntimeError(msg)

    for root_dir in listdir(row_data_dir):
        root_dir_path = join(row_data_dir, root_dir)
        for class_name in listdir(root_dir_path):
            dest_dir_path = join(ser_data_dir, root_dir, class_name)
            makedirs(dest_dir_path)

            ser_path_base = join(dest_dir_path, class_name)
            img_arr_path = f"{ser_path_base}_bin"
            pos_arr_path = f"{ser_path_base}_pos.npy"
            imgs_dir_path = join(root_dir_path, class_name)

            ContinuousImageArray.serialize_dir_imgs(
                imgs_dir_path, img_arr_path, pos_arr_path
            )

            if not check_serialization:
                continue

            cia = ContinuousImageArray(img_arr_path, pos_arr_path)
            img_names = sorted(listdir(imgs_dir_path))
            if len(img_names) != len(cia):
                msg = f"\
                    \rIncorrect number of images:\n\
                    \r    len(img_names) != len(cia) => {len(img_names)} != {len(cia)}\n\
                    \r    img_arr_path = {img_arr_path!r}\n\
                    \r    pos_arr_path = {pos_arr_path!r}"
                raise RuntimeError(msg)

            for idx, img_name in enumerate(img_names):
                img_path = join(imgs_dir_path, img_name)
                with open(img_path, "rb") as img_file:
                    img_bytes = img_file.read()
                    ser_bytes = cia.get_byte_slice(idx)

                    if img_bytes != ser_bytes:
                        raise RuntimeError(f"{img_path} incorrect serialization")


def init_serialized_data():
    row_data_dir = "row_data"
    serialized_data_dir = "serialized_data"
    check_serialization = True

    download_row_imgs(row_data_dir)
    serialize_row_data(row_data_dir, serialized_data_dir, check_serialization)
    rmtree(row_data_dir)


if __name__ == "__main__":
    try:
        init_serialized_data()
    except Exception as exc:
        print(f"Error: {exc}")
