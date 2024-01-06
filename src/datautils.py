from io import BytesIO
from os import listdir
from os.path import join

from numpy import asarray, load, ndarray, save, uint64, zeros
from PIL import Image
from pyarrow import memory_map, output_stream
from tqdm import tqdm


class ContinuousImageArray:
    def __init__(self, img_arr_path, pos_arr_path) -> None:
        mmap = memory_map(img_arr_path)
        self.byte_buf = mmap.read_buffer()
        self.pos_arr = load(pos_arr_path)

    def __len__(self) -> int:
        return len(self.pos_arr)

    def get_byte_slice(self, idx: int) -> memoryview:
        begin_pos = self.pos_arr[idx - 1] if idx > 0 else 0
        byte_slice = self.byte_buf[begin_pos : self.pos_arr[idx]]
        return byte_slice

    def __getitem__(self, idx: int) -> ndarray:
        begin_pos = self.pos_arr[idx - 1] if idx > 0 else 0
        byte_slice = self.byte_buf[begin_pos : self.pos_arr[idx]]
        pil_img = Image.open(BytesIO(byte_slice))
        return asarray(pil_img)

    @staticmethod
    def serialize_dir_imgs(
        imgs_dir_path, img_arr_path, pos_arr_path, progress=True
    ) -> None:
        img_names = sorted(listdir(imgs_dir_path))
        num_imgs = len(img_names)

        pos_arr = zeros(num_imgs, dtype=uint64)
        with output_stream(img_arr_path) as stream:
            for idx, img_name in tqdm(
                enumerate(img_names), total=num_imgs, disable=not progress
            ):
                img_path = join(imgs_dir_path, img_name)
                with open(img_path, "rb") as img_file:
                    stream.write(img_file.read())

                pos_arr[idx] = stream.tell()

        save(pos_arr_path, pos_arr, allow_pickle=False)
