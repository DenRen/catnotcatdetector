from io import BytesIO
from os import listdir
from os.path import join

from numpy import asarray, load, save, uint64, zeros
from PIL import Image
from pyarrow import memory_map, output_stream
from tqdm import tqdm


class ContinuousImageArray:
    def __init__(self, img_arr_path, pos_img_arr) -> None:
        mmap = memory_map(img_arr_path)
        self.byte_buf = memoryview(mmap.read_buffer())
        self.pos_arr = load(pos_img_arr)

    def __len__(self) -> int:
        return len(self.pos_arr)

    def __getitem__(self, idx):
        begin_pos = self.pos_arr[idx - 1] if idx > 0 else 0
        byte_slice = self.byte_buf[begin_pos : self.pos_arr[idx]]
        pil_img = Image.open(BytesIO(byte_slice))
        return asarray(pil_img)

    @staticmethod
    def serialize_dir_imgs(dir_path, img_arr_path, pos_img_arr):
        img_names = sorted(listdir(dir_path))
        num_imgs = len(img_names)

        pos_arr = zeros(num_imgs, dtype=uint64)
        with output_stream(img_arr_path) as stream:
            for idx, img_name in tqdm(enumerate(img_names), total=num_imgs):
                img_path = join(dir_path, img_name)
                with open(img_path, "rb") as img_file:
                    stream.write(img_file.read())

                pos_arr[idx] = stream.tell()

        save(pos_img_arr, pos_arr, allow_pickle=False)
