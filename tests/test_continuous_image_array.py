import tempfile
from os.path import join

import numpy as np
import pytest
from numpy import asarray, uint8
from numpy.random import randint
from PIL import Image

from src.datautils import ContinuousImageArray


def test_serializ_deserializ_cont_img_arr():
    num_imgs = 100
    seed = 4

    np.random.seed(seed)

    save_data = []
    img_ctr = 0

    def add_rand_img(shape):
        nonlocal img_ctr
        img_path = join(imgs_dir, f"{img_ctr:06}.jpg")
        img_ctr += 1

        orig_img = randint(0, 255, shape, dtype=uint8)
        pil_img = Image.fromarray(orig_img)
        pil_img.save(img_path)

        decomp_img = asarray(Image.open(img_path))
        save_data.append([img_path, decomp_img])

    def add_n_rand_img(num: int, *args):
        for _ in range(num):
            add_rand_img(*args)

    with tempfile.TemporaryDirectory() as imgs_dir:
        [add_rand_img(randint(1, 600, size=2)) for _ in range(num_imgs)]
        add_n_rand_img(100, (1, 1))
        add_n_rand_img(5, (2, 2))
        add_n_rand_img(100, (1, 100))
        add_n_rand_img(15, (100, 1))
        add_rand_img((1009, 36809))

        ser_path_base = join(imgs_dir, "ser_res")
        img_arr_path = f"{ser_path_base}_bin"
        pos_arr_path = f"{ser_path_base}_pos.npy"
        ContinuousImageArray.serialize_dir_imgs(imgs_dir, img_arr_path, pos_arr_path)

        cia = ContinuousImageArray(img_arr_path, pos_arr_path)
        assert len(cia) == len(save_data)

        for idx, (img_path, decomp_img) in enumerate(save_data):
            cia_img = cia[idx]
            assert cia_img.shape == decomp_img.shape
            if not np.array_equal(cia_img, decomp_img):
                for cia_val, val in zip(
                    cia_img.flatten(), decomp_img.flatten(), strict=True
                ):
                    print(f"{cia_val == val:6}: {cia_val:4} {val:4}")

                pytest.exit(f"Failed to serialize img {img_path!r}")
