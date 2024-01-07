from os import listdir
from os.path import join

from torchvision.datasets import VisionDataset

from .datautils import ContinuousImageArray


# Dataset special for cats and dogs
class DatasetSerializedTwoClassImages(VisionDataset):
    def __init__(
        self,
        root_dir: str,
        transform: None,
        num_classes: int = 2,
    ) -> None:
        def two_class_exc(true_cond: bool = False) -> None:
            if not true_cond:
                raise NotImplementedError("DatasetSerializedImages only for 2 class")

        two_class_exc(num_classes == 2)
        super().__init__(root_dir, transform=transform)

        class_names = sorted(listdir(root_dir))
        two_class_exc(len(class_names) == 2)

        self.datasets = []
        for class_name in class_names:
            class_path = join(root_dir, class_name)
            cia = ContinuousImageArray(
                img_arr_path=join(class_path, f"{class_name}_bin"),
                pos_arr_path=join(class_path, f"{class_name}_pos.npy"),
            )
            self.datasets.append(cia)

        self.size_first_class = len(self.datasets[0])

    def __getitem__(self, idx: int):
        ds_idx = int(idx >= self.size_first_class)
        idx -= ds_idx * self.size_first_class

        img = self.datasets[ds_idx][idx].copy()
        if self.transform is not None:
            img = self.transform(img)

        return img, ds_idx

    def __len__(self) -> int:
        return self.size_first_class + len(self.datasets[1])
