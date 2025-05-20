import os

import numpy as np
import SimpleITK as sitk
import torch
import torchio as tio
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset
from torchio.data.io import sitk_to_nib


class Dataset_Union_ALL(Dataset):

    def __init__(
        self,
        paths,
        mode="train",
        data_type="Tr",
        image_size=128,
        transform=None,
        threshold=500,
        split_num=1,
        split_idx=0,
        pcc=False,
        get_all_meta_info=False,
    ):
        self.paths = paths
        self.data_type = data_type
        self.split_num = split_num
        self.split_idx = split_idx

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
        self.mode = mode
        self.pcc = pcc
        self.get_all_meta_info = get_all_meta_info

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):

        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        sitk_image_arr, _ = sitk_to_nib(sitk_image)
        sitk_label_arr, _ = sitk_to_nib(sitk_label)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=sitk_image_arr),
            label=tio.LabelMap(tensor=sitk_label_arr),
        )

        if "/ct_" in self.image_paths[index]:
            subject = tio.Clamp(-1000, 1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except BaseException:
                print(self.image_paths[index])

        if self.pcc:
            print("using pcc setting")
            # crop from random click point
            random_index = torch.argwhere(subject.label.data == 1)
            if len(random_index) >= 1:
                random_index = random_index[np.random.randint(0, len(random_index))]
                # print(random_index)
                crop_mask = torch.zeros_like(subject.label.data)
                # print(crop_mask.shape)
                crop_mask[random_index[0]][random_index[1]][random_index[2]][random_index[3]] = 1
                subject.add_image(
                    tio.LabelMap(tensor=crop_mask, affine=subject.label.affine),
                    image_name="crop_mask",
                )
                subject = tio.CropOrPad(
                    mask_name="crop_mask",
                    target_shape=(self.image_size, self.image_size, self.image_size),
                )(subject)

        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))
        if self.mode == "train" and self.data_type == "Tr":
            return {
                "image": subject.image.data.clone().detach(),
                "label": subject.label.data.clone().detach(),
            }
        elif self.get_all_meta_info:
            meta_info = {
                "image_path": self.image_paths[index],
                "origin": sitk_label.GetOrigin(),
                "direction": sitk_label.GetDirection(),
                "spacing": sitk_label.GetSpacing(),
            }
            return {
                "image": subject.image.data.clone().detach(),
                "label": subject.label.data.clone().detach(),
                "meta_info": meta_info
            }
        else:
            return {
                "image": subject.image.data.clone().detach(),
                "label": subject.label.data.clone().detach(),
                "path": self.image_paths[index],
            }

    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        # if ${path}/labelsTr exists, search all .nii.gz
        for path in paths:
            d = os.path.join(path, f"labels{self.data_type}")
            if os.path.exists(d):
                for name in os.listdir(d):
                    base = os.path.basename(name).split(".nii.gz")[0]
                    label_path = os.path.join(path, f"labels{self.data_type}", f"{base}.nii.gz")
                    self.image_paths.append(label_path.replace("labels", "images"))
                    self.label_paths.append(label_path)


class Dataset_Union_ALL_Val(Dataset_Union_ALL):

    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        # if ${path}/labelsTr exists, search all .nii.gz
        for path in paths:
            for dt in ["Tr", "Val", "Ts"]:
                d = os.path.join(path, f"labels{dt}")
                if os.path.exists(d):
                    for name in os.listdir(d):
                        base = os.path.basename(name).split(".nii.gz")[0]
                        label_path = os.path.join(path, f"labels{dt}", f"{base}.nii.gz")
                        self.image_paths.append(label_path.replace("labels", "images"))
                        self.label_paths.append(label_path)
        self.image_paths = self.image_paths[self.split_idx::self.split_num]
        self.label_paths = self.label_paths[self.split_idx::self.split_num]


class Dataset_Union_ALL_Infer(Dataset):
    """Only for inference, no label is returned from __getitem__."""

    def __init__(
        self,
        paths,
        data_type="infer",
        image_size=128,
        transform=None,
        split_num=1,
        split_idx=0,
        pcc=False,
        get_all_meta_info=False,
    ):
        self.paths = paths
        self.data_type = data_type
        self.split_num = split_num
        self.split_idx = split_idx

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.pcc = pcc
        self.get_all_meta_info = get_all_meta_info

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        sitk_image = sitk.ReadImage(self.image_paths[index])

        sitk_image_arr, _ = sitk_to_nib(sitk_image)

        subject = tio.Subject(image=tio.ScalarImage(tensor=sitk_image_arr), )

        if "/ct_" in self.image_paths[index]:
            subject = tio.Clamp(-1000, 1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except BaseException:
                print("Could not transform", self.image_paths[index])

        if self.pcc:
            print("using pcc setting")
            # crop from random click point
            random_index = torch.argwhere(subject.label.data == 1)
            if len(random_index) >= 1:
                random_index = random_index[np.random.randint(0, len(random_index))]
                crop_mask = torch.zeros_like(subject.label.data)
                crop_mask[random_index[0]][random_index[1]][random_index[2]][random_index[3]] = 1
                subject.add_image(
                    tio.LabelMap(tensor=crop_mask, affine=subject.label.affine),
                    image_name="crop_mask",
                )
                subject = tio.CropOrPad(
                    mask_name="crop_mask",
                    target_shape=(self.image_size, self.image_size, self.image_size),
                )(subject)

        elif self.get_all_meta_info:
            meta_info = {
                "image_path": self.image_paths[index],
                "direction": sitk_image.GetDirection(),
                "origin": sitk_image.GetOrigin(),
                "spacing": sitk_image.GetSpacing(),
            }
            return subject.image.data.clone().detach(), meta_info
        else:
            return subject.image.data.clone().detach(), self.image_paths[index]

    def _set_file_paths(self, paths):
        self.image_paths = []

        # if ${path}/infer exists, search all .nii.gz
        for path in paths:
            d = os.path.join(path, f"{self.data_type}")
            if os.path.exists(d):
                for name in os.listdir(d):
                    base = os.path.basename(name).split(".nii.gz")[0]
                    image_path = os.path.join(path, f"{self.data_type}", f"{base}.nii.gz")
                    self.image_paths.append(image_path)

        self.image_paths = self.image_paths[self.split_idx::self.split_num]


class Union_Dataloader(tio.SubjectsLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Test_Single(Dataset):

    def __init__(self, paths, image_size=128, transform=None, threshold=500):
        self.paths = paths

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):

        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        subject = tio.Subject(
            image=tio.ScalarImage.from_sitk(sitk_image),
            label=tio.LabelMap.from_sitk(sitk_label),
        )

        if "/ct_" in self.image_paths[index]:
            subject = tio.Clamp(-1000, 1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except BaseException:
                print(self.image_paths[index])

        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))

        return (
            subject.image.data.clone().detach(),
            subject.label.data.clone().detach(),
            self.image_paths[index],
        )

    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        self.image_paths.append(paths)
        self.label_paths.append(paths.replace("images", "labels"))


if __name__ == "__main__":
    test_dataset = Dataset_Union_ALL_Infer(
        paths=[
            './data/inference/heart/hearts/',
        ],
        data_type='infer',
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(target_shape=(128, 128, 128)),
        ]),
        pcc=False,
        get_all_meta_info=True,
        split_idx=0,
        split_num=1,
    )

    # test_dataset = Dataset_Union_ALL_Val(
    # paths=["./data/validation/experimental/heart/hearts"],
    # mode="Val",
    # transform=tio.Compose(
    # [
    # tio.ToCanonical(),
    # tio.CropOrPad(target_shape=(128, 128, 128)),
    # ]
    # ),
    # threshold=0,
    # pcc=False,
    # get_all_meta_info=True,
    # )

    test_dataloader = DataLoader(dataset=test_dataset, sampler=None, batch_size=1, shuffle=True)

    print(len(test_dataset))

    # for i, j, n in test_dataloader:
    for i, j in test_dataloader:
        print(i.shape)
        # print(j.shape)
        # print(n)
        print(j)
