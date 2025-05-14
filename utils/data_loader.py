from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchio as tio
from torchio.data.io import sitk_to_nib
import torch
import numpy as np
import os
import torch
import SimpleITK as sitk
from prefetch_generator import BackgroundGenerator


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
            except:
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
                crop_mask[random_index[0]][random_index[1]][random_index[2]][
                    random_index[3]
                ] = 1
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
                    label_path = os.path.join(
                        path, f"labels{self.data_type}", f"{base}.nii.gz"
                    )
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
        self.image_paths = self.image_paths[self.split_idx :: self.split_num]
        self.label_paths = self.label_paths[self.split_idx :: self.split_num]


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

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=sitk_image_arr),
        )

        if "/ct_" in self.image_paths[index]:
            subject = tio.Clamp(-1000, 1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print("Could not transform", self.image_paths[index])

        if self.pcc:
            print("using pcc setting")
            # crop from random click point
            random_index = torch.argwhere(subject.label.data == 1)
            if len(random_index) >= 1:
                random_index = random_index[np.random.randint(0, len(random_index))]
                crop_mask = torch.zeros_like(subject.label.data)
                crop_mask[random_index[0]][random_index[1]][random_index[2]][
                    random_index[3]
                ] = 1
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
                    image_path = os.path.join(
                        path, f"{self.data_type}", f"{base}.nii.gz"
                    )
                    self.image_paths.append(image_path)
                    
        self.image_paths = self.image_paths[self.split_idx :: self.split_num]


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

        # if "/ct_" in self.image_paths[index]:
        #     subject = tio.Clamp(-1000, 1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
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

class SegFM3D_Dataset(Dataset):
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
        npz_file = np.load(self.image_paths[index])
        image = npz_file['imgs'][None] # 3D -> 4D
        label = npz_file['gts'][None]

        unique_nonzero_labels = np.unique(label)[1:]
        if len(unique_nonzero_labels) <= 0:
            print("No non-zero labels found in the resampled ground truth of", self.image_paths[index], unique_nonzero_labels)
            return self.__getitem__(np.random.randint(self.__len__()))
        selected_label_idx = np.random.randint(len(unique_nonzero_labels))
        label[label != unique_nonzero_labels[selected_label_idx]] = 0

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.tensor(image, dtype=torch.float32)),
            label=tio.LabelMap(tensor=torch.tensor(label, dtype=torch.float32)),
        )

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print("fail to load data:", self.image_paths[index])

        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))
        return {
            "image": subject.image.data.clone().detach(),
            "label": subject.label.data.clone().detach(),
            "spacing": npz_file['spacing'],
        }

    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        # if ${path}/labelsTr exists, search all .nii.gz
        for path in paths:
            d = path
            if os.path.exists(d):
                for name in os.listdir(d):
                    base = os.path.basename(name).split(".npz")[0]
                    img_path = os.path.join(
                        path, f"{base}.npz"
                    )
                    self.image_paths.append(img_path)
                    self.label_paths.append(img_path)




if __name__ == "__main__":
    from data_paths import img_datas
    test_dataset = SegFM3D_Dataset(paths=img_datas, transform=tio.Compose([
        tio.ToCanonical(),
        # tio.CropOrPad(mask_name='label', target_shape=(256, 256, 256)), # crop only object region
        # tio.RandomFlip(axes=(0, 1, 2)),
    ]),
    threshold=1000)

    test_dataloader = DataLoader(
        dataset=test_dataset, sampler=None, batch_size=1, shuffle=True
    )

    print(len(test_dataset))
    
    # for i, j, n in test_dataloader:
    # for data in test_dataloader:
    #     # import pdb; pdb.set_trace()
    #     print(*(data["image"].shape), *(data["label"].shape), *list(data['spacing'].tolist()[0]))

    import os
    from tqdm import tqdm

    def mask2D_to_bbox(gt2D, file):
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        bbox_shift = np.random.randint(0, 6, 1)[0]
        scale_y, scale_x = gt2D.shape
        bbox_shift_x = int(bbox_shift * scale_x/256)
        bbox_shift_y = int(bbox_shift * scale_y/256)
        #print(f'{bbox_shift_x=} {bbox_shift_y=} with orig {bbox_shift=}')
        x_min = max(0, x_min - bbox_shift_x)
        x_max = min(W-1, x_max + bbox_shift_x)
        y_min = max(0, y_min - bbox_shift_y)
        y_max = min(H-1, y_max + bbox_shift_y)
        boxes = np.array([x_min, y_min, x_max, y_max])
        return boxes

    def mask3D_to_bbox(gt3D, file):
        b_dict = {}
        z_indices, y_indices, x_indices = np.where(gt3D > 0)
        z_min, z_max = np.min(z_indices), np.max(z_indices)
        # middle of z_indices
        z_middle = z_indices[len(z_indices)//2]
        D, H, W = gt3D.shape
        b_dict['z_min'] = z_min
        b_dict['z_max'] = z_max
        b_dict['z_mid'] = z_middle

        gt_mid = gt3D[z_middle]

        box_2d = mask2D_to_bbox(gt_mid, file)
        x_min, y_min, x_max, y_max = box_2d
        b_dict['z_mid_x_min'] = x_min
        b_dict['z_mid_y_min'] = y_min
        b_dict['z_mid_x_max'] = x_max
        b_dict['z_mid_y_max'] = y_max

        assert z_min == max(0, z_min)
        assert z_max == min(D-1, z_max)
        return b_dict

    class BufferedLogger:
        def __init__(self, file_path, buffer_size=50):
            self.file_path = file_path
            self.buffer_size = buffer_size
            self.buffer = []

        def log(self, message):
            self.buffer.append(message)
            if len(self.buffer) >= self.buffer_size:
                self.flush()

        def flush(self):
            with open(self.file_path, 'a') as f:
                for message in self.buffer:
                    f.write(message + '\n')
            self.buffer = []

        def close(self):
            if self.buffer:
                self.flush()

    def process_data_and_log(test_dataloader, logger):
        for data in tqdm(test_dataloader):
            b_dict = mask3D_to_bbox(data['label'].squeeze(), "test_fname")
            gt_shape = (b_dict["z_max"]-b_dict["z_min"], 
                        b_dict["z_mid_x_max"]-b_dict["z_mid_x_min"], 
                        b_dict["z_mid_y_max"]-b_dict["z_mid_y_min"])
            spacing = data['spacing'].tolist()[0]
            phy_shape = (gt_shape[0]*spacing[2], gt_shape[1]*spacing[0], gt_shape[2]*spacing[1])
            message = f"{list(data['spacing'].tolist()[0])} {gt_shape} {phy_shape} {data['image'].shape}"
            logger.log(message)


    log_file_path = 'all_info.log'
    logger = BufferedLogger(log_file_path)

    try:
        process_data_and_log(test_dataloader, logger)
    finally:
        logger.close()
