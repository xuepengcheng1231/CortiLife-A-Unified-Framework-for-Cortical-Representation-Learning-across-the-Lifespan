import re

import clip
import numpy as np
from nltk import RegexpTokenizer
from sympy import shape
from torchvision.datasets import VisionDataset
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from util.utils import *
import torch
import random

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask # [196]

class DataAugmentationForMAE_parcel(object):
    def __init__(self, args):
        #imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        #mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        #std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        # self.transform = transforms.Compose([
        #     # transforms.RandomResizedCrop(args.input_size),
        #     transforms.ToTensor(),
        #     # transforms.Normalize(
        #     #     mean=torch.tensor(mean),
        #     #     std=torch.tensor(std))
        # ])
        self.max_noise_level = 0.15
        # self.masked_position_generator = RandomMaskingGenerator(
        #     args.window_size, args.mask_ratio
        # )
        self.neighbor = Get_neighs_order_2ring(40962)
        self.parcel_index = Get_parcellation()
        self.neighbor = np.concatenate([self.neighbor, self.neighbor + 40962])
        self.parcel_index = np.concatenate([self.parcel_index, self.parcel_index + 40962])
        #self.mean = np.load("/data1/dataset/NSD_surface/data_preprocess/processed_data/mean_010205.npy")
        #self.std = np.load("/data1/dataset/NSD_surface/data_preprocess/processed_data/std_010205.npy")
        #self.ToTensor = torch.tensor()
    def __call__(self, x):
        #print(x)
        #x = np.nan_to_num(x)
        #x = (x - self.mean) / self.std
        #print(x)
        #x = np.nan_to_num(x)
        x = self.augment_for_spherical(x)
        x = torch.tensor(x)
        x = torch.nan_to_num(x)
        x_parcel = x[:, :, self.parcel_index]
        x_parcel = x_parcel.transpose(-1,-2).contiguous().reshape([2, -1, 640])
        #print(x)
        #noise_level = random.uniform(0, self.max_noise_level)
        #noise = torch.randn(x.size()) * noise_level
        #x = x + noise
        # return x, self.masked_position_generator()
        return x_parcel

    def augment_for_spherical(self, x):
        x = x[:,self.neighbor].reshape([1,81924,19])
        random_index_1 = np.random.randint(0,19,size=(1,81924,9))
        random_index_2 = np.random.randint(0,19,size=(1,81924,9))
        x1 = np.mean(np.take_along_axis(x,random_index_1,axis=2),axis=2)
        x2 = np.mean(np.take_along_axis(x,random_index_2,axis=2),axis=2)
        x1 = np.expand_dims(x1,axis=0)
        x2 = np.expand_dims(x2,axis=0)
        x = np.concatenate([x1,x2],axis=0)
        # print(x.shape)
        return x

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str('Z-Score')
        repr += "  transform = %s,\n" % str('To Tensor')
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

class DataAugmentationForParcel(object):
    def __init__(self, args):
        #imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        #mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        #std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        # self.transform = transforms.Compose([
        #     # transforms.RandomResizedCrop(args.input_size),
        #     transforms.ToTensor(),
        #     # transforms.Normalize(
        #     #     mean=torch.tensor(mean),
        #     #     std=torch.tensor(std))
        # ])
        # self.max_noise_level = 0.15
        # self.masked_position_generator = RandomMaskingGenerator(
        #     args.window_size, args.mask_ratio
        # )
        # self.neighbor = Get_neighs_order_2ring(40962)
        self.parcel_index = Get_parcellation()
        self.num_patches = 640
        # self.neighbor = np.concatenate([self.neighbor, self.neighbor + 40962])
        self.parcel_index = np.concatenate([self.parcel_index, self.parcel_index + 40962])
        #self.mean = np.load("/data1/dataset/NSD_surface/data_preprocess/processed_data/mean_010205.npy")
        #self.std = np.load("/data1/dataset/NSD_surface/data_preprocess/processed_data/std_010205.npy")
        #self.ToTensor = torch.tensor()
    def __call__(self, x):
        #print(x)
        #x = np.nan_to_num(x)
        #x = (x - self.mean) / self.std
        #print(x)
        #x = np.nan_to_num(x)
        # x = self.augment_for_spherical(x)
        # print(x.shape)
        x = torch.tensor(x)
        x = torch.nan_to_num(x)
        x_parcel = x[:, self.parcel_index]
        # x_parcel = torch.mean(x_parcel, axi)
        x_parcel = x_parcel.transpose(-1,-2).contiguous().reshape([-1, self.num_patches])
        #print(x)
        #noise_level = random.uniform(0, self.max_noise_level)
        #noise = torch.randn(x.size()) * noise_level
        #x = x + noise
        # return x, self.masked_position_generator()
        return x_parcel

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str('Z-Score')
        repr += "  transform = %s,\n" % str('To Tensor')
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def make_dataset(directory, image_path, text_path):
    surfaces = np.load(os.path.join(directory, image_path))
    text_embeddings = np.load(os.path.join(directory, text_path))
    # surfaces = np.concatenate([surfaces[:, :, :40962], surfaces[:, :, 163842:163842 + 40962]], axis=1)
    # mean = np.load("./data_surfclip/No_ADHD_mean.npy").reshape([1,6,1])
    # std = np.load("./data_surfclip/No_ADHD_std.npy").reshape([1,6,1])
    print("successfully load ADHD sMRI data")
    # surface = (surfaces - mean) / std
    surface = surfaces.astype(np.float32)
    print("text",text_embeddings[0].dtype)
    if text_embeddings[0].dtype.kind == 'U':
        text_embeddings = text_embeddings.astype(str)
    else:
        print("label")
        text_embeddings = text_embeddings.astype(np.float32)
    print('surface shape:', surface.shape)
    print('text_embeddings shape:', text_embeddings.shape)
    print(np.isnan(surface).any())
    return surface, text_embeddings

def make_finetune_dataset(directory, image_path, label_path):
    surfaces = np.load(os.path.join(directory, image_path))
    label = np.load(os.path.join(directory, label_path))
    print("successfully load ADHD sMRI data")
    # surface = (surfaces - mean) / std
    surface = surfaces.astype(np.float32)
    label = label.astype(np.float32)
    print('surface shape:', surface.shape)
    print(np.isnan(surface).any())
    return surface, label

# def tokenizer(age, sex):
#     if sex == 1:
#         token = clip.tokenize("This is a "+str(age)+"-year-old male.")
#     else:
#         token = clip.tokenize("This is a "+str(age)+"-year-old female.")
#     return token

class DatasetFolder(VisionDataset):
    def __init__(
            self,
            root: str,
            image_path: str,
            text_path: str,
            loader: Callable[[str], Any],
            transform: Optional[Callable] = None,
            tokenizer: Optional[Callable] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform)
                                            #target_transform=target_transform)
        #classes, class_to_idx = self._find_classes(self.root)
        samples,text_embeddings = make_dataset(self.root, image_path, text_path)

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            # if extensions is not None:
            #     msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.samples = samples
        self.text_embeddings = text_embeddings
        self.tokenizer = tokenizer

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        while True:
            try:
                sample = self.samples[index]
                text_embedding = self.text_embeddings[index]
                if self.tokenizer is not None:
                    text_embedding = self.tokenizer(text_embedding)
                break

            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, text_embedding

    def __len__(self) -> int:
        return len(self.samples)

class DatasetFolder_CAR(VisionDataset):
    def __init__(
            self,
            root: str,
            image_path: str,
            text_path: str,
            loader: Callable[[str], Any],
            transform: Optional[Callable] = None,
            tokenizer: Optional[Callable] = None,
    ) -> None:
        super(DatasetFolder_CAR, self).__init__(root, transform=transform)
                                            #target_transform=target_transform)
        #classes, class_to_idx = self._find_classes(self.root)
        samples,text_embeddings = make_dataset(self.root, image_path, text_path)

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            # if extensions is not None:
            #     msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.samples = samples
        self.text_embeddings = text_embeddings
        self.tokenizer = tokenizer
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        while True:
            try:
                sample = self.samples[index]
                text_embedding = self.text_embeddings[index]
                text_embedding = str(text_embedding)
                if self.tokenizer is not None:
                    # text_embedding = self.tokenizer(text_embedding)
                    caption_ids, attention_mask, token_type_ids = self.process_text(text_embedding)
                break

            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)

        if self.transform is not None:
            sample = self.transform(sample)
        # return sample, text_embedding
        return sample, caption_ids, attention_mask, token_type_ids

    def __len__(self) -> int:
        return len(self.samples)

    def process_text(self, text):

        if isinstance(text, (str, bytes, np.str_, np.bytes_)):
            text = [str(text)]
        elif isinstance(text, (list, tuple)):
            # 把 list 里的 np.str_ 也转掉
            text = [str(t) for t in text]
        else:
            text = [str(text)]

        processed_text_tensors = []
        for t in text:
            # use space instead of newline
            t = t.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(t)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            all_sents = []

            for t in captions:
                t = t.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(t.lower())

                if len(tokens) <= 1:
                    continue

                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                all_sents.append(" ".join(included_tokens))

            t = " ".join(all_sents)

            text_tensors = self.tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=97,
            )
            text_tensors["sent"] = [
                self.ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()
            ]
            processed_text_tensors.append(text_tensors)

        caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
        attention_mask = torch.stack(
            [x["attention_mask"] for x in processed_text_tensors]
        )
        token_type_ids = torch.stack(
            [x["token_type_ids"] for x in processed_text_tensors]
        )

        if len(text) == 1:
            caption_ids = caption_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
            token_type_ids = token_type_ids.squeeze(0)
        else:
            caption_ids = caption_ids.squeeze()
            attention_mask = attention_mask.squeeze()
            token_type_ids = token_type_ids.squeeze()
        caption_ids = caption_ids.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        token_type_ids = token_type_ids.squeeze(0)

        caption_ids = caption_ids.clone().contiguous()
        attention_mask = attention_mask.clone().contiguous()
        token_type_ids = token_type_ids.clone().contiguous()
        # cap_lens = []
        # for txt in text:
        #     cap_lens.append(len([w for w in txt if not w.startswith("[")]))

        return caption_ids,attention_mask,token_type_ids

class Finetune_DatasetFolder(VisionDataset):
    def __init__(
            self,
            root: str,
            image_path: str,
            label_path: str,
            loader: Callable[[str], Any],
            transform: Optional[Callable] = None,
    ) -> None:
        super(Finetune_DatasetFolder, self).__init__(root, transform=transform)
                                            #target_transform=target_transform)
        #classes, class_to_idx = self._find_classes(self.root)
        samples,label = make_finetune_dataset(self.root, image_path, label_path)

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            # if extensions is not None:
            #     msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.samples = samples
        self.label = label

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        while True:
            try:
                sample = self.samples[index]
                label = self.label[index]
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self) -> int:
        return len(self.samples)

def default_loader(path: str):
    data = nib.load(path)
    data = data.get_fdata()
    return data

class SurfaceFolder(DatasetFolder):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            image_path: Optional[str] = None,
            text_path: Optional[str] = None,
            tokenizer: Optional[Callable[[str], Any]] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            #is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(SurfaceFolder, self).__init__(root, image_path,text_path, loader,
                                          transform=transform, tokenizer=tokenizer
                                          #target_transform=target_transform,
                                          )
        self.surfaces = self.samples

class SurfaceFolder_CAR(DatasetFolder_CAR):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            image_path: Optional[str] = None,
            text_path: Optional[str] = None,
            tokenizer: Optional[Callable[[str], Any]] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            #is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(SurfaceFolder_CAR, self).__init__(root, image_path,text_path, loader,
                                          transform=transform, tokenizer=tokenizer
                                          #target_transform=target_transform,
                                          )
        self.surfaces = self.samples

class Finetune_SurfaceFolder(Finetune_DatasetFolder):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            image_path: Optional[str] = None,
            label_path: Optional[str] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            #is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(Finetune_SurfaceFolder, self).__init__(root, image_path,label_path, loader,
                                          transform=transform,
                                          #target_transform=target_transform,
                                          )
        self.surfaces = self.samples


