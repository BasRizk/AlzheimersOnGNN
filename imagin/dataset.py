import os, torch
import numpy as np
from PIL import ImageFile
import random 
import numpy as np
import pandas as pd
from random import shuffle
import random
# from augmentations import *
import traceback
from tqdm import tqdm
from torch import float32, save, load
from torch.utils.data import Dataset
from nibabel import Nifti1Image
from nilearn.image import load_img, smooth_img, clean_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_atlas_aal, fetch_atlas_destrieux_2009, fetch_atlas_harvard_oxford
from sklearn.model_selection import StratifiedKFold
from multiprocessing.pool import ThreadPool # TODO change to Parallel

from loguru import logger

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ADNI(Dataset):

    def __init__(self, 
        data_dir='../data', 
        data_type='t1_linear',
        atlas_dir='../data',
        splits_dir= '../splits',
        mode='Train',
        num_classes=3,
        depth_of_slice=40,
        k_fold=1,
    #  smoothing_fwhm=None,
        slicing_stride=5,
        parallize_brains=True
    ):
        if data_type == "t1_volume":
            self.suffix = 't1/spm/segmentation/normalized_space'
            self.seg_substring = 'Space_T1w'
        elif data_type == 't1_linear':
            self.suffix = 't1_linear'
            self.seg_substring = 'Sym_res-1x1x1_T1w.nii'
        else:
            raise Exception(f'Data type {data_type} is not supported')

        self.parallized_brains = parallize_brains
        self.n_processes = os.cpu_count() - 1
        logger.debug(f'ADNI ({mode}) {data_type} prepared dataset to preprocess using {self.n_processes} processes')
        if self.parallized_brains:
            logger.debug('..Parellizing over brains processing')
        else:
            logger.debug('..Parallizing over slices processing')
        self.threading_pool = ThreadPool(self.n_processes)
        
        self.num_classes = num_classes
        if self.num_classes == 3:
            LABEL_MAPPING = ["CN", "MCI", "AD"]
        elif self.num_classes == 2:
            LABEL_MAPPING = ["CN", "AD"]
        self.LABEL_MAPPING = LABEL_MAPPING     
    
        subject_tsv = pd.io.parsers.read_csv(
            os.path.join(splits_dir, mode + '_diagnosis_ADNI.tsv'),
            sep='\t'
        )

        # Clean sessions without labels
        indices_not_missing = []
        for i in range(len(subject_tsv)):
            if mode == 'Train':
                if (subject_tsv.iloc[i].diagnosis in LABEL_MAPPING):
                    indices_not_missing.append(i)
            else:
                if (subject_tsv.iloc[i].diagnosis in LABEL_MAPPING):
                    indices_not_missing.append(i)

        self.subject_tsv = subject_tsv.iloc[indices_not_missing]
        self.dir_to_scans = os.path.join(data_dir, 'processed', 'subjects')
        self.mode = mode
        self.slicing_stride = slicing_stride
        self.depth_of_slice = depth_of_slice
        self.experiment_name = f'prepared_adni_{self.mode}_{self.num_classes}classes_{self.depth_of_slice}ds'
        self.load_atlases(atlas_dir)

        if os.path.isfile(os.path.join(data_dir, f'{self.experiment_name}.pth')):
            self.slices_stack_dict = load(os.path.join(data_dir, f'{self.experiment_name}.pth'))
            logger.success(f'Loaded saved slices.')

        else:
            logger.info('Preparing data and generating slices..')
            self.slices_stack_dict = {}
            def get_slices(item):
                idx, row = item
                participant_id = row['participant_id']
                session_id = row['session_id']   
                slices = self.get_prepared_sample(participant_id, session_id)
                return idx, slices

            if self.parallized_brains:
                for idx, slices in tqdm(
                    self.threading_pool.imap_unordered(
                        get_slices,
                        subject_tsv.iterrows(),
                        chunksize=self.n_processes
                    ),
                    ncols=60, total=subject_tsv.shape[0],
                    desc="Preparing subject brains in parallel"
                ):
                    self.slices_stack_dict[idx] = slices
            else:
                for idx, row in tqdm(
                    subject_tsv.iterrows(), ncols=60, total=subject_tsv.shape[0],
                    desc="Preparing subject brains sequentially"
                ): 
                    _, self.slices_stack_dict[idx] = get_slices((idx, row))

            save(self.slices_stack_dict, os.path.join(data_dir, f'{self.experiment_name}.pth'))
            
            logger.success('Saved succesfully generated prepared slices.')
            
        _somekey = list(self.slices_stack_dict.keys())[0]
        self.nums_nodes = []
        for i in range(len(self.roi_maskers)):
            self.nums_nodes.append(self.slices_stack_dict[_somekey][i].shape[1])

        logger.success(f'Retrived # of nodes: {self.nums_nodes}')

        self.full_sample_id_list = list(self.slices_stack_dict.keys())

        if k_fold <= 1:
            self.k_fold = 1
            self.sample_id_list = self.full_sample_id_list
            logger.success(f'No fold / 1-fold applied hence sample_id_list=full_sample_id_list')
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.k = None
            logger.success(f'Using stratified shuffled KFold of {k_fold}')

        self.threading_pool.close()
        
    def set_fold(self, fold, train=True):
        # TODO merge with logic from above to encapsulate logic
        assert self.k_fold is not None and self.k_fold > 1
        self.k = fold
        
        labels = self.subject_tsv.iloc[self.full_sample_id_list]['diagnosis'].to_list()
        train_idx, test_idx = list(
            self.k_fold.split(self.full_sample_id_list, labels)
        )[fold]
        if train:
            shuffle(train_idx)
            self.sample_id_list = [
                self.full_sample_id_list[idx]
                for idx in train_idx
            ]
        else:
            self.sample_id_list = [
                self.full_sample_id_list[idx]
                for idx in test_idx
            ]

    def load_atlases(self, atlas_dir):
        # if roi=='schaefer':
        self.atlases_names = ['aal', 'cc200', 'schaefer']
        self.roi_maskers: NiftiLabelsMasker = []
        atlases = [
            fetch_atlas_aal(data_dir=os.path.join(atlas_dir, 'roi')),
            # Commented out as orginally  meant for ADHD
            {
                "maps": os.path.join(atlas_dir, 'roi', 'cc200', 'cc200_roi_atlas.nii.gz'),
                "labels": 200
            },
            fetch_atlas_schaefer_2018(data_dir=os.path.join(atlas_dir, 'roi')),
        ]
        for atlas in atlases:
            # TODO Add other sides/rotations
            atlas_img = load_img(atlas['maps'])
            # Note Standarize to false as already applied.. if applied again will lead to zeros
            atlas_labels = atlas.get('labels')
            self.roi_maskers.append(
                NiftiLabelsMasker(
                    atlas_img, labels=atlas_labels, standardize=False,
                    # memory="nilearn_cache",
                    # verbose=5,
                    resampling_target="labels"
                )
            )        
            self.roi_maskers[-1].fit()
    
    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, idx):
        sample_id = self.sample_id_list[idx]
        if self.subject_tsv.iloc[sample_id].diagnosis == 'CN':
            label = 0
        elif self.subject_tsv.iloc[sample_id].diagnosis == 'MCI':
            label = 1
        elif self.subject_tsv.iloc[sample_id].diagnosis == 'AD':
            if self.LABEL_MAPPING == ["CN", "AD"]:
                label = 1
            else:
                label = 2
        else:
            logger.error('WRONG LABEL VALUE!!!')
            label = -100
        
        slices = self.slices_stack_dict[sample_id]
        
        # TODO ensure it works
        return {
            "id": sample_id,
            "slices_per_atlas": slices,
            "label": label
        }
        # return image.astype(np.float32),label,idx_out,mmse,cdr_sub,age, os.path.join(path,seg_name)


    def get_prepared_sample(self, participant_id, session_id):
        sample_id = f"{participant_id}_{session_id}"
        try:

            path = os.path.join(self.dir_to_scans,participant_id,
                    session_id, self.suffix)
            all_segs = list(os.listdir(path))
            img_filename = [
                seg_name for seg_name in all_segs 
                if self.seg_substring in seg_name
            ]
            assert len(img_filename) == 1
            img_filename = img_filename[0]
            slices = self.prepare_brain(
                load_img(os.path.join(path, img_filename))
            )

        except Exception as e:
            logger.error(f"Failed to load #{sample_id}: {path}")
            logger.error(f"Errors encountered: {e}")
            print(traceback.format_exc())
            return None

        return slices

    def prepare_brain(self, img_nib):
        def preprocess_slice(img_slice):
            img_nib_slice = Nifti1Image(img_slice.squeeze(), affine=img_nib.affine)
            img_slice_trans = []
            for masker in self.roi_maskers:
                img_slice_trans.append(masker.transform(img_nib_slice).squeeze())
            return img_slice_trans
            
        # Dropping one 2d slice to have equal 3D slices later
        img = img_nib.get_data()
        # img = img[:,:,:-1] 
        img[np.isnan(img)] = 0.0
        img = (img - img.min())/(img.max() - img.min() + 1e-6)
        # img = np.expand_dims(img, axis=0)
        # if self.mode == 'Train':
        #     img = self.augment_image(img)
        # if self.mode == 'Train':
        #     img = self.randomCrop(img,96,96,96)
        # else:
        #     img = self.centerCrop(img,96,96,96)
            
        # breakpoint()
        # slices = np.array_split(img, self.num_slices, axis=-1) 

        slices = np.lib.stride_tricks.sliding_window_view(
            img, 
            window_shape=(*img.shape[:-1], self.depth_of_slice), 
            axis=(0,1,2)
        ).squeeze()


        slices = slices[::self.slicing_stride, :, :, :]
        
        data = [[] for _ in self.roi_maskers]

        if self.parallized_brains:
            mapping_func = map
            # logger.info(f'processing {len(slices)} slices sequentially')
        else:
            mapping_func = lambda func, itr: self.threading_pool.map(
                func, itr, chunksize=self.n_processes
            )
            # logger.info(f'processing {len(slices)} slices in parallel')
            
        for ss in mapping_func(preprocess_slice, slices):
            for r_i, s in enumerate(ss):
                data[r_i].append(s)
        
        for r_i in range(len(self.roi_maskers)):
            data[r_i] = torch.tensor(
                np.array(data[r_i]), 
                dtype=float32
            ).to_dense()
        return data



    # def centerCrop(self, img, length, width, height):
    #     assert img.shape[1] >= length
    #     assert img.shape[2] >= width
    #     assert img.shape[3] >= height

    #     x = img.shape[1]//2 - length//2
    #     y = img.shape[2]//2 - width//2
    #     z = img.shape[3]//2 - height//2
    #     img = img[:,x:x+length, y:y+width, z:z+height]
    #     return img

    # def randomCrop(self, img, length, width, height):
    #     assert img.shape[1] >= length
    #     assert img.shape[2] >= width
    #     assert img.shape[3] >= height
    #     x = random.randint(0, img.shape[1] - length)
    #     y = random.randint(0, img.shape[2] - width)
    #     z = random.randint(0, img.shape[3] - height )
    #     img = img[:,x:x+length, y:y+width, z:z+height]
    #     return img
