import os, torch
import numpy as np
import pandas as pd
from PIL import ImageFile
from random import shuffle
from joblib import Parallel, delayed
from tqdm import tqdm
from torch.utils.data import Dataset
from nibabel import Nifti1Image
from nilearn.image import load_img, smooth_img, clean_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_atlas_aal, fetch_atlas_destrieux_2009, fetch_atlas_harvard_oxford
from sklearn.model_selection import StratifiedKFold

from loguru import logger

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ADNI(Dataset):

    def __init__(self, 
        caps_dir,
        caps_type,
        atlases_dir,
        split_filepath,
        num_classes,
        depth_of_slice,
        slicing_stride,
        preserve_img_shape_in_slices=True,
        k_fold=1,
        parallize_brains_not_slices=True,
        n_processes=None,
        overwrite_cache=False,
        save_every=500
    ):
        """
        Args:
            caps_dir (str): CAPS directory containing subjects folder with all processed subjects. 
            caps_type (str): Type of data to use (i.e, t1_volume or t1_linear).
            atlases_dir (str): Directory containing atlases directories containing each atlas template.
            split_filepath (str): filepath containing subject list and their diagnosis (labels).
            num_classes (int): Number of classes to use (i.e, 2 for 2, 3, or 5).
            depth_of_slice (int): Depth of each slice per brain 3D MRI image.
            slicing_stride (int): Stride to use when slicing each brain 3D MRI image.
            k_fold (int): Number of folds to use for cross-validation if needed. (Default: 1)
            parallize_brains_not_slices (bool): Whether to parallize over brains or slices. (Default: True)
            n_processes (int): Number of processes to use when parallizing. (Default: None, os.cpu_count()- 1)
        """
        self.preserve_img_shape_in_slices = preserve_img_shape_in_slices

        self.num_classes = num_classes
        if self.num_classes == 5:
            LABEL_MAPPING = ["CN", "MCI", "AD", "LMCI", "EMCI"]
        elif self.num_classes == 3:
            LABEL_MAPPING = ["CN", "MCI", "AD"]
        elif self.num_classes == 2:
            LABEL_MAPPING = ["CN", "AD"]
        self.LABEL_MAPPING = LABEL_MAPPING     
    
        self.mode = split_filepath.split('/')[-1].split('.')[0]
        subject_tsv = pd.io.parsers.read_csv(os.path.join(split_filepath), sep='\t')
        
        logger.success(f'Loaded {self.mode} split with {len(subject_tsv)} subjects')

        # Clean sessions without labels
        indices_not_missing = []
        for i in range(len(subject_tsv)):
            if 'train' in self.mode: # TODO verify
                if (subject_tsv.iloc[i].diagnosis in LABEL_MAPPING):
                    indices_not_missing.append(i)
            else:
                if (subject_tsv.iloc[i].diagnosis in LABEL_MAPPING):
                    indices_not_missing.append(i)
                    
        logger.success(f'Cleaned {len(subject_tsv) - len(indices_not_missing)} subjects without labels')

        self.subject_tsv = subject_tsv.iloc[indices_not_missing]
        self.slicing_stride = slicing_stride
        self.depth_of_slice = depth_of_slice
        
        self.dir_to_scans = os.path.join(caps_dir, 'subjects')        
        
        experiment_name = f'prepared_adni_{self.mode}_{self.num_classes}classes_{self.depth_of_slice}ds_{self.slicing_stride}ss'

        if self.preserve_img_shape_in_slices:
            logger.info(f'Preserving original image shape in slices')
            experiment_name += '_preservedshape'


        self.experiment_dir = os.path.join(caps_dir, 'experiments', experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        logger.info(f'Experiment dir: {self.experiment_dir}')
        
        slices_stack_dict_filepath = os.path.join(self.experiment_dir, f'{experiment_name}.pth')
        self.slices_stack_list = [None for _ in range(len(self.subject_tsv))]
        if os.path.isfile(slices_stack_dict_filepath) and not overwrite_cache:
            self.slices_stack_list = torch.load(slices_stack_dict_filepath)['slices_stack_list']
            logger.success(f'Loaded already existing slices.')    

        skip_indices_from_slices_stack_list = [
            slices_stack for slices_stack in self.slices_stack_list 
            if slices_stack is not None
        ]
        if len(skip_indices_from_slices_stack_list) != len(self.subject_tsv):
            if len(skip_indices_from_slices_stack_list) > 0:
                logger.warning(f'Found {len(skip_indices_from_slices_stack_list)} finished slices stacks but {len(self.subject_tsv)} subjects. Continuing..')
                
            logger.info('Preparing data and generating slices stacks..')
            
            if caps_type == "t1_volume":
                self.suffix = 't1/spm/segmentation/normalized_space'
                self.seg_substring = 'Space_T1w'
            elif caps_type == 't1_linear':
                self.suffix = 't1_linear'
                self.seg_substring = 'Sym_res-1x1x1_T1w.nii'
            else:
                raise Exception(f'CAPS type {caps_type} is not supported')
            

            logger.info(f'Loading ADNI ({caps_type}) dataset from {caps_dir}..')
            
            self.load_atlases(atlases_dir)

            self.parallized_brains = parallize_brains_not_slices
            self.n_processes = max(1, os.cpu_count() - 2) if n_processes is None else n_processes

            
            def get_slices(item):
                idx, row = item
                participant_id = row['participant_id']
                session_id = row['session_id']   
                slices = self.get_prepared_sample(idx, participant_id, session_id)
                return idx, slices

            self.n_parallel_brains = self.n_processes
            if not self.parallized_brains:
                self.n_parallel_brains = 1
                
            logger.info(f'ADNI ({self.mode}) {caps_type} prepared dataset to preprocess using {self.n_parallel_brains} processes')

            
            if len(skip_indices_from_slices_stack_list) > 0:
                logger.warning(f'Skipping {len(skip_indices_from_slices_stack_list)} already processed subjects')
                subjects_to_process = subject_tsv.drop(skip_indices_from_slices_stack_list)
            else:
                subjects_to_process = subject_tsv
            
            
            # cache_dir = os.path.join(self.experiment_dir, "joblib_cache")
            # os.makedirs(cache_dir, exist_ok=True)
            with tqdm(
                total=subjects_to_process.shape[0],
                desc="Preparing subject brains in parallel",
                position=0,
                leave=True,
            ) as pbar:   
                for idx, slices in Parallel(
                    n_jobs=self.n_parallel_brains,
                    return_as='generator_unordered',
                    # temp_folder=cache_dir,
                    max_nbytes='10M',
                    mmap_mode='c',
                    # prefer='threads',
                    # require='sharedmem',
                    # verbose=5
                )(
                    delayed(get_slices)(item) for item in subjects_to_process.iterrows()
                ):
                    self.slices_stack_list[idx] = slices
                    # save ocassionaly
                    if idx % save_every == 0:
                        torch.save({'slices_stack_list': self.slices_stack_list}, slices_stack_dict_filepath)
                    pbar.update(1)
                 
            torch.save({'slices_stack_list': self.slices_stack_list}, slices_stack_dict_filepath)            
            logger.success('Saved succesfully generated prepared slices.')
            
            
        self.nums_nodes = []
        for slices_per_atlas in self.slices_stack_list[0]:
            self.nums_nodes.append(slices_per_atlas.shape[1])

        logger.success(f'Retrieved # of nodes: {self.nums_nodes}')

        self.full_sample_id_list = list(range(len(self.subject_tsv)))

        if k_fold is None or k_fold <= 1:
            self.k_fold = 1
            self.sample_id_list = self.full_sample_id_list
            logger.success(f'No fold / 1-fold applied hence sample_id_list=full_sample_id_list')
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.k = None
            logger.success(f'Using stratified shuffled KFold of {k_fold}')

        
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

    def load_atlases(self, atlases_dir):
        # if roi=='schaefer':
        self.atlases_names = ['aal', 'cc200', 'schaefer']
        self.roi_maskers: NiftiLabelsMasker = []
        
        atlases = [
            fetch_atlas_aal(data_dir=os.path.join(atlases_dir)),
            # Commented out as orginally  meant for ADHD
            {
                "maps": os.path.join(atlases_dir, 'cc200', 'cc200_roi_atlas.nii.gz'),
                "labels": 200
            },
            fetch_atlas_schaefer_2018(data_dir=os.path.join(atlases_dir)),
        ]
        
        cache_dir = os.path.join(self.experiment_dir, "nilearn_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        for atlas in atlases:
            # TODO Add other sides/rotations
            atlas_img = load_img(atlas['maps'])
            # Note Standarize to false as already applied.. if applied again will lead to zeros
            atlas_labels = atlas.get('labels')
            self.roi_maskers.append(
                NiftiLabelsMasker(
                    atlas_img, labels=atlas_labels,
                    standardize=False,
                    # verbose=5,
                    resampling_target="labels" if self.preserve_img_shape_in_slices else "data",
                    memory=cache_dir,
                )
            )
            self.roi_maskers[-1].fit()
        
        logger.success(f'Loaded {len(self.roi_maskers)} atlases: {self.atlases_names}')
    
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
        
        slices = self.slices_stack_list[sample_id]
        
        return {
            "id": sample_id,
            "slices_per_atlas": slices,
            "label": label
        }

    def get_prepared_sample(self, idx, participant_id, session_id):
        sample_id = f"{participant_id}_{session_id}"
        try:
            path = os.path.join(
                self.dir_to_scans,
                participant_id,
                session_id, 
                self.suffix
            )
            all_segs = list(os.listdir(path))
            img_filename = [
                seg_name for seg_name in all_segs 
                if self.seg_substring in seg_name
            ]
            assert len(img_filename) == 1
            img_filename = img_filename[0]
            slices = self.prepare_brain(
                idx,
                load_img(os.path.join(path, img_filename)),
            )

        except Exception as e:
            logger.error(f"Failed to load #{sample_id}: {path}")
            logger.error(f"Errors encountered: {e}")
            # print(traceback.format_exc())
            raise e

        return slices

    def prepare_brain(self, brain_idx, img_nib):
        # def preprocess_slices(img_slices):
        #     imgs_nib_slices = []
        #     for img_slice in img_slices:
        #         imgs_nib_slices.append(Nifti1Image(img_slice, affine=img_nib.affine))
            
        #     imgs_slices_trans = []
        #     for masker in tqdm(
        #         self.roi_maskers, 
        #         desc=f"Extracting feats for brain #{brain_idx} using {len(self.roi_maskers)} atlases",
        #         position=brain_idx%self.n_parallel_brains + 1,
        #         leave=False
        #     ):
        #         imgs_slices_trans.append(masker.transform(imgs_nib_slices))
        #     return imgs_slices_trans
        
        def preprocess_slice(img_slice):
            img_nib_slice = Nifti1Image(img_slice.squeeze(), affine=img_nib.affine)
            img_slice_trans = []
            for masker in self.roi_maskers:
                img_slice_trans.append(masker.transform(img_nib_slice).squeeze())
            return img_slice_trans
        
        # Dropping one 2d slice to have equal 3D slices later
        img = img_nib.get_data()
        # img = img[:,:,:-1] 
        # Set dead pixels to 0
        img[np.isnan(img)] = 0.0
        
        
        # Normalize to [0,1]
        img = (img - img.min())/(img.max() - img.min() + 1e-6)
        
        #assert if image is normalized
        assert img.max() <= 1.0, f"Max value is {img.max()}"
        assert img.min() >= 0.0, f"Min value is {img.min()}"
        
        # img = np.expand_dims(img, axis=0)
        
        assert self.depth_of_slice <= img.shape[-1], f"Depth of slice {self.depth_of_slice} is larger than image size {img.shape[-1]}"
        slices = np.lib.stride_tricks.sliding_window_view(
            img, 
            window_shape=(*img.shape[:-1], self.depth_of_slice), 
            axis=(0,1,2)
        ).squeeze()
        
        def pad_slice(slice, start_amount, end_amount):
            # padding slice to last dimension
            if start_amount > 0:
                slice = np.pad(
                    slice, 
                    ((0, 0), (0, 0), (0, start_amount)), 
                    mode='constant', 
                    constant_values=0
                )
            if end_amount > 0:
                slice = np.pad(
                    slice, 
                    ((0, 0), (0, 0), (end_amount, 0)), 
                    mode='constant', 
                    constant_values=0
                )
            return slice
        

        logger.trace(f'Brain {img.shape} -> Slices {slices.shape}')

        slices = slices[::self.slicing_stride, :, :, :]
        logger.trace(f'Brain -> (after stride) {slices.shape})')
        
        
        if self.preserve_img_shape_in_slices:
            logger.trace(f'Preserving original image shape in slices; padding with zeros')
            # pad each slice to match the size of the original img
            padded_slices = []
            for i, slice in enumerate(slices):
                slice_idx = i * self.slicing_stride
                padded_slices.append(
                    pad_slice(
                        slice, 
                        start_amount=slice_idx,
                        end_amount=img.shape[-1] - self.depth_of_slice - slice_idx
                    )
                )                
            del slices
            slices = np.array(padded_slices)
            del padded_slices    
        
            logger.trace(f'Brain -> (after padding) {slices.shape})')
        

        
        
        
        # data = [[] for _ in self.roi_maskers]
        # for ss in preprocess_slices(slices):
        #     for r_i, s in enumerate(ss):
        #         data[r_i].append(s)                
        
        
        if self.parallized_brains:
            logger.trace(f'Parallizing over brains processing')
            mapping_func = map
        else:
            logger.trace(f'Parallizing over slices processing')
            mapping_func = lambda func, itr: Parallel(n_jobs=self.n_processes, verbose=5)(
                delayed(func)(item) for item in itr
            )
        data = [[] for _ in self.roi_maskers]

        with tqdm(
            slices, total=len(slices),
            desc=f"Extracting slices features for brain #{brain_idx}",
            position=(brain_idx%self.n_parallel_brains) + 2,
            leave=False,
            disable=True
        ) as slices_pbar:
            for ss in mapping_func(preprocess_slice, slices):
                for r_i, s in enumerate(ss):
                    data[r_i].append(s)                
                slices_pbar.update(1)
        
        for r_i in range(len(self.roi_maskers)):
            data[r_i] = torch.tensor(
                np.array(data[r_i]), 
                dtype=torch.float32
            ).to_dense()
        return data