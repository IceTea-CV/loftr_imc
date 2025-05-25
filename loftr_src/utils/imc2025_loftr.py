import os
import cv2
from PIL import Image
import numpy as np
import torch
import h5py
from tqdm import tqdm
from pdb import set_trace as bb

from loftr_src.utils.merge_kpts import *

def lower_config(yacs_cfg):
    from yacs.config import CfgNode as CN
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def process_resize(w, h, resize, df=None, resize_no_larger_than=False):
    assert(len(resize) > 0 and len(resize) <= 2)
    if resize_no_larger_than and (max(h, w) <= max(resize)):
        w_new, h_new = w, h
    else:
        if len(resize) == 1 and resize[0] > -1:  # resize the larger side
            scale = resize[0] / max(h, w)
            w_new, h_new = int(round(w*scale)), int(round(h*scale))
        elif len(resize) == 1 and resize[0] == -1:
            w_new, h_new = w, h
        else:  # len(resize) == 2:
            w_new, h_new = resize[0], resize[1]

    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w_new, h_new])
    return w_new, h_new

def resize_image(image, size, interp):
    # NOTE: from hloc
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith('pil_'):
        interp = getattr(Image, interp[len('pil_'):].upper())
        resized = Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:])
    pad_size_h, pad_size_w = pad_size, pad_size

    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size_h, pad_size_w), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size_h, pad_size_w), dtype=inp.dtype)
            mask[:inp.shape[0], :inp.shape[1]] = 1
    elif inp.ndim == 3:
        padded = np.zeros((pad_size_h, pad_size_w, inp.shape[-1]), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1], :] = inp
        if ret_mask:
            mask = np.zeros((pad_size_h, pad_size_w), dtype=inp.dtype)
            mask[:inp.shape[0], :inp.shape[1]] = 1
    else:
        raise NotImplementedError()
    return padded, mask

def rgb2tensor(image):
    return torch.from_numpy(image/255.).float().permute(2, 0, 1).contiguous()  # (3, h, w)


def read_rgb(path, resize=None, resize_no_larger_than=False, resize_float=False, df=None, client=None,
                   pad_to=None, ret_scales=False, ret_pad_mask=False,
                   augmentor=None):
    resize = tuple(resize) if resize is not None else None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if image is None:
        print(f"Problem exists when loading image: {path}")
    
    # import ipdb; ipdb.set_trace()
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize if resize is not None else (w, h), df, resize_no_larger_than=resize_no_larger_than)
    scales = torch.tensor([float(h) / float(h_new), float(w) / float(w_new)]) # [2]
    original_hw = torch.tensor([h,w]) #[2]

    image = resize_image(image, (w_new, h_new), interp="pil_LANCZOS").astype('float32')

    if pad_to is not None:
        if pad_to == -1:
            pad_to = max(w_new, h_new)

        image, mask = pad_bottom_right(image, pad_to, ret_mask=ret_pad_mask)

    ts_image = rgb2tensor(image)
    ret_val = [ts_image]

    if ret_scales:
        ret_val += [scales, original_hw]

    return ret_val[0] if len(ret_val) == 1 else ret_val

def grayscale2tensor(image, mask=None):
    return torch.from_numpy(image/255.).float()[None]  # (1, h, w)


def read_grayscale(path, resize=None, resize_no_larger_than=False, resize_float=False, df=None, client=None,
                   pad_to=None, ret_scales=False, ret_pad_mask=False,
                   augmentor=None):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) 

    if image is None:
        print(f"Problem exists when loading image: {path}")

    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize if resize is not None else (w, h), df, resize_no_larger_than=resize_no_larger_than)
    scales = torch.tensor([float(h) / float(h_new), float(w) / float(w_new)]) # [2]
    original_hw = torch.tensor([h,w]) #[2]

    image = resize_image(image, (w_new, h_new), interp="pil_LANCZOS").astype('float32')

    if pad_to is not None:
        if pad_to == -1:
            pad_to = max(w_new, h_new)

        image, mask = pad_bottom_right(image, pad_to, ret_mask=ret_pad_mask)

    ts_image = grayscale2tensor(image)
    ret_val = [ts_image]

    if ret_scales:
        ret_val += [scales, original_hw]

    return ret_val[0] if len(ret_val) == 1 else ret_val

def extract_preds(data):
    """extract predictions assuming bs==1"""
    m_bids = data["m_bids"].cpu().numpy()
    assert (np.unique(m_bids) == 0).all()
    mkpts0 = data["mkpts0_f"].cpu().numpy()
    mkpts1 = data["mkpts1_f"].cpu().numpy()
    mconfs = data["mconf"].cpu().numpy()

    return mkpts0, mkpts1, mconfs






def pairwise_loftr_match_quant(loftr_quant_model, img_path_1, img_path_2,               # List of (idx1, idx2) tuples for images to match
                    device=torch.device('cpu'),
                    max_size_resol=1200,
                    ransac_reproj_threshold_cv=1.0, # RANSAC reprojection threshold for OpenCV's F-matrix
                    verbose_cv_errors=False, logger=None): 
    img_scale0 = read_grayscale(
        img_path_1,
        (max_size_resol,),
        df=8,
        pad_to=None,
        ret_scales=True,
    )
    img_scale1 = read_grayscale(
        img_path_2,
        (max_size_resol,),
        pad_to=None,
        df=8,
        ret_scales=True,
    )

    img0, scale0, original_hw0 = img_scale0
    img1, scale1, original_hw1 = img_scale1
    
    data = {
        "image0": img0[None,:],
        "image1": img1[None,:],
        "scale0": scale0[None,:],  # 1*2
        "scale1": scale1[None,:],
    }

    data_c = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }
    loftr_quant_model(data_c)
    mkpts1, mkpts2, mconfs = extract_preds(data_c)


    return mkpts1, mkpts2


def pairwise_loftr_match_raw(loftr_model, img_path_1, img_path_2,               # List of (idx1, idx2) tuples for images to match
                    device=torch.device('cpu'),
                    max_size_resol=1200,
                    ransac_reproj_threshold_cv=1.0, # RANSAC reprojection threshold for OpenCV's F-matrix
                    verbose_cv_errors=False, logger=None): 
    img_scale0 = read_grayscale(
        img_path_1,
        (max_size_resol,),
        df=8,
        pad_to=None,
        ret_scales=True,
    )
    img_scale1 = read_grayscale(
        img_path_2,
        (max_size_resol,),
        pad_to=None,
        df=8,
        ret_scales=True,
    )

    img0, scale0, original_hw0 = img_scale0
    img1, scale1, original_hw1 = img_scale1
    
    data = {
        "image0": img0[None,:],
        "image1": img1[None,:],
        "scale0": scale0[None,:],  # 1*2
        "scale1": scale1[None,:],
    }

    data_c = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }
    loftr_model(data_c)
    mkpts1, mkpts2, mconfs = extract_preds(data_c)


    return mkpts1, mkpts2



def match_with_loftr_raw(loftr_model, img_fnames,
                    index_pairs,               # List of (idx1, idx2) tuples for images to match
                    feature_dir='.featureout', # Directory containing HDF5 feature files
                    device=torch.device('cpu'),
                    min_raw_matches_for_h5=15, # Min raw matches to save to matches.h5
                    ransac_reproj_threshold_cv=1.0, # RANSAC reprojection threshold for OpenCV's F-matrix
                    verbose_cv_errors=False,
                    quantize_ratio=1, logger=None): 

    pair_name_split = " "

    keypoints_h5_path = os.path.join(feature_dir, 'keypoints.h5')
    # descriptors_h5_path = os.path.join(feature_dir, 'descriptors.h5')
    matches_h5_path = os.path.join(feature_dir, 'matches.h5')
    logger.info(f"Saving DKM matches to {matches_h5_path} if computed.")
    num_images = len(img_fnames)
    similarity_matrix = np.zeros((num_images, num_images), dtype=np.int32)

    if not (os.path.exists(keypoints_h5_path)):
        logger.error(f"Feature files (keypoints.h5) not found in {feature_dir}. Cannot perform matching.")
        return

    import time
    for pair_idx_tuple in tqdm(index_pairs, desc="LoFTR Matching", leave=False):
        idx1, idx2 = pair_idx_tuple
        fname1_basename = os.path.basename(img_fnames[idx1])
        fname2_basename = os.path.basename(img_fnames[idx2])

        img_scale0 = read_grayscale(
            img_fnames[idx1],
            (1200,),
            df=8,
            pad_to=None,
            ret_scales=True,
        )
        img_scale1 = read_grayscale(
            img_fnames[idx2],
            (1200,),
            pad_to=None,
            df=8,
            ret_scales=True,
        )
        img0, scale0, original_hw0 = img_scale0
        img1, scale1, original_hw1 = img_scale1
        
        data = {
            "image0": img0[None,:],
            "image1": img1[None,:],
            "scale0": scale0[None,:],  # 1*2
            "scale1": scale1[None,:],
        }

        data_c = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }
        loftr_model(data_c)
        mkpts1, mkpts2, mconfs = extract_preds(data_c)
        try:            
            if mkpts1.shape[0] >= 10:
                try:
                    F_matrix, inlier_mask_cv = cv2.findFundamentalMat(mkpts1, mkpts2,
                                                                    cv2.FM_RANSAC,
                                                                    ransacReprojThreshold=ransac_reproj_threshold_cv,
                                                                    confidence=0.99) 
                    
                    if inlier_mask_cv is not None:
                        num_inliers = np.sum(inlier_mask_cv)
                        similarity_matrix[idx1, idx2] = num_inliers
                        similarity_matrix[idx2, idx1] = num_inliers 
                    else: 
                        similarity_matrix[idx1, idx2] = 0
                        similarity_matrix[idx2, idx1] = 0
                
                except Exception as e_verif:
                    if verbose_cv_errors:
                        logger.warning(f"Error during OpenCV verification for {fname1_basename}-{fname2_basename}: {e_verif}")
                    similarity_matrix[idx1, idx2] = 0
                    similarity_matrix[idx2, idx1] = 0

            else:
                similarity_matrix[idx1, idx2] = 0
                similarity_matrix[idx2, idx1] = 0

        except Exception as e:
            logger.error(f"Error matching {fname1_basename}-{fname2_basename}: {e}")
        
    print('LoFTR match done')
    torch.save(similarity_matrix, '/mnt/nas2/ksy/IMC2025/similarity_matrix_loftr.pt')
    return similarity_matrix



def match_with_loftr_quant(loftr_model, img_fnames,
                    index_pairs,               # List of (idx1, idx2) tuples for images to match
                    feature_dir='.featureout', # Directory containing HDF5 feature files
                    device=torch.device('cpu'),
                    min_raw_matches_for_h5=15, # Min raw matches to save to matches.h5
                    ransac_reproj_threshold_cv=1.0, # RANSAC reprojection threshold for OpenCV's F-matrix
                    verbose_cv_errors=False,
                    quantize_ratio=1, logger=None): 
    pair_name_split = " "
    loftr_model.to(device)

    keypoints_h5_path = os.path.join(feature_dir, 'keypoints.h5')
    # descriptors_h5_path = os.path.join(feature_dir, 'descriptors.h5')
    matches_h5_path = os.path.join(feature_dir, 'matches.h5')
    logger.info(f"Saving LoFTR matches to {matches_h5_path} if computed.")
    num_images = len(img_fnames)
    similarity_matrix = np.zeros((num_images, num_images), dtype=np.int32)

    if not (os.path.exists(keypoints_h5_path)):
        logger.error(f"Feature files (keypoints.h5) not found in {feature_dir}. Cannot perform matching.")
        return


    matches = {}
    for pair_idx_tuple in tqdm(index_pairs, desc="LoFTR Matching", leave=False):
        idx1, idx2 = pair_idx_tuple
        fname1_basename = os.path.basename(img_fnames[idx1])
        fname2_basename = os.path.basename(img_fnames[idx2])
        
        img_scale0 = read_grayscale(
            img_fnames[idx1],
            (1200,),
            df=8,
            pad_to=None,
            ret_scales=True,
        )
        img_scale1 = read_grayscale(
            img_fnames[idx2],
            (1200,),
            pad_to=None,
            df=8,
            ret_scales=True,
        )

        img0, scale0, original_hw0 = img_scale0
        img1, scale1, original_hw1 = img_scale1

        data = {
            "image0": img0[None,:],
            "image1": img1[None,:],
            "scale0": scale0[None,:],  # 1*2
            "scale1": scale1[None,:],
        }

        data_c = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }


        loftr_model(data_c)
        mkpts1, mkpts2, mconfs = extract_preds(data_c)

        try:            
            if mkpts1.shape[0] >= 10:
                try:
                    F_matrix, inlier_mask_cv = cv2.findFundamentalMat(mkpts1, mkpts2,
                                                                    cv2.FM_RANSAC,
                                                                    ransacReprojThreshold=ransac_reproj_threshold_cv,
                                                                    confidence=0.99) 
                    
                    if inlier_mask_cv is not None:
                        num_inliers = np.sum(inlier_mask_cv)
                        similarity_matrix[idx1, idx2] = num_inliers
                        similarity_matrix[idx2, idx1] = num_inliers 
                    else: 
                        similarity_matrix[idx1, idx2] = 0
                        similarity_matrix[idx2, idx1] = 0
                
                except Exception as e_verif:
                    if verbose_cv_errors:
                        logger.warning(f"Error during OpenCV verification for {fname1_basename}-{fname2_basename}: {e_verif}")
                    similarity_matrix[idx1, idx2] = 0
                    similarity_matrix[idx2, idx1] = 0

            else:
                similarity_matrix[idx1, idx2] = 0
                similarity_matrix[idx2, idx1] = 0

        except Exception as e:
            logger.error(f"Error matching {fname1_basename}-{fname2_basename}: {e}")

        matches[pair_name_split.join([img_fnames[idx1], img_fnames[idx2]])] = np.concatenate([mkpts1, mkpts2, \
            mconfs[:,None]], -1)

    loftr_keypoints, loftr_matches = merge_kpts(matches, img_fnames)

    num_alike_kpts = {}
    with h5py.File(keypoints_h5_path, mode='a') as f_kp, \
         h5py.File(matches_h5_path, mode='a') as f_match: 

        for idx_kpt in range(len(img_fnames)):
            # kpt add 
            fname_basename = os.path.basename(img_fnames[idx_kpt])
            kpt_info = np.array(f_kp[fname_basename]).copy()
            num_alike_kpts[fname_basename] = kpt_info.shape[0]

            del f_kp[fname_basename]

            f_kp[fname_basename] = np.vstack([kpt_info, loftr_keypoints[fname_basename]])
                        
        
        for pair_idx_tuple in tqdm(index_pairs, desc="Match fusion"):
            try:
                idx1, idx2 = pair_idx_tuple
                fname1_basename = os.path.basename(img_fnames[idx1])
                fname2_basename = os.path.basename(img_fnames[idx2])
                
                # Check if features for these specific images exist in the HDF5 files
                if fname1_basename not in f_kp or fname2_basename not in f_kp :
                    logger.warning(f"Features for pair {fname1_basename}-{fname2_basename} not found in HDF5 files. Skipping pair.")
                    similarity_matrix[idx1, idx2] = 0 
                    similarity_matrix[idx2, idx1] = 0
                    continue

                loftr_match_info = loftr_matches[' '.join([fname1_basename, fname2_basename])].transpose()
                if loftr_match_info.shape[0] >= 50:
                    group = f_match.require_group(fname1_basename)
                    loftr_match_info[:, 0] += num_alike_kpts[fname1_basename]
                    loftr_match_info[:, 1] += num_alike_kpts[fname2_basename]                            

                    if fname2_basename in group:
                        lightglue_match_info = np.array(group[fname2_basename]).copy()
                        if len(lightglue_match_info.shape) > 1:
                            combined_match_info = np.vstack([lightglue_match_info, loftr_match_info])
                        del group[fname2_basename]
                    else:
                        combined_match_info = loftr_match_info
                    
                    
                    group.create_dataset(fname2_basename, data=combined_match_info)    
            except:
                bb()

    print('LoFTR match done')
    torch.save(similarity_matrix, '/mnt/nas2/ksy/IMC2025/similarity_matrix_loftr.pt')
    return similarity_matrix