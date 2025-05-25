import numpy as np
from loguru import logger
from typing import ChainMap
import os
import os.path as osp
import math
from pdb import set_trace as bb
from collections import defaultdict

pair_name_split = " "

def split_dict(_dict, n):
    for _items in chunks(list(_dict.items()), n):
        yield dict(_items)

def chunks(lst, n, length=None):
    """Yield successive n-sized chunks from lst."""
    try:
        _len = len(lst)
    except TypeError as _:
        assert length is not None
        _len = length

    for i in range(0, _len, n):
        yield lst[i : i + n]
    # TODO: Check that lst is fully iterated

def chunk_index(total_len, sub_len, shuffle=True):
    index_array = np.arange(total_len)
    if shuffle:
        random.shuffle(index_array)

    index_list = []
    for i in range(0, total_len, sub_len):
        index_list.append(list(index_array[i : i + sub_len]))
    
    return index_list

def agg_groupby_2d(keys, vals, agg='avg'):
    """
    Args:
        keys: (N, 2) 2d keys
        vals: (N,) values to average over
        agg: aggregation method
    Returns:
        dict: {key: agg_val}
    """
    assert agg in ['avg', 'sum']
    unique_keys, group, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    group_sums = np.bincount(group, weights=vals)
    values = group_sums if agg == 'sum' else group_sums / counts
    return dict(zip(map(tuple, unique_keys), values))


def transform_keypoints(keypoints, pba=None, verbose=True):
    """assume keypoints sorted w.r.t. score"""
    ret_kpts = {}
    ret_scores = {}

    if verbose:
        keypoints_items = tqdm(keypoints.items()) if pba is None else keypoints.items()
    else:
        assert pba is None
        keypoints_items = keypoints.items()

    for k, v in keypoints_items:
        v = {_k: _v for _k, _v in v.items() if len(_k) == 2}
        kpts = np.array([list(kpt) for kpt in v.keys()]).astype(np.float32)
        scores = np.array([s[-1] for s in v.values()]).astype(np.float32)
        if len(kpts) == 0:
            logger.warning("corner-case n_kpts=0 exists!")
            kpts = np.empty((0,2))
        ret_kpts[k] = kpts
        ret_scores[k] = scores
        if pba is not None:
            pba.update.remote(1)
    return ret_kpts, ret_scores

class Match2Kpts(object):
    """extract all possible keypoints for each image from all image-pair matches"""
    def __init__(self, matches, names, name_split='-', cov_threshold=0):
        self.names = names
        self.matches = matches
        self.cov_threshold = cov_threshold
        self.name2matches = {name: [] for name in names}
        for k in matches.keys():
            try:
                name0, name1 = k.split(name_split)
            except ValueError as _:
                name0, name1 = k.split('-')
            self.name2matches[name0].append((k, 0))
            self.name2matches[name1].append((k, 1))
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            name = self.names[idx]
            kpts = np.concatenate([self.matches[k][:, [2*id, 2*id+1, 4]]
                        for k, id in self.name2matches[name] if self.matches[k].shape[0] >= self.cov_threshold], 0)
            return name, kpts
        elif isinstance(idx, slice):
            names = self.names[idx]
            try:
                kpts = [np.concatenate([self.matches[k][:, [2*id, 2*id+1, 4]]
                            for k, id in self.name2matches[name] if self.matches[k].shape[0] >= self.cov_threshold], 0) for name in names]
            except:
                kpts = []
                for name in names:
                    kpt = [self.matches[k][:, [2*id, 2*id+1, 4]]
                            for k, id in self.name2matches[name] if self.matches[k].shape[0] >= self.cov_threshold]
                    if len(kpt) != 0:
                        kpts.append(np.concatenate(kpt,0))
                    else:
                        kpts.append(np.empty((0,3)))
                        logger.warning(f"no keypoints in image:{name}")
            return list(zip(names, kpts))
        else:
            raise TypeError(f'{type(self).__name__} indices must be integers')


def keypoint_worker(name_kpts, pba=None, verbose=True):
    """merge keypoints associated with one image.
    """
    keypoints = {}

    if verbose:
        name_kpts = tqdm(name_kpts) if pba is None else name_kpts
    else:
        assert pba is None

    for name, kpts in name_kpts:
        kpt2score = agg_groupby_2d(kpts[:, :2].astype(int), kpts[:, -1], agg="sum")
        kpt2id_score = {
            k: (i, v)
            for i, (k, v) in enumerate(
                sorted(kpt2score.items(), key=lambda kv: kv[1], reverse=True)
            )
        }
        keypoints[name] = kpt2id_score

        if pba is not None:
            pba.update.remote(1)
    return keypoints


def update_matches(matches, keypoints, merge=False, pba=None, verbose=True, **kwargs):
    # convert match to indices
    ret_matches = {}

    if verbose:
        matches_items = tqdm(matches.items()) if pba is None else matches.items()
    else:
        assert pba is None
        matches_items = matches.items()

    for k, v in matches_items:
        mkpts0, mkpts1 = (
            map(tuple, v[:, :2].astype(int)),
            map(tuple, v[:, 2:4].astype(int)),
        )
        name0, name1 = k.split(pair_name_split)
        _kpts0, _kpts1 = keypoints[name0], keypoints[name1]

        mids = np.array(
            [
                [_kpts0[p0][0], _kpts1[p1][0]]
                for p0, p1 in zip(mkpts0, mkpts1)
                if p0 in _kpts0 and p1 in _kpts1
            ]
        )

        if len(mids) == 0:
            mids = np.empty((0, 2))

        def _merge_possible(name):  # only merge after dynamic nms (for now)
            return f'{name}_no-merge' not in keypoints

        if merge and _merge_possible(name0) and _merge_possible(name1):
            merge_ids = []
            mkpts0, mkpts1 = map(tuple, v[:, :2].astype(int)), map(tuple,  v[:, 2:4].astype(int))
            for p0, p1 in zip(mkpts0, mkpts1): 
                if (*p0, -2) in _kpts0 and (*p1, -2) in _kpts1:
                    merge_ids.append([_kpts0[(*p0, -2)][0], _kpts1[(*p1, -2)][0]])
                elif p0 in _kpts0 and (*p1, -2) in _kpts1:
                    merge_ids.append([_kpts0[p0][0], _kpts1[(*p1, -2)][0]])
                elif (*p0, -2) in _kpts0 and p1 in _kpts1:
                    merge_ids.append([_kpts0[(*p0, -2)][0], _kpts1[p1][0]]) 
            merge_ids = np.array(merge_ids)

            if len(merge_ids) == 0:
                merge_ids = np.empty((0, 2))
            try:
                mids_multiview = np.concatenate([mids, merge_ids], axis=0)
            except ValueError:
                import ipdb; ipdb.set_trace()
        
            mids = np.unique(mids_multiview, axis=0)
        else:
            assert (
                len(mids) == v.shape[0]
            ), f"len mids: {len(mids)}, num matches: {v.shape[0]}"

        ret_matches[k] = mids.astype(int)  # (N,2)
        if pba is not None:
            pba.update.remote(1)

    return ret_matches


def refine_matches_bidirectional(kpts_a, kpts_b, est_F):
    """
    양방향으로 중복된 매칭을 제거하고 Fundamental Matrix 기반으로 신뢰도 높은 매칭만 유지
    
    Args:
        kpts_a: 이미지 A의 키포인트 좌표 (N,2) 배열
        kpts_b: 이미지 B의 키포인트 좌표 (N,2) 배열 - kpts_a와 매칭됨
        
    Returns:
        refined_kpts_a, refined_kpts_b: 정제된 매칭 좌표 쌍
    """
    assert kpts_a.shape == kpts_b.shape, "키포인트 배열 크기가 일치해야 함"
    assert kpts_a.shape[1] == 2, "키포인트는 (x,y) 좌표쌍이어야 함"
    
    
    if est_F is None or est_F.shape != (3, 3):
        return kpts_a, kpts_b  # 원본 반환
    
    # 2. 매칭 인덱스 배열 초기화
    indices = np.arange(len(kpts_a))
    
    # 3. 이미지 A의 중복 키포인트 처리
    matches_by_a = defaultdict(list)
    for i in range(len(kpts_a)):
        a_key = tuple(kpts_a[i])
        matches_by_a[a_key].append(i)
    
    # 4. 이미지 B의 중복 키포인트 처리
    matches_by_b = defaultdict(list)
    for i in range(len(kpts_b)):
        b_key = tuple(kpts_b[i])
        matches_by_b[b_key].append(i)
    
    # 5. 양방향 중복 필터링
    filtered_indices = set(range(len(kpts_a)))  # 시작점: 모든 인덱스 포함
    
    # 먼저 이미지 A 중복 처리
    for a_key, idx_list in matches_by_a.items():
        if len(idx_list) <= 1:
            continue  # 중복 없음
            
        # 중복된 A 키포인트에 대해 에피폴라 에러가 가장 작은 매칭 찾기
        best_idx = None
        min_error = float('inf')
        
        a_homogeneous = np.array([a_key[0], a_key[1], 1.0]).reshape(3, 1)
        
        for idx in idx_list:
            b_point = kpts_b[idx]
            b_homogeneous = np.array([b_point[0], b_point[1], 1.0]).reshape(3, 1)
            
            # 에피폴라 제약 에러 계산
            error = np.abs(np.dot(b_homogeneous.T, np.dot(est_F, a_homogeneous)))[0, 0]
            
            if error < min_error:
                min_error = error
                best_idx = idx
        
        # 최상의 매칭만 유지하고 나머지는 제거
        for idx in idx_list:
            if idx != best_idx:
                filtered_indices.discard(idx)
    
    # 다음으로 이미지 B 중복 처리 (이미 필터링된 인덱스에서만 처리)
    remaining_indices = list(filtered_indices)
    filtered_indices_after_b = set()
    
    # B 키포인트별로 그룹화 (이미 필터링된 매칭만 고려)
    matches_by_b_filtered = defaultdict(list)
    for idx in remaining_indices:
        b_key = tuple(kpts_b[idx])
        matches_by_b_filtered[b_key].append(idx)
    
    # 각 B 중복 그룹에서 최상의 매칭 선택
    for b_key, idx_list in matches_by_b_filtered.items():
        if len(idx_list) <= 1:
            filtered_indices_after_b.add(idx_list[0])  # 중복 없음
            continue
            
        # 중복된 B 키포인트에 대해 에피폴라 에러가 가장 작은 매칭 찾기
        best_idx = None
        min_error = float('inf')
        
        b_homogeneous = np.array([b_key[0], b_key[1], 1.0]).reshape(3, 1)
        
        for idx in idx_list:
            a_point = kpts_a[idx]
            a_homogeneous = np.array([a_point[0], a_point[1], 1.0]).reshape(3, 1)
            
            # 에피폴라 제약 에러 계산 (역방향)
            error = np.abs(np.dot(a_homogeneous.T, np.dot(est_F.T, b_homogeneous)))[0, 0]
            
            if error < min_error:
                min_error = error
                best_idx = idx
        
        # 최상의 매칭만 유지
        filtered_indices_after_b.add(best_idx)
    
    # 최종 인덱스 목록
    final_indices = list(filtered_indices_after_b)

    # 정제된 키포인트 배열 생성
    refined_kpts_a = kpts_a[final_indices]
    refined_kpts_b = kpts_b[final_indices]
    
    # print(f"Number of changed: {len(kpts_a)} -> {len(refined_kpts_a)}")
    return refined_kpts_a, refined_kpts_b


def merge_kpts(matches, image_lists, verbose=False):        
    n_imgs = len(image_lists)
    logger.info("Combine keypoints!")
    all_kpts = Match2Kpts(
        matches, image_lists, name_split=pair_name_split
    )
    sub_kpts = chunks(all_kpts, math.ceil(n_imgs / 1))  # equal to only 1 worker
    obj_refs = [keypoint_worker(sub_kpt, verbose=verbose) for sub_kpt in sub_kpts]
    keypoints = dict(ChainMap(*obj_refs))

    # Convert keypoints match to keypoints indexs
    logger.info("Update matches")
    obj_refs = [
        update_matches(
            sub_matches,
            keypoints,
            merge=False,
            verbose=verbose,
            pair_name_split=pair_name_split,
        )
        for sub_matches in split_dict(matches, math.ceil(len(matches) / 1))
    ]
    updated_matches = dict(ChainMap(*obj_refs))

    # Post process keypoints:
    keypoints = {
        k: v for k, v in keypoints.items() if isinstance(v, dict)
    }
    logger.info("Post-processing keypoints...")
    kpts_scores = [
        transform_keypoints(sub_kpts, verbose=verbose)
        for sub_kpts in split_dict(keypoints, math.ceil(len(keypoints) / 1))
    ]
    final_keypoints = dict(ChainMap(*[k for k, _ in kpts_scores]))
    final_scores = dict(ChainMap(*[s for _, s in kpts_scores]))

    # Reformat keypoints_dict and matches_dict
    # from (abs_img_path0 abs_img_path1) -> (img_name0, img_name1)
    keypoints_renamed = {}
    for key, value in final_keypoints.items():
        keypoints_renamed[osp.basename(key)] = value

    matches_renamed = {}
    for key, value in updated_matches.items():
        name0, name1 = key.split(pair_name_split)
        new_pair_name = pair_name_split.join(
            [osp.basename(name0), osp.basename(name1)]
        )
        matches_renamed[new_pair_name] = value.T

    return keypoints_renamed, matches_renamed