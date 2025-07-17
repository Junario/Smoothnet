from lib.dataset import BaseDataset
import numpy as np
import os
import bisect
from lib.utils.geometry_utils import *


class CustomHybrik3DDataset(BaseDataset):

    def __init__(self, cfg, estimator='hybrik', return_type='3D', phase='test'):

        BaseDataset.__init__(self, cfg)

        self.dataset_name = "custom_hybrik_3D"

        if phase == 'train':
            self.phase = phase  # 'train' | 'test' | 'validate'
        elif phase == 'test':
            self.phase = phase
        elif phase == 'validate':
            self.phase = phase
        else:
            raise NotImplementedError(
                "Unknown phase type! Valid phase: [\'train\',\'test\',\'validate\']. You can edit the code for additional implements"
            )

        if return_type in ['3D', 'smpl']:  # no 2D
            self.return_type = return_type  # '3D'
        else:
            raise NotImplementedError(
                "Unknown return type! Valid phase: [\'3D\','smpl']. You can edit the code for additional implement"
            )

        if estimator in ['hybrik']:
            self.estimator = estimator  # 'hybrik'
        else:
            raise NotImplementedError(
                "Unknown estimator! Valid phase: [\'hybrik\']. You can edit the code for additional implement"
            )

        print('#############################################################')
        print('You are loading the [' + self.phase + 'ing set] of dataset [' +
              self.dataset_name + ']')
        print('You are using pose esimator [' + str(self.estimator) + ']')
        print('The type of the data is [' + str(self.return_type) + ']')

        self.slide_window_size = cfg.MODEL.SLIDE_WINDOW_SIZE
        self.evaluate_slide_window_step = cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE

        # HybrIK에서 생성한 NPZ 파일 경로
        self.base_data_path = 'data/custom_hybrik_3D'  # 고정 경로 사용
        
        # NPZ 파일 경로 (명령행 인자 또는 기본값 사용)
        npz_filename = getattr(cfg.DATASET, 'NPZ_FILE', 'ohyeah_hybrik_3D_test.npz')
        if not npz_filename:  # 빈 문자열인 경우 기본값 사용
            npz_filename = 'ohyeah_hybrik_3D_test.npz'
        npz_file_path = os.path.join(self.base_data_path, npz_filename)
        
        if not os.path.exists(npz_file_path):
            raise ImportError(f"NPZ file not found at {npz_file_path}")

        try:
            # HybrIK에서 생성한 NPZ 파일 로드
            hybrik_data = np.load(npz_file_path, allow_pickle=True)
            print(f"Loaded NPZ file: {npz_file_path}")
            print(f"Available keys: {list(hybrik_data.keys())}")
        except Exception as e:
            raise ImportError(f"Failed to load NPZ file: {e}")

        # HybrIK 데이터 구조 확인
        if 'keypoints_3d' not in hybrik_data:
            raise ImportError("NPZ file does not contain 'keypoints_3d' key")
        
        if 'imgname' not in hybrik_data:
            raise ImportError("NPZ file does not contain 'imgname' key")

        # 데이터 준비
        self.keypoints_3d = hybrik_data['keypoints_3d']  # (frames, num_keypoints, 3)
        self.imgname = hybrik_data['imgname']  # (frames,)
        
        # SMPL 파라미터가 있는 경우
        self.has_smpl = 'pose' in hybrik_data and 'shape' in hybrik_data
        if self.has_smpl:
            self.pose = hybrik_data['pose']  # (frames, pose_dim)
            self.shape = hybrik_data['shape']  # (frames, shape_dim)
            print(f"SMPL pose shape: {self.pose.shape}")
            print(f"SMPL shape shape: {self.shape.shape}")

        # 데이터 형태 확인 및 조정
        if len(self.keypoints_3d.shape) == 3:
            # (frames, num_keypoints, 3) -> (frames, num_keypoints * 3)
            self.keypoints_3d = self.keypoints_3d.reshape(self.keypoints_3d.shape[0], -1)
        
        self.frame_num = self.keypoints_3d.shape[0]
        self.sequence_num = 1  # 단일 시퀀스
        
        print(f'Keypoints shape: {self.keypoints_3d.shape}')
        print(f'The frame number is [{self.frame_num}]')
        print(f'The sequence number is [{self.sequence_num}]')
        print('#############################################################')

        # 입력 차원 설정
        if self.return_type == '3D':
            self.input_dimension = self.keypoints_3d.shape[1]
        elif self.return_type == 'smpl' and self.has_smpl:
            if cfg.TRAIN.USE_6D_SMPL:
                self.input_dimension = self.pose.shape[1]  # 이미 6D로 변환되어 있을 것으로 가정
            else:
                self.input_dimension = self.pose.shape[1]
        else:
            raise ValueError(f"Invalid return_type: {self.return_type} or SMPL data not available")

        # 슬라이딩 윈도우를 위한 데이터 길이 계산
        self.data_len = [max(0, self.frame_num - self.slide_window_size)]
        self.data_start_num = [0]

    def __len__(self):
        if self.phase == "train":
            return self.frame_num
        elif self.phase == "test":
            return self.sequence_num

    def __getitem__(self, index):
        if self.phase == "train":
            return self.get_data(index)
        elif self.phase == "test":
            return self.get_test_data(index)

    def get_data(self, index):
        """훈련용 데이터 반환 (슬라이딩 윈도우 적용)"""
        if self.return_type == '3D':
            # 3D 키포인트 데이터
            pred_data = self.keypoints_3d.copy()
            
            # Ground truth는 없으므로 detected data를 그대로 사용
            gt_data = pred_data.copy()
            
        elif self.return_type == 'smpl' and self.has_smpl:
            # SMPL 파라미터 데이터
            pred_data = self.pose.copy()
            gt_data = pred_data.copy()  # Ground truth 없음
            
        else:
            raise ValueError(f"Invalid return_type: {self.return_type}")

        # 슬라이딩 윈도우 적용
        if self.slide_window_size <= self.frame_num:
            # 패딩 추가
            gt_data = np.concatenate(
                (gt_data, np.zeros((1, gt_data.shape[1]))),
                axis=0)
            pred_data = np.concatenate(
                (pred_data, np.zeros((1, pred_data.shape[1]))),
                axis=0)

            # 슬라이딩 윈도우 시작 인덱스 계산
            start_idx = index % (self.frame_num - self.slide_window_size + 1)
            end_idx = start_idx + self.slide_window_size

            gt_data = gt_data[start_idx:end_idx, :]
            pred_data = pred_data[start_idx:end_idx, :]
        else:
            # 프레임 수가 슬라이딩 윈도우보다 작은 경우 패딩
            gt_data = np.concatenate((
                gt_data,
                np.zeros((self.slide_window_size - self.frame_num, gt_data.shape[1]))),
                axis=0)
            pred_data = np.concatenate((
                pred_data,
                np.zeros((self.slide_window_size - self.frame_num, pred_data.shape[1]))),
                axis=0)

        return {"gt": gt_data, "pred": pred_data}

    def get_test_data(self, index):
        """테스트용 데이터 반환 (슬라이딩 윈도우 적용)"""
        if self.return_type == '3D':
            # 3D 키포인트 데이터
            pred_data = self.keypoints_3d.copy()
            gt_data = pred_data.copy()  # Ground truth 없음
            
        elif self.return_type == 'smpl' and self.has_smpl:
            # SMPL 파라미터 데이터
            pred_data = self.pose.copy()
            gt_data = pred_data.copy()  # Ground truth 없음
            
        else:
            raise ValueError(f"Invalid return_type: {self.return_type}")

        # 슬라이딩 윈도우 적용 (테스트에서도 동일하게)
        if self.slide_window_size <= self.frame_num:
            # 패딩 추가
            gt_data = np.concatenate(
                (gt_data, np.zeros((1, gt_data.shape[1]))),
                axis=0)
            pred_data = np.concatenate(
                (pred_data, np.zeros((1, pred_data.shape[1]))),
                axis=0)

            # 슬라이딩 윈도우 시작 인덱스 계산
            start_idx = index % (self.frame_num - self.slide_window_size + 1)
            end_idx = start_idx + self.slide_window_size

            gt_data = gt_data[start_idx:end_idx, :]
            pred_data = pred_data[start_idx:end_idx, :]
        else:
            # 프레임 수가 슬라이딩 윈도우보다 작은 경우 패딩
            gt_data = np.concatenate((
                gt_data,
                np.zeros((self.slide_window_size - self.frame_num, gt_data.shape[1]))),
                axis=0)
            pred_data = np.concatenate((
                pred_data,
                np.zeros((self.slide_window_size - self.frame_num, pred_data.shape[1]))),
                axis=0)

        # 배치 차원 추가: (time, features) -> (1, time, features)
        pred_data = pred_data[np.newaxis, :, :]
        gt_data = gt_data[np.newaxis, :, :]

        return {"gt": gt_data, "pred": pred_data} 