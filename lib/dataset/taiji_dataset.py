import numpy as np
import os
import bisect
from lib.utils.geometry_utils import *
from lib.dataset import BaseDataset

class TaijiDataset(BaseDataset):

    def __init__(self, cfg, estimator='vibe', return_type='3D', phase='test'):

        BaseDataset.__init__(self, cfg)

        self.dataset_name = "taiji"
        self.base_data_path = "./data"

        if phase == 'train':
            self.phase = phase  # 'train' | 'test'
        elif phase == 'test':
            self.phase = phase
        else:
            raise NotImplementedError(
                "Unknown phase type! Valid phase: [\'train\',\'test\']. You can edit the code for additional implements"
            )

        if return_type in ['3D', 'smpl']:  # no 2D
            self.return_type = return_type  # '3D' | '2D' | 'smpl'
        else:
            raise NotImplementedError(
                "Unknown return type! Valid phase: [\'3D\','smpl']. You can edit the code for additional implement"
            )

        if estimator in ['spin', 'eft', 'pare','vibe','tcmr']:
            self.estimator = estimator  # 'spin' | 'eft' | 'pare'
        else:
            raise NotImplementedError(
                "Unknown estimator! Valid phase: [\'spin\',\'eft\','pare']. You can edit the code for additional implement"
            )

        print('#############################################################')
        print('You are loading the [' + self.phase + 'ing set] of dataset [' +
              self.dataset_name + ']')
        print('You are using pose esimator [' + self.estimator + ']')
        print('The type of the data is [' + self.return_type + ']')

        self.slide_window_size = cfg.MODEL.SLIDE_WINDOW_SIZE
        self.evaluate_slide_window_step=cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE

        # taiji 데이터는 detected 폴더에 있음
        try:
            detected_data = np.load(os.path.join(
                "data/detected",
                "taiji_3D_test.npz"),
                                        allow_pickle=True)
        except:
            raise ImportError("Detected data do not exist!")

        # taiji 데이터는 ground truth가 없으므로 detected 데이터를 ground truth로 사용
        self.ground_truth_data_joints_3d = [detected_data["keypoints_3d"]]
        self.detected_data_joints_3d = [detected_data["keypoints_3d"]]
        
        # 이미지 이름은 프레임 번호로 생성
        num_frames = detected_data["keypoints_3d"].shape[0]
        self.ground_truth_data_imgname = [["frame_{:06d}.jpg".format(i) for i in range(num_frames)]]
        self.detected_data_imgname = [["frame_{:06d}.jpg".format(i) for i in range(num_frames)]]

        self.data_len = [len(seq)-self.slide_window_size if (len(seq)-self.slide_window_size)>0 else 0 for seq in self.ground_truth_data_imgname]
        self.data_start_num = [
                sum(self.data_len[0:i]) for i in range(len(self.data_len))
            ]
        if len(self.data_start_num) > 2:
            for i in range(len(self.data_start_num)-2,1):
                if self.data_start_num[i]==self.data_start_num[i-1]:
                    self.data_start_num[i]=self.data_start_num[i+1]

        self.frame_num = sum(self.data_len)
        print('The frame number is [' + str(self.frame_num) + ']')

        self.sequence_num = len(self.ground_truth_data_imgname)
        print('The sequence number is [' + str(self.sequence_num) + ']')

        print('#############################################################')

        if self.return_type == '3D':
            self.input_dimension = self.ground_truth_data_joints_3d[0].shape[-1] * self.ground_truth_data_joints_3d[0].shape[-2]

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
        position = bisect.bisect(self.data_start_num, index)-1

        ground_truth_data_len = len(self.ground_truth_data_imgname[position])
        detected_data_len = len(self.detected_data_imgname[position])

        if ground_truth_data_len != detected_data_len:
            raise ImportError(
                "Detected data is not the same size with ground_truth data!")

        if self.return_type == '3D':
            gt_data = self.ground_truth_data_joints_3d[position].reshape(
                ground_truth_data_len, -1, 3)
            pred_data = self.detected_data_joints_3d[position].reshape(
                ground_truth_data_len, -1, 3)

            gt_data = gt_data.reshape(ground_truth_data_len, -1)
            pred_data = pred_data.reshape(ground_truth_data_len, -1)

        if self.slide_window_size <= ground_truth_data_len:
            gt_data = np.concatenate(
                (gt_data, np.zeros(tuple((1, )) + tuple(gt_data.shape[1:]))),
                axis=0)
            pred_data = np.concatenate(
                (pred_data,
                 np.zeros(tuple((1, )) + tuple(pred_data.shape[1:]))),
                axis=0)

            start_idx = (index - self.data_start_num[position]) % (
                ground_truth_data_len - self.slide_window_size + 1)
            end_idx = start_idx + self.slide_window_size

            gt_data = gt_data[start_idx:end_idx, :]
            pred_data = pred_data[start_idx:end_idx, :]
        else:
            gt_data = np.concatenate((
                gt_data,
                np.zeros(
                    tuple((self.slide_window_size - ground_truth_data_len, )) +
                    tuple(gt_data.shape[1:]))),
                                     axis=0)
            pred_data = np.concatenate((
                pred_data,
                np.zeros(
                    tuple((self.slide_window_size - ground_truth_data_len, )) +
                    tuple(pred_data.shape[1:]))),
                                       axis=0)

        return {"gt": gt_data, "pred": pred_data}

    def get_test_data(self, index):
        ground_truth_data_len = len(self.ground_truth_data_imgname[index])
        detected_data_len = len(self.detected_data_imgname[index])

        if ground_truth_data_len != detected_data_len:
            raise ImportError(
                "Detected data is not the same size with ground_truth data!")

        if self.return_type == '3D':
            gt_data = self.ground_truth_data_joints_3d[index].reshape(
                ground_truth_data_len, -1, 3)
            pred_data = self.detected_data_joints_3d[index].reshape(
                detected_data_len, -1, 3)

            gt_data = gt_data.reshape(ground_truth_data_len, -1)
            pred_data = pred_data.reshape(detected_data_len, -1)

        return {"gt": gt_data, "pred": pred_data} 