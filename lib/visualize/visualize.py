import torch
from lib.utils.eval_metrics import *
from lib.utils.geometry_utils import *
import os
import cv2
import json
import numpy as np
from lib.visualize.visualize_3d import visualize_3d
from lib.visualize.visualize_smpl import visualize_smpl
from lib.visualize.visualize_2d import visualize_2d
import sys

H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
J17_TO_J14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

class Visualize():

    def __init__(self,test_dataset, cfg):

        self.cfg = cfg
        self.device = cfg.DEVICE

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.estimator = self.cfg.ESTIMATOR
        self.dataset_name = self.cfg.DATASET_NAME
        self.body_representation = self.cfg.BODY_REPRESENTATION

        self.vis_seq_index = self.cfg.VIS.INPUT_VIDEO_NUMBER
        self.vis_output_video_path = self.cfg.VIS.OUTPUT_VIDEO_PATH

        self.slide_window_size = self.cfg.MODEL.SLIDE_WINDOW_SIZE
        self.slide_window_step = self.cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE

        # Custom dataset의 경우 고정 경로 사용
        if self.dataset_name == "custom_hybrik_3D":
            self.base_data_path = "data/custom_hybrik_3D"
        else:
            self.base_data_path = self.cfg.DATASET.BASE_DIR

        self.phase="test"

        try:
            self.ground_truth_data = np.load(os.path.join(
                self.base_data_path,
                self.dataset_name+"_"+self.estimator+"_"+self.body_representation,
                "groundtruth",
                self.dataset_name + "_" + "gt"+"_"+self.body_representation + "_" + self.phase + ".npz"),
                                        allow_pickle=True)
            self.has_groundtruth = True
        except:
            print("Warning: Ground-truth data not found. Will visualize detected data only.")
            self.has_groundtruth = False

        try:
            self.detected_data = np.load(os.path.join(
                self.base_data_path, 
                self.dataset_name+"_"+self.estimator+"_"+self.body_representation,
                "detected",
                self.dataset_name + "_" + self.estimator+"_"+self.body_representation + "_" + self.phase + ".npz"),
                                        allow_pickle=True)
        except:
            # Fallback: try to load from custom path
            try:
                # 명령행 인자 또는 기본값 사용
                npz_filename = getattr(self.cfg.DATASET, 'NPZ_FILE', 'ohyeah_hybrik_3D_test.npz')
                if not npz_filename:  # 빈 문자열인 경우 기본값 사용
                    npz_filename = 'ohyeah_hybrik_3D_test.npz'
                self.detected_data = np.load(os.path.join(
                    self.base_data_path, npz_filename), allow_pickle=True)
                print(f"Loaded detected data from fallback path: {npz_filename}")
            except:
                raise ImportError("Detected data do not exist!")

        self.device = self.cfg.DEVICE

        if self.body_representation == '3D':
            if self.has_groundtruth:
                self.input_dimension = self.ground_truth_data["joints_3d"][0].shape[-1]
            else:
                self.input_dimension = self.detected_data["keypoints_3d"].shape[-1]
        elif self.body_representation == 'smpl':
            if cfg.TRAIN.USE_6D_SMPL:
                self.input_dimension = 6 * 24
            else:
                self.input_dimension = 3 * 24
        elif self.body_representation == '2D':
            if self.has_groundtruth:
                self.input_dimension = self.ground_truth_data["joints_2d"][0].shape[-1]
            else:
                self.input_dimension = self.detected_data["keypoints_2d"].shape[-1]

    def visualize_3d(self, model):
        keypoint_number =self.input_dimension//3
        try:
            keypoint_root = eval("self.cfg.DATASET." +
                                   "ROOT_"+self.cfg.DATASET_NAME.upper() +"_"+ self.cfg.ESTIMATOR.upper()+"_3D")
        except:
            # Fallback: use default root joint (usually pelvis at index 0)
            keypoint_root = 0
            print(f"Using default root joint index: {keypoint_root}")

        # Handle different key names and data structures
        if "joints_3d" in self.detected_data:
            if isinstance(self.detected_data["joints_3d"], list):
                data_pred = self.detected_data["joints_3d"][self.vis_seq_index].reshape(-1,keypoint_number,3)
            else:
                data_pred = self.detected_data["joints_3d"].reshape(-1,keypoint_number,3)
        elif "keypoints_3d" in self.detected_data:
            if isinstance(self.detected_data["keypoints_3d"], list):
                data_pred = self.detected_data["keypoints_3d"][self.vis_seq_index].reshape(-1,keypoint_number,3)
            else:
                data_pred = self.detected_data["keypoints_3d"].reshape(-1,keypoint_number,3)
        else:
            raise KeyError("Neither 'joints_3d' nor 'keypoints_3d' found in detected data")
        
        # Debug: print data shape
        print(f"data_pred shape after reshape: {data_pred.shape}")
        print(f"keypoint_number: {keypoint_number}")
        print(f"keypoint_root: {keypoint_root}")
        
        if self.has_groundtruth:
            if "joints_3d" in self.ground_truth_data:
                data_gt = self.ground_truth_data["joints_3d"][self.vis_seq_index].reshape(-1,keypoint_number,3)
            elif "keypoints_3d" in self.ground_truth_data:
                data_gt = self.ground_truth_data["keypoints_3d"].reshape(-1,keypoint_number,3)
            data_gt = data_gt - data_gt[:, keypoint_root, :].mean(axis=1).reshape(-1, 1, 3)
            data_gt = torch.tensor(data_gt).to(self.device)
        else:
            # Groundtruth가 없으면 detected 데이터를 groundtruth로 사용
            data_gt = data_pred.copy()
        
        # Root relative transformation
        root_pos = data_pred[:, keypoint_root, :]  # (1198, 3)
        data_pred = data_pred - root_pos[:, np.newaxis, :]  # (1198, 71, 3)
        data_pred = torch.tensor(data_pred).to(self.device)
        
        if self.estimator in ["eft","spin","pare"]:
            data_pred=data_pred[:, H36M_TO_J17, :][:, J17_TO_J14, :].contiguous()
            data_gt=data_gt[:, H36M_TO_J17, :][:, J17_TO_J14, :].contiguous()
            keypoint_number=data_gt.shape[1]
            self.input_dimension=keypoint_number*3

        data_len = data_pred.shape[0]
        data_pred_window = torch.as_strided(
            data_pred, ((data_len - self.slide_window_size) // self.slide_window_step+1,
                        self.slide_window_size, keypoint_number, 3),
            (self.slide_window_step * keypoint_number * 3,
             keypoint_number * 3, 3, 1),
            storage_offset=0).reshape(-1, self.slide_window_size,
                                      self.input_dimension)

        with torch.no_grad():
            data_pred_window=data_pred_window.permute(0,2,1)
            predicted_pos= model(data_pred_window)
            data_pred_window=data_pred_window.permute(0,2,1)
            predicted_pos=predicted_pos.permute(0,2,1)

        predicted_pos = slide_window_to_sequence(predicted_pos,self.slide_window_step,self.slide_window_size).reshape(-1, keypoint_number, 3)

        data_len = predicted_pos.shape[0]
        data_pred = data_pred[:data_len, :].reshape(-1, keypoint_number, 3)
        data_gt = data_gt[:data_len, :].reshape(-1, keypoint_number, 3)

        if self.dataset_name in ["aist","h36m","mpiinf3dhp","mupots","pw3d","custom_hybrik_3D"]:
            data_gt = data_gt.reshape(-1, keypoint_number, 3)
            data_pred = data_pred.reshape(-1, keypoint_number, 3)
            predicted_pos = predicted_pos.reshape(-1, keypoint_number, 3)

            vis_output_video_name = self.dataset_name+"_"+self.estimator+"_3D_" + str(
                self.vis_seq_index) +"_frame_" +str(self.cfg.VIS.START)+"-"+str(self.cfg.VIS.END)+".mp4"

            # Save smoothed keypoints to JSON
            self.save_smoothed_keypoints_to_json(predicted_pos, data_pred, data_gt)
            
            print("✅ 지터 제거 완료! JSON 파일이 생성되었습니다.")
            print(f"📁 저장 위치: {os.path.join(self.vis_output_video_path, 'json_output')}")
            
            # 시각화는 건너뛰고 JSON 파일만 생성
            # visualize_3d(
            #     self.vis_output_video_path,
            #     vis_output_video_name,
            #     data_pred,
            #     data_gt,
            #     predicted_pos,
            #     self.cfg.VIS.START,
            #     self.cfg.VIS.END,
            #     self.dataset_name,
            #     self.estimator
            # )
        else:
            print("Not Implemented!")


    def visualize_smpl(self, model):

        data_gt = self.ground_truth_data["pose"][self.vis_seq_index]
        data_pred = self.detected_data["pose"][self.vis_seq_index]

        if self.cfg.TRAIN.USE_6D_SMPL:
            data_pred = numpy_axis_to_rot6D(data_pred.reshape(-1, 3)).reshape(
                -1, self.input_dimension)

        data_imgname = self.ground_truth_data["imgname"][self.vis_seq_index]

        data_gt = torch.tensor(data_gt).to(self.device)
        data_pred = torch.tensor(data_pred).to(self.device)

        data_len = data_pred.shape[0]
        data_pred_window = torch.as_strided(
            data_pred, ((data_len - self.slide_window_size) // self.slide_window_step+1,
                        self.slide_window_size, self.input_dimension),
            (self.slide_window_step * self.input_dimension,
             self.input_dimension, 1),
            storage_offset=0).reshape(-1, self.slide_window_size,
                                      self.input_dimension)

        with torch.no_grad():
            data_pred_window=data_pred_window.permute(0,2,1)
            predicted_pos= model(data_pred_window)
            data_pred_window=data_pred_window.permute(0,2,1)
            predicted_pos=predicted_pos.permute(0,2,1)

        predicted_pos = slide_window_to_sequence(predicted_pos,self.slide_window_step,self.slide_window_size).reshape(-1, self.input_dimension)

        data_len = predicted_pos.shape[0]
        data_pred = data_pred[:data_len, :]
        data_gt = data_gt[:data_len, :]

        data_imgname = data_imgname[:data_len]

        if self.cfg.TRAIN.USE_6D_SMPL:
            data_pred = rot6D_to_axis(data_pred.reshape(-1, 6)).reshape(
                -1, 24 * 3)
            predicted_pos = rot6D_to_axis(predicted_pos.reshape(-1,
                                                                6)).reshape(
                                                                    -1, 24 * 3)

        data_gt = np.array(data_gt.reshape(-1, 24 * 3).cpu())
        data_pred = np.array(data_pred.reshape(-1, 24 * 3).cpu())
        predicted_pos = np.array(predicted_pos.reshape(-1, 24 * 3).cpu())

        smpl_neural = SMPL(model_path=self.cfg.SMPL_MODEL_DIR,
                           create_transl=False)

        if self.dataset_name in ["aist","h36m","pw3d"]:
            vis_output_video_name = self.dataset_name+"_"+self.estimator+"_SMPL_" + str(
                self.vis_seq_index) +"_frame_" +str(self.cfg.VIS.START)+"-"+str(self.cfg.VIS.END)+ ".mp4"

            visualize_smpl(
                self.vis_output_video_path,
                vis_output_video_name,
                smpl_neural,
                data_pred,
                data_gt,
                predicted_pos,
                self.cfg.VIS.START,
                self.cfg.VIS.END,
            )

    def visualize_2d(self, model):
        keypoint_number =self.input_dimension//2

        data_gt = self.ground_truth_data["joints_2d"][self.vis_seq_index]
        data_pred = self.detected_data["joints_2d"][self.vis_seq_index]

        if self.dataset_name=="jhmdb":
            data_imageshape=self.ground_truth_data["imgshape"][self.vis_seq_index][:2][::-1].copy()
        elif self.dataset_name=="h36m":
            data_imageshape=1000

        data_gt = torch.tensor(data_gt).to(self.device)
        len_seq = data_gt.shape[0]
        data_pred=data_pred[:len_seq,:]
        if self.dataset_name=="jhmdb":
            data_pred_norm=torch.tensor(data_pred.reshape(-1,2)/data_imageshape-0.5).to(self.device).reshape_as(data_gt)
        elif self.dataset_name=="h36m":
            data_pred_norm =(torch.tensor(data_pred).to(self.device)-data_imageshape/2)/(data_imageshape/2) # normalization

        data_pred = torch.tensor(data_pred).to(self.device)

        data_len = data_pred.shape[0]
        data_pred_window = torch.as_strided(
            data_pred_norm, ((data_len - self.slide_window_size) // self.slide_window_step+1,
                        self.slide_window_size, keypoint_number, 2),
            (self.slide_window_step * keypoint_number * 2,
             keypoint_number * 2, 2, 1),
            storage_offset=0).reshape(-1, self.slide_window_size,
                                      self.input_dimension)

        with torch.no_grad():
            data_pred_window=data_pred_window.permute(0,2,1)
            predicted_pos = model(data_pred_window)
            data_pred_window=data_pred_window.permute(0,2,1)
            predicted_pos=predicted_pos.permute(0,2,1)

        predicted_pos = slide_window_to_sequence(predicted_pos,self.slide_window_step,self.slide_window_size).reshape(-1, keypoint_number, 2)

        data_len = predicted_pos.shape[0]
        data_pred = data_pred[:data_len, :].reshape(-1, keypoint_number, 2)
        data_gt = data_gt[:data_len, :].reshape(-1, keypoint_number, 2)
        if self.dataset_name=="jhmdb":
            predicted_pos = (predicted_pos.reshape(-1, keypoint_number, 2)+0.5)*torch.tensor(data_imageshape).to(predicted_pos.device)
        elif self.dataset_name=="h36m":
            predicted_pos = predicted_pos.reshape(-1, keypoint_number, 2)*torch.tensor(data_imageshape/2).to(predicted_pos.device)+torch.tensor(data_imageshape/2)

        if self.dataset_name in ["jhmdb","h36m"]:
            vis_output_video_name = self.dataset_name+"_"+self.estimator+"_2D_" + str(
                self.vis_seq_index) +"_frame_" +str(self.cfg.VIS.START)+"-"+str(self.cfg.VIS.END)+ ".mp4"

            visualize_2d(
                self.vis_output_video_path,
                vis_output_video_name,
                predicted_pos,
                data_pred,
                data_gt,
                self.cfg.VIS.START,
                self.cfg.VIS.END,
                self.dataset_name,
                self.estimator
            )
        else:
            print("Not Implemented!")

    def visualize(self, model):
        model.eval()
        if self.cfg.BODY_REPRESENTATION == "3D":
            self.visualize_3d(model)

        elif self.cfg.BODY_REPRESENTATION == "smpl":
            self.visualize_smpl(model)

        elif self.cfg.BODY_REPRESENTATION == "2D":
            self.visualize_2d(model)

    def save_smoothed_keypoints_to_json(self, predicted_pos, data_pred, data_gt):
        """
        Save smoothed keypoints to JSON format
        """
        # Convert tensors to numpy arrays
        if torch.is_tensor(predicted_pos):
            predicted_pos = predicted_pos.cpu().numpy()
        if torch.is_tensor(data_pred):
            data_pred = data_pred.cpu().numpy()
        if torch.is_tensor(data_gt):
            data_gt = data_gt.cpu().numpy()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(self.vis_output_video_path, "json_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for JSON
        num_frames = predicted_pos.shape[0]
        num_joints = predicted_pos.shape[1]
        
        # Create joint names (you can customize these based on your dataset)
        joint_names = [f"joint_{i}" for i in range(num_joints)]
        
        # Create the JSON structure
        json_data = {
            "metadata": {
                "dataset_name": self.dataset_name,
                "estimator": self.estimator,
                "body_representation": self.body_representation,
                "num_frames": num_frames,
                "num_joints": num_joints,
                "joint_names": joint_names,
                "processing_info": {
                    "smoothnet_window_size": self.slide_window_size,
                    "smoothnet_step_size": self.slide_window_step
                }
            },
            "frames": []
        }
        
        # Add frame data
        for frame_idx in range(num_frames):
            frame_data = {
                "frame_id": frame_idx,
                "keypoints": {
                    "original": data_pred[frame_idx].tolist(),
                    "smoothed": predicted_pos[frame_idx].tolist(),
                    "ground_truth": data_gt[frame_idx].tolist() if self.has_groundtruth else None
                }
            }
            json_data["frames"].append(frame_data)
        
        # Save to JSON file
        output_filename = f"{self.dataset_name}_{self.estimator}_smoothed_keypoints.json"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Smoothed keypoints saved to: {output_path}")
        
        # Also save a simplified version with only smoothed keypoints
        simplified_data = {
            "metadata": {
                "dataset_name": self.dataset_name,
                "estimator": self.estimator,
                "num_frames": num_frames,
                "num_joints": num_joints
            },
            "smoothed_keypoints": predicted_pos.tolist()
        }
        
        simplified_filename = f"{self.dataset_name}_{self.estimator}_smoothed_only.json"
        simplified_path = os.path.join(output_dir, simplified_filename)
        
        with open(simplified_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_data, f, indent=2, ensure_ascii=False)
        
        print(f"Simplified smoothed keypoints saved to: {simplified_path}")
