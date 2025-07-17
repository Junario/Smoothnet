import json
import numpy as np

def json_to_npz(json_path, npz_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    keypoints_3d_list = [frame['keypoints_3d'] for frame in data['frames']]
    keypoints_3d = np.array(keypoints_3d_list)  # [num_frames, num_joints, 3]
    np.savez(npz_path, keypoints_3d=keypoints_3d)
    print(f"NPZ 파일 저장: {npz_path}")

# 사용 예시
json_path = "converted_taiji_24_joints_dance_mapping.json"  # JSON 파일 경로
npz_path = "data/detected/taiji_3D_test.npz"  # README 구조에 맞춤
json_to_npz(json_path, npz_path) 