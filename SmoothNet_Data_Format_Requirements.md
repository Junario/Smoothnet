# SmoothNet 데이터 형식 요구사항 및 변환 과정

## 개요
SmoothNet은 3D 포즈 추정 결과의 지터를 제거하는 네트워크입니다. 이 문서는 다양한 입력 데이터를 SmoothNet이 처리할 수 있는 형식으로 변환하는 과정을 설명합니다.

##1. 기본 데이터 형식 요구사항

### 1.1PZ 파일 구조
SmoothNet은 `.npz` 파일을 입력으로 받으며, 다음 키들을 포함해야 합니다:

```python
# 필수 키들
[object Object]
    keypoints_3d: np.array,  # 또는joints_3d
    keypoints_2d: np.array,  # 2 키포인트 (선택사항)
   bboxnp.array,          # 바운딩 박스 (선택사항)
}
```

### 1.2터 차원
- **keypoints_3d**: `(N, J,3)` 형태
  - N: 프레임 수
  - J: 관절 수 (보통 17개 - Human36M 기준)
  - 3: x, y, z 좌표

## 2 일반적인 입력 데이터 형식별 변환 과정

### 20.1N 형식 데이터
```python
import json
import numpy as np

def json_to_npz(json_file, output_file):
    with open(json_file, ras f:
        data = json.load(f)
    
    # JSON에서 키포인트 추출
    keypoints_3d = []
    for frame_data in data:
        frame_keypoints = []
        for joint in frame_data['keypoints']:
            frame_keypoints.append([jointx joint['y'], joint[z
        keypoints_3d.append(frame_keypoints)
    
    keypoints_3d = np.array(keypoints_3
    
    # NPZ 파일로 저장
    np.savez(output_file, 
             keypoints_3d=keypoints_3d,
             keypoints_2=keypoints_3d:, :, :2],  # 2D는 3D의 x,y만 사용
             bbox=np.ones((len(keypoints_3d), 4 # 기본 바운딩 박스
```

### 2.2V 형식 데이터
```python
import pandas as pd
import numpy as np

def csv_to_npz(csv_file, output_file, joint_names):
    df = pd.read_csv(csv_file)
    
    # 프레임별로 키포인트 재구성
    frames = df[frame'].unique()
    keypoints_3d = []
    
    for frame in frames:
        frame_data = df[df['frame'] == frame]
        frame_keypoints = []
        
        for joint in joint_names:
            joint_data = frame_data[frame_data['joint] == joint].iloc[0]
            frame_keypoints.append([joint_data[x'], joint_data[y'], joint_data['z']])
        
        keypoints_3d.append(frame_keypoints)
    
    keypoints_3d = np.array(keypoints_3d)
    
    np.savez(output_file, 
             keypoints_3d=keypoints_3d,
             keypoints_2=keypoints_3d[:, :, :2
             bbox=np.ones((len(keypoints_3d),4)))
```

### 20.3비디오에서 추출된 키포인트
```python
import cv2
import numpy as np
from pose_estimation_model import PoseEstimator  # 사용하는 모델에 따라

def video_to_npz(video_file, output_file):
    cap = cv2.VideoCapture(video_file)
    keypoints_3d =   
    pose_estimator = PoseEstimator()
    
    while true        ret, frame = cap.read()
        if not ret:
            break
            
        # 3D 포즈 추정
        pose_3d = pose_estimator.predict_3d(frame)
        keypoints_3d.append(pose_3d)
    
    cap.release()
    keypoints_3d = np.array(keypoints_3d)
    
    np.savez(output_file, 
             keypoints_3d=keypoints_3d,
             keypoints_2=keypoints_3d[:, :, :2
             bbox=np.ones((len(keypoints_3d), 4)))
```

## 3 좌표계 변환

### 3.1 카메라 좌표계에서 월드 좌표계로 변환
```python
def camera_to_world_coordinates(keypoints_3a_matrix, rotation_matrix, translation):
    
    카메라 좌표계의 키포인트를 월드 좌표계로 변환
   
    # 카메라 좌표계에서 월드 좌표계로 변환
    world_keypoints = []
    
    for frame_keypoints in keypoints_3d:
        frame_world = []
        for joint in frame_keypoints:
            # 카메라 좌표계에서 월드 좌표계로 변환
            world_joint = np.dot(rotation_matrix.T, joint - translation)
            frame_world.append(world_joint)
        world_keypoints.append(frame_world)
    
    return np.array(world_keypoints)
```

### 3.2 루트 상대 좌표계로 변환
```python
def to_root_relative(keypoints_3d, root_joint_idx=0:
   
    루트 관절을 기준으로 한 상대 좌표계로 변환
    root_relative = keypoints_3opy()
    
    for i in range(len(keypoints_3:
        root_pos = keypoints_3d[i, root_joint_idx]
        root_relative[i] = keypoints_3d[i] - root_pos
    
    return root_relative
```

## 4. 데이터 전처리

### 40.1 키포인트 처리
```python
def interpolate_missing_keypoints(keypoints_3d, max_gap=5):
       누락된 키포인트를 선형 보간으로 채움
   
    for joint_idx in range(keypoints_3d.shape[1        joint_data = keypoints_3d[:, joint_idx]
        
        # NaN 또는 0 값 찾기
        missing_mask = np.isnan(joint_data).any(axis=1) | (joint_data == 0).all(axis=1)
        
        if missing_mask.any():
            # 선형 보간 적용
            valid_indices = np.where(~missing_mask)[0]
            
            for i in range(3각각
                if len(valid_indices) > 1:
                    joint_data[missing_mask, i] = np.interp(
                        np.where(missing_mask)[0],
                        valid_indices,
                        joint_data[valid_indices, i]
                    )
    
    return keypoints_3d
```

### 4.2 노이즈 필터링
```python
def apply_smoothing_filter(keypoints_3d, window_size=5):
        간단한 이동평균 필터 적용
        smoothed = keypoints_3opy()
    
    for joint_idx in range(keypoints_3d.shape[1]):
        for coord_idx in range(3):
            joint_coord = keypoints_3d[:, joint_idx, coord_idx]
            
            # 이동평균 필터
            kernel = np.ones(window_size) / window_size
            smoothed[:, joint_idx, coord_idx] = np.convolve(
                joint_coord, kernel, mode=same'
            )
    
    return smoothed
```

## 5 SmoothNet 설정 파일 준비

### 5.1 데이터셋 설정
```yaml
# configs/custom_dataset.yaml
DATASET:
  NAME: custom_dataset'
  ROOT_CUSTOM_DATASET: '/path/to/your/data'
  TRAIN: 'train'
  VAL: 'val
  TEST: 'test'
  INPUT_SIZE: 256  OUTPUT_SIZE: 256
  SIGMA: 2  FLIP: True
  ROT_FACTOR: 30
  SCALE_FACTOR: 0.25
  ROT_PROB: 00.6

MODEL:
  NAME: 'smoothnet'
  INPUT_SIZE: 256  OUTPUT_SIZE: 256
  NUM_JOINTS: 17 # 관절 수에 맞게 조정
  HIDDEN_DIM: 512
  NUM_LAYERS: 8
  DROPOUT: 00.1RAIN:
  BATCH_SIZE: 32
  LEARNING_RATE: 0.1
  NUM_EPOCHS: 100
  SAVE_INTERVAL: 10

VIS:
  START:0
  END: 100
  FPS: 30
```

##6. 일반적인 오류 및 해결방법

###6.1원 불일치 오류
```python
# 오류: ValueError: cannot reshape array of size X into shape (N,J,3
# 해결: 데이터 차원 확인 및 조정
print(f"데이터 형태: {keypoints_3d.shape}")
print(f"예상 형태: ({num_frames}, {num_joints}, 3))# 차원 조정
if len(keypoints_3d.shape) ==2
    # (N*J, 3) -> (N, J, 3)
    keypoints_3ypoints_3.reshape(-1m_joints,3
```

### 6.2 키 이름 불일치
```python
# NPZ 파일의 키 확인
data = np.load('data.npz)print("사용 가능한 키:", data.files)

# 키 이름 매핑
if 'joints_3d' in data.files:
    keypoints_3d = data['joints_3d]
elifkeypoints_3d' in data.files:
    keypoints_3d = data['keypoints_3d]
else:  raise ValueError(3D 키포인트 데이터를 찾을 수 없습니다)
```

### 6.3 좌표계 문제
```python
# 좌표계 확인 및 변환
def check_coordinate_system(keypoints_3d):
       좌표계가 올바른지 확인하고 필요시 변환
    
    # Y축이 위쪽인지 확인 (일반적으로 Y축이 아래쪽이면 뒤집기)
    if np.mean(keypoints_3, 1]) < 0:
        keypoints_3:, 1= -keypoints_3d[:, :, 1   
    return keypoints_3d
```

##7검증 과정

###7.1이터 품질 확인
```python
def validate_keypoints(keypoints_3d):
      키포인트 데이터의 품질 확인
        # NaN 값 확인
    nan_count = np.isnan(keypoints_3d).sum()
    print(f"NaN 값 개수: {nan_count}")
    
    # 무한대 값 확인
    inf_count = np.isinf(keypoints_3d).sum()
    print(f"무한대 값 개수: {inf_count}")
    
    # 범위 확인
    print(fX 범위: [object Object]keypoints_3d:, :,0.min():.2f} ~ [object Object]keypoints_3d:, :, 0].max():.2f})
    print(fY 범위: [object Object]keypoints_3d:, :,1.min():.2f} ~ [object Object]keypoints_3d:, :, 1].max():.2f})
    print(fZ 범위: [object Object]keypoints_3d:, :,2.min():.2f} ~ [object Object]keypoints_3d:, :, 2].max():.2f}")
    
    # 움직임 확인
    motion = np.diff(keypoints_3d, axis=0  motion_magnitude = np.linalg.norm(motion, axis=2)
    print(f"평균 움직임 크기: {np.mean(motion_magnitude):.4})
```

### 7.2화를 통한 검증
```python
import matplotlib.pyplot as plt

def visualize_keypoints(keypoints_3d, title="3D Keypoints):
   
    키포인트를 3D로 시각화
     fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111 projection=3d)
    
    # 첫 번째 프레임의 키포인트 시각화
    frame_keypoints = keypoints_3d0  ax.scatter(frame_keypoints[:, 0], frame_keypoints[:, 1], frame_keypoints[:, 2])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.show()
```

## 8 완전한 변환 파이프라인 예시

```python
def complete_data_pipeline(input_file, output_file, input_format='json):
    
    완전한 데이터 변환 파이프라인
 
    # 1. 입력 데이터 로드
    if input_format == 'json':
        keypoints_3d = load_json_data(input_file)
    elif input_format == csv
        keypoints_3oad_csv_data(input_file)
    else:
        raise ValueError(f"지원하지 않는 형식: {input_format}")
    
    # 2. 데이터 검증
    validate_keypoints(keypoints_3d)
    
    #3누락된 데이터 보간
    keypoints_3nterpolate_missing_keypoints(keypoints_3d)
    
    #4계 변환
    keypoints_3d = check_coordinate_system(keypoints_3d)
    
    #5 루트 상대 좌표계로 변환
    keypoints_3d = to_root_relative(keypoints_3d)
    
    #6 NPZ 파일로 저장
    np.savez(output_file,
             keypoints_3d=keypoints_3d,
             keypoints_2=keypoints_3d[:, :, :2
             bbox=np.ones((len(keypoints_3)))
    
    print(f"변환 완료: {output_file})    print(f"데이터 형태: {keypoints_3d.shape})
```

##9. 주의사항

1*좌표계 일관성**: 모든 데이터가 동일한 좌표계를 사용하는지 확인
2 **단위 일관성**: 미터, 밀리미터 등 단위가 일관되는지 확인
3 **프레임 레이트**: 원본 비디오의 프레임 레이트와 일치하는지 확인
4 **관절 순서**: 사용하는 포즈 추정 모델의 관절 순서와 일치하는지 확인
5. **데이터 품질**: 노이즈가 심한 데이터는 사전 필터링 고려

## 10참고 자료

- SmoothNet 원본 논문: [링크]
- Human3.6 데이터셋: 링크]
- COCO 키포인트 형식: [링크]
- OpenPose 출력 형식: [링크] 