import json
import numpy as np

def compare_original_vs_smoothed():
    """
    원본 NPZ 파일과 지터 제거 후 JSON 파일을 비교
    """
    print("🔍 원본 vs 지터 제거 후 데이터 비교 중...")
    
    # 원본 NPZ 파일 로드
    original_data = np.load('data/custom_hybrik_3D/ohyeah_hybrik_3D_test.npz', allow_pickle=True)
    original_keypoints = original_data['keypoints_3d']  # (326, 213)
    
    # 지터 제거 후 JSON 파일 로드
    with open('results/json_output/custom_hybrik_3D_hybrik_3D_smoothed_keypoints.json', 'r', encoding='utf-8') as f:
        smoothed_data = json.load(f)
    
    print(f"📊 원본 데이터 형태: {original_keypoints.shape}")
    print(f"📊 지터 제거 후 프레임 수: {smoothed_data['num_frames']}")
    
    # 발 관련 키포인트 인덱스
    foot_joints = {
        'left_ankle': 7,
        'right_ankle': 8,
        'left_foot': 10,
        'right_foot': 11,
        'left_toe': 13,
        'right_toe': 14
    }
    
    # 원본 데이터에서 발 위치 추출
    original_foot_data = {}
    for joint_name, joint_idx in foot_joints.items():
        if joint_idx * 3 + 2 < original_keypoints.shape[1]:
            y_values = original_keypoints[:, joint_idx * 3 + 1]  # Y축
            original_foot_data[joint_name] = {
                'min': np.min(y_values),
                'max': np.max(y_values),
                'mean': np.mean(y_values)
            }
    
    # 지터 제거 후 데이터에서 발 위치 추출
    smoothed_foot_data = {}
    for joint_name, joint_idx in foot_joints.items():
        if joint_idx < smoothed_data['num_joints']:
            y_values = []
            for frame_data in smoothed_data['smoothed_keypoints']:
                if joint_idx < len(frame_data['keypoints']):
                    y_values.append(frame_data['keypoints'][joint_idx]['y'])
            
            if y_values:
                smoothed_foot_data[joint_name] = {
                    'min': np.min(y_values),
                    'max': np.max(y_values),
                    'mean': np.mean(y_values)
                }
    
    # 비교 결과 출력
    print("\n📏 발 위치 비교 (Y축):")
    print("=" * 80)
    print(f"{'관절':<12} {'원본(최소/최대/평균)':<25} {'지터제거(최소/최대/평균)':<25} {'변화':<15}")
    print("=" * 80)
    
    for joint_name in foot_joints.keys():
        if joint_name in original_foot_data and joint_name in smoothed_foot_data:
            orig = original_foot_data[joint_name]
            smooth = smoothed_foot_data[joint_name]
            
            orig_str = f"{orig['min']:.3f}/{orig['max']:.3f}/{orig['mean']:.3f}"
            smooth_str = f"{smooth['min']:.3f}/{smooth['max']:.3f}/{smooth['mean']:.3f}"
            
            # 평균값 변화
            change = smooth['mean'] - orig['mean']
            change_str = f"{change:+.3f}"
            
            print(f"{joint_name:<12} {orig_str:<25} {smooth_str:<25} {change_str:<15}")
    
    # 전체 Y축 범위 비교
    print("\n🎯 전체 Y축 범위 비교:")
    print("=" * 50)
    
    # 원본 전체 Y축 범위
    all_original_y = []
    for i in range(original_keypoints.shape[1] // 3):
        y_values = original_keypoints[:, i * 3 + 1]
        all_original_y.extend(y_values)
    
    original_min_y = np.min(all_original_y)
    original_max_y = np.max(all_original_y)
    
    # 지터 제거 후 전체 Y축 범위
    all_smoothed_y = []
    for frame_data in smoothed_data['smoothed_keypoints']:
        for joint in frame_data['keypoints']:
            all_smoothed_y.append(joint['y'])
    
    smoothed_min_y = np.min(all_smoothed_y)
    smoothed_max_y = np.max(all_smoothed_y)
    
    print(f"원본 Y축 범위: {original_min_y:.3f} ~ {original_max_y:.3f}")
    print(f"지터제거 Y축 범위: {smoothed_min_y:.3f} ~ {smoothed_max_y:.3f}")
    print(f"Y축 범위 변화: {smoothed_max_y - smoothed_min_y - (original_max_y - original_min_y):+.3f}")

if __name__ == "__main__":
    compare_original_vs_smoothed() 