import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_foot_positions(json_file_path):
    """
    JSON 파일에서 발의 위치를 분석
    """
    print("🔍 발의 위치 분석 중...")
    
    # JSON 파일 로드
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 데이터셋: {data['dataset_name']}")
    print(f"📊 프레임 수: {data['num_frames']}")
    print(f"📊 키포인트 수: {data['num_joints']}")
    
    # 발 관련 키포인트 인덱스 (일반적인 71개 키포인트 기준)
    # 실제 인덱스는 데이터셋에 따라 다를 수 있음
    foot_joints = {
        'left_ankle': 7,      # 왼쪽 발목
        'right_ankle': 8,     # 오른쪽 발목
        'left_foot': 10,      # 왼쪽 발
        'right_foot': 11,     # 오른쪽 발
        'left_toe': 13,       # 왼쪽 발가락
        'right_toe': 14       # 오른쪽 발가락
    }
    
    # 모든 프레임의 발 위치 데이터 추출
    foot_data = {name: {'x': [], 'y': [], 'z': []} for name in foot_joints.keys()}
    
    for frame_data in data['smoothed_keypoints']:
        frame_idx = frame_data['frame']
        
        for joint_name, joint_idx in foot_joints.items():
            if joint_idx < len(frame_data['keypoints']):
                joint = frame_data['keypoints'][joint_idx]
                foot_data[joint_name]['x'].append(joint['x'])
                foot_data[joint_name]['y'].append(joint['y'])
                foot_data[joint_name]['z'].append(joint['z'])
    
    # Y축(높이) 분석
    print("\n📏 발의 높이(Y축) 분석:")
    print("=" * 50)
    
    for joint_name, coords in foot_data.items():
        y_values = np.array(coords['y'])
        min_y = np.min(y_values)
        max_y = np.max(y_values)
        mean_y = np.mean(y_values)
        std_y = np.std(y_values)
        
        print(f"{joint_name:12s}: 최소={min_y:8.3f}, 최대={max_y:8.3f}, 평균={mean_y:8.3f}, 표준편차={std_y:8.3f}")
    
    # 가장 낮은 발 위치 찾기
    all_y_values = []
    for coords in foot_data.values():
        all_y_values.extend(coords['y'])
    
    min_overall_y = np.min(all_y_values)
    max_overall_y = np.max(all_y_values)
    mean_overall_y = np.mean(all_y_values)
    
    print(f"\n🎯 전체 발 위치 요약:")
    print(f"   최소 높이: {min_overall_y:.3f}")
    print(f"   최대 높이: {max_overall_y:.3f}")
    print(f"   평균 높이: {mean_overall_y:.3f}")
    print(f"   높이 범위: {max_overall_y - min_overall_y:.3f}")
    
    # 발이 땅에 붙어있는지 판단
    # 일반적으로 Y축이 0에 가까우면 땅에 붙어있는 것
    ground_threshold = 0.1  # 임계값 (데이터에 따라 조정 필요)
    
    print(f"\n🌍 발의 지면 접촉 분석:")
    print(f"   지면 임계값: {ground_threshold}")
    
    if min_overall_y < ground_threshold:
        print(f"   ✅ 발이 지면에 접촉함 (최소 높이: {min_overall_y:.3f})")
    else:
        print(f"   ❌ 발이 지면에서 떨어져 있음 (최소 높이: {min_overall_y:.3f})")
    
    # 첫 10개 프레임의 발 위치 상세 분석
    print(f"\n📋 첫 10개 프레임의 발 위치:")
    print("=" * 50)
    
    for i in range(min(10, len(data['smoothed_keypoints']))):
        frame_data = data['smoothed_keypoints'][i]
        print(f"프레임 {i:2d}: ", end="")
        
        for joint_name, joint_idx in foot_joints.items():
            if joint_idx < len(frame_data['keypoints']):
                joint = frame_data['keypoints'][joint_idx]
                print(f"{joint_name}({joint['y']:.3f}) ", end="")
        print()
    
    return foot_data

if __name__ == "__main__":
    json_file = "results/json_output/custom_hybrik_3D_hybrik_3D_smoothed_keypoints.json"
    foot_data = analyze_foot_positions(json_file) 