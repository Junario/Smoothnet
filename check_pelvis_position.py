import json
import numpy as np

def check_pelvis_position():
    """
    지터 제거 후 골반의 위치값 확인
    """
    print("🔍 지터 제거 후 골반 위치 확인 중...")
    
    # 지터 제거 후 JSON 파일 로드
    with open('results/json_output/custom_hybrik_3D_hybrik_3D_smoothed_keypoints.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 프레임 수: {data['num_frames']}")
    print(f"📊 키포인트 수: {data['num_joints']}")
    
    # 골반 키포인트 인덱스 (일반적으로 0번이 골반)
    pelvis_idx = 0
    
    # 모든 프레임의 골반 위치 추출
    pelvis_positions = []
    
    for frame_data in data['smoothed_keypoints']:
        frame_idx = frame_data['frame']
        
        if pelvis_idx < len(frame_data['keypoints']):
            pelvis = frame_data['keypoints'][pelvis_idx]
            pelvis_positions.append({
                'frame': frame_idx,
                'x': pelvis['x'],
                'y': pelvis['y'],
                'z': pelvis['z']
            })
    
    # 골반 위치 통계
    x_values = [pos['x'] for pos in pelvis_positions]
    y_values = [pos['y'] for pos in pelvis_positions]
    z_values = [pos['z'] for pos in pelvis_positions]
    
    print(f"\n📊 골반 위치 통계:")
    print("=" * 50)
    print(f"X축: 최소={np.min(x_values):.6f}, 최대={np.max(x_values):.6f}, 평균={np.mean(x_values):.6f}")
    print(f"Y축: 최소={np.min(y_values):.6f}, 최대={np.max(y_values):.6f}, 평균={np.mean(y_values):.6f}")
    print(f"Z축: 최소={np.min(z_values):.6f}, 최대={np.max(z_values):.6f}, 평균={np.mean(z_values):.6f}")
    
    # (0,0,0)에 가까운지 확인
    tolerance = 1e-6  # 허용 오차
    
    is_x_zero = all(abs(x) < tolerance for x in x_values)
    is_y_zero = all(abs(y) < tolerance for y in y_values)
    is_z_zero = all(abs(z) < tolerance for z in z_values)
    
    print(f"\n🎯 골반이 (0,0,0)에 있는지 확인:")
    print("=" * 50)
    print(f"X축이 0인가? {'✅ 예' if is_x_zero else '❌ 아니오'}")
    print(f"Y축이 0인가? {'✅ 예' if is_y_zero else '❌ 아니오'}")
    print(f"Z축이 0인가? {'✅ 예' if is_z_zero else '❌ 아니오'}")
    
    if is_x_zero and is_y_zero and is_z_zero:
        print(f"\n🎉 모든 프레임에서 골반이 (0,0,0)에 있습니다!")
    else:
        print(f"\n⚠️ 골반이 (0,0,0)에 있지 않습니다.")
    
    # 첫 10개 프레임의 골반 위치 상세 출력
    print(f"\n📋 첫 10개 프레임의 골반 위치:")
    print("=" * 50)
    
    for i in range(min(10, len(pelvis_positions))):
        pos = pelvis_positions[i]
        print(f"프레임 {pos['frame']:2d}: ({pos['x']:8.6f}, {pos['y']:8.6f}, {pos['z']:8.6f})")
    
    # (0,0,0)에서 가장 멀리 떨어진 프레임 찾기
    distances = []
    for pos in pelvis_positions:
        distance = np.sqrt(pos['x']**2 + pos['y']**2 + pos['z']**2)
        distances.append((pos['frame'], distance))
    
    max_distance_frame = max(distances, key=lambda x: x[1])
    print(f"\n📍 (0,0,0)에서 가장 멀리 떨어진 프레임:")
    print(f"   프레임 {max_distance_frame[0]}: 거리 {max_distance_frame[1]:.6f}")

if __name__ == "__main__":
    check_pelvis_position() 