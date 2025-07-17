import os
import torch
import json
import numpy as np
from lib.dataset import find_dataset_using_name
from lib.models.smoothnet import SmoothNet
from lib.core.evaluate import Evaluator
from torch.utils.data import DataLoader
from lib.utils.utils import prepare_output_dir, worker_init_fn
from lib.core.evaluate_config import parse_args


def save_smoothed_keypoints_to_json(model, test_dataset, cfg, output_dir):
    """
    지터 제거된 키포인트를 JSON 파일로 저장
    """
    model.eval()
    
    # 데이터셋에서 원본 키포인트 가져오기
    original_data = test_dataset.keypoints_3d  # (frames, 213)
    
    # 키포인트 수 계산
    keypoint_number = original_data.shape[1] // 3  # 213 // 3 = 71
    
    # 루트 관절 인덱스 (기본값: 0)
    try:
        keypoint_root = eval("cfg.DATASET." + "ROOT_"+cfg.DATASET_NAME.upper() +"_"+ cfg.ESTIMATOR.upper()+"_3D")
    except:
        keypoint_root = 0
        print(f"Using default root joint index: {keypoint_root}")
    
    # 데이터를 (frames, keypoints, 3) 형태로 변환
    data_pred = original_data.reshape(-1, keypoint_number, 3)
    
    # Root relative transformation
    root_pos = data_pred[:, keypoint_root, :]
    data_pred = data_pred - root_pos[:, np.newaxis, :]
    data_pred = torch.tensor(data_pred).to(cfg.DEVICE)
    
    # 슬라이딩 윈도우 적용
    slide_window_size = cfg.MODEL.SLIDE_WINDOW_SIZE
    slide_window_step = cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE
    
    data_len = data_pred.shape[0]
    data_pred_window = torch.as_strided(
        data_pred, ((data_len - slide_window_size) // slide_window_step + 1,
                    slide_window_size, keypoint_number, 3),
        (slide_window_step * keypoint_number * 3,
         keypoint_number * 3, 3, 1),
        storage_offset=0).reshape(-1, slide_window_size, keypoint_number * 3)
    
    # 모델로 지터 제거
    with torch.no_grad():
        data_pred_window = data_pred_window.permute(0, 2, 1)
        predicted_pos = model(data_pred_window)
        predicted_pos = predicted_pos.permute(0, 2, 1)
    
    # 슬라이딩 윈도우 결과를 시퀀스로 변환
    from lib.utils.utils import slide_window_to_sequence
    predicted_pos = slide_window_to_sequence(predicted_pos, slide_window_step, slide_window_size).reshape(-1, keypoint_number, 3)
    
    # 결과를 numpy로 변환
    predicted_pos = predicted_pos.cpu().numpy()
    
    # JSON 파일로 저장
    json_output_dir = os.path.join(output_dir, "json_output")
    os.makedirs(json_output_dir, exist_ok=True)
    
    # 키포인트 이름 (71개 키포인트)
    joint_names = [f"joint_{i}" for i in range(keypoint_number)]
    
    # JSON 데이터 구조 생성
    json_data = {
        "dataset_name": cfg.DATASET_NAME,
        "estimator": cfg.ESTIMATOR,
        "body_representation": cfg.BODY_REPRESENTATION,
        "num_frames": predicted_pos.shape[0],
        "num_joints": predicted_pos.shape[1],
        "joint_names": joint_names,
        "smoothed_keypoints": []
    }
    
    # 각 프레임의 키포인트 데이터 추가
    for frame_idx in range(predicted_pos.shape[0]):
        frame_data = {
            "frame": frame_idx,
            "keypoints": []
        }
        
        for joint_idx in range(predicted_pos.shape[1]):
            joint_data = {
                "joint_id": joint_idx,
                "joint_name": joint_names[joint_idx],
                "x": float(predicted_pos[frame_idx, joint_idx, 0]),
                "y": float(predicted_pos[frame_idx, joint_idx, 1]),
                "z": float(predicted_pos[frame_idx, joint_idx, 2])
            }
            frame_data["keypoints"].append(joint_data)
        
        json_data["smoothed_keypoints"].append(frame_data)
    
    # JSON 파일 저장
    json_filename = f"{cfg.DATASET_NAME}_{cfg.ESTIMATOR}_{cfg.BODY_REPRESENTATION}_smoothed_keypoints.json"
    json_path = os.path.join(json_output_dir, json_filename)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 지터 제거 완료! JSON 파일이 생성되었습니다.")
    print(f"📁 저장 위치: {json_path}")
    print(f"📊 프레임 수: {predicted_pos.shape[0]}")
    print(f"🦴 키포인트 수: {predicted_pos.shape[1]}")


def main(cfg):
    test_datasets=[]

    all_estimator=cfg.ESTIMATOR.split(",")
    all_body_representation=cfg.BODY_REPRESENTATION.split(",")
    all_dataset=cfg.DATASET_NAME.split(",")

    for dataset_index in range(len(all_dataset)):
        estimator=all_estimator[dataset_index]
        body_representation=all_body_representation[dataset_index]
        dataset=all_dataset[dataset_index]

        dataset_class = find_dataset_using_name(dataset)

        print("Loading dataset ("+str(dataset_index)+")......")

        test_datasets.append(dataset_class(cfg,
                                    estimator=estimator,
                                    return_type=body_representation,
                                    phase='test'))
    test_loader=[]

    for test_dataset in test_datasets:
        test_loader.append(DataLoader(dataset=test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=cfg.TRAIN.WORKERS_NUM,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn))

    model = SmoothNet(window_size=cfg.MODEL.SLIDE_WINDOW_SIZE,
                    output_size=cfg.MODEL.SLIDE_WINDOW_SIZE,
                    hidden_size=cfg.MODEL.HIDDEN_SIZE,
                    res_hidden_size=cfg.MODEL.RES_HIDDEN_SIZE,
                    num_blocks=cfg.MODEL.NUM_BLOCK,
                    dropout=cfg.MODEL.DROPOUT).to(cfg.DEVICE)

    if cfg.EVALUATE.PRETRAINED != '' and os.path.isfile(
            cfg.EVALUATE.PRETRAINED):
        checkpoint = torch.load(cfg.EVALUATE.PRETRAINED)
        performance = checkpoint['performance']
        model.load_state_dict(checkpoint['state_dict'])
        print(f'==> Loaded pretrained model from {cfg.EVALUATE.PRETRAINED}...')
    else:
        print(f'{cfg.EVALUATE.PRETRAINED} is not a pretrained model!!!!')
        exit()

    evaluator = Evaluator(model=model, test_loader=test_loader, cfg=cfg)
    evaluator.calculate_flops()
    evaluator.calculate_parameter_number()
    evaluator.run()
    
    # 지터 제거된 키포인트를 JSON 파일로 저장
    print("\n" + "="*50)
    print("지터 제거된 키포인트를 JSON 파일로 저장 중...")
    print("="*50)
    
    # 첫 번째 데이터셋에 대해서만 JSON 파일 생성 (단일 데이터셋인 경우)
    if len(test_datasets) > 0:
        save_smoothed_keypoints_to_json(model, test_datasets[0], cfg, cfg.OUTPUT_DIR)
    
    print("\n🎉 모든 작업이 완료되었습니다!")


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)