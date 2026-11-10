import numpy as np
from pathlib import Path


def read_skeleton(path):
    path = Path(path)
    metadata = _parse_sample_name(path.stem)

    with path.open("r") as f:
        num_frames = int(f.readline())
        frames = []

        for _ in range(num_frames):
            num_bodies = int(f.readline())
            bodies = []

            for _ in range(num_bodies):
                body_info = f.readline().strip().split()
                body_id = int(body_info[0])

                num_joints = int(f.readline())
                joints = np.zeros((num_joints, 3), dtype=np.float32)

                for joint_idx in range(num_joints):
                    joint_info = f.readline().strip().split()
                    joints[joint_idx] = [
                        float(joint_info[0]),
                        float(joint_info[1]),
                        float(joint_info[2])
                    ]

                bodies.append({
                    "body_id": body_id,
                    "joints": joints,
                })

            frames.append(bodies)

    return {
        "sample_id": path.stem,
        **metadata,
        "num_frames": num_frames,
        "frames": frames,
    }

def _parse_sample_name(sample_name):
    return {
        "setup": int(sample_name[1:4]),
        "camera": int(sample_name[5:8]),
        "subject": int(sample_name[9:12]),
        "repetition": int(sample_name[13:16]),
        "action": int(sample_name[17:20]),
    }

def skeleton_to_array(sample):
    frames = sample["frames"]
    num_frames = sample["num_frames"]

    body_ids = []
    for frame in frames:
        for body in frame:
            body_id = body["body_id"]
            
            if body_id not in body_ids:
                body_ids.append(body_id)

    body_to_idx = {body_id: i for i, body_id in enumerate(body_ids)}

    num_bodies = len(body_ids)

    data = np.zeros((num_bodies, num_frames, 25, 3), dtype=np.float32)
    mask = np.zeros((num_bodies, num_frames), dtype=bool)

    for frame_idx, frame in enumerate(frames):
        for body in frame:
            body_idx = body_to_idx[body["body_id"]]

            if body["joints"].shape != (25, 3):
                raise ValueError(f"Expected (25, 3) got {body['joints'].shape}")

            data[body_idx, frame_idx] = body["joints"]
            mask[body_idx, frame_idx] = True

    return data, mask, body_ids

def find_skeleton_files(*roots):
    files = []

    for root in roots:
        root = Path(root)
        files.extend(root.glob("*.skeleton"))

    return sorted(files)

def read_missing_samples(path):
    path = Path(path)

    with path.open("r") as f:
        return {line.strip() for line in f if line.strip()}

def filter_missing_samples(files, missing_samples):
    return [path for path in files if path.stem not in missing_samples]


if __name__ == "__main__":
    root1 = Path("D:/데이터셋/NTU RGB+D 120 - Skeletons/nturgbd_skeletons_s001_to_s017")
    root2 = Path("D:/데이터셋/NTU RGB+D 120 - Skeletons/nturgbd_skeletons_s018_to_s032")

    files = find_skeleton_files(root1, root2)

    print("Total:", len(files))
    print("First:", files[0].name)
    print("Last :", files[-1].name)

    assert len(files) == 114480

    missing_path = "D:/데이터셋/NTU RGB+D 120 - Skeletons/NTU_RGBD120_samples_with_missing_skeletons.txt"

    missing_samples = read_missing_samples(missing_path)
    filtered_files = filter_missing_samples(files, missing_samples)

    print("Missing :", len(missing_samples))
    print("Before  :", len(files))
    print("After   :", len(filtered_files))

    file_ids = {path.stem for path in files}

    not_found = missing_samples - file_ids

    print("Missing list:", len(missing_samples))
    print("Actually removed:", len(files) - len(filtered_files))
    print("Not found:", len(not_found))
    print(not_found)