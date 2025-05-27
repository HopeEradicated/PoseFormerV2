import os
import random
import cv2
import numpy as np
import torch

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def compare_keypoints(npz1_path, npz2_path, atol=1e-3, rtol=1e-5, mismatch_threshold=0.01):
    data1 = np.load(npz1_path, allow_pickle=True)
    data2 = np.load(npz2_path, allow_pickle=True)

    if data1.files != data2.files:
        print("Key mismatch in keypoints files")
        return 0, []

    total = 0
    matched = 0
    mismatches = []

    for key in data1.files:
        arr1 = data1[key]
        arr2 = data2[key]

        if arr1.shape != arr2.shape:
            print(f"Shape mismatch in key: {key}")
            continue

        total += arr1.size
        match_mask = np.isclose(arr1, arr2, atol=atol, rtol=rtol)
        matched += np.count_nonzero(match_mask)

        mismatch_count = np.count_nonzero(~match_mask)
        mismatch_ratio = mismatch_count / arr1.size

        if mismatch_ratio > mismatch_threshold:
            mismatch_indices = np.argwhere(~match_mask)
            mismatches.append((key, mismatch_indices.tolist()))

    accuracy = (matched / total) * 100 if total > 0 else 0
    return accuracy, mismatches

def compare_images(img1_path, img2_path, tolerance=5):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print(f"Could not load one of the images: {img1_path} or {img2_path}")
        return None

    if img1.shape != img2.shape:
        print(f"Image sizes do not match: {img1_path} vs {img2_path}")
        return None

    diff = cv2.absdiff(img1, img2)
    match = (diff <= tolerance).all(axis=2)
    accuracy = (np.count_nonzero(match) / match.size) * 100

    return accuracy

def compare_folders(folder1, folder2, tolerance=5):
    files1 = sorted([f for f in os.listdir(folder1) if f.endswith(".png")])
    files2 = sorted([f for f in os.listdir(folder2) if f.endswith(".png")])

    if len(files1) != len(files2):
        print(f"Warning: number of files does not match ({len(files1)} vs {len(files2)})")

    matched_files = zip(files1, files2)
    accuracies = []

    for file1, file2 in matched_files:
        path1 = os.path.join(folder1, file1)
        path2 = os.path.join(folder2, file2)

        acc = compare_images(path1, path2, tolerance=tolerance)
        if acc is not None:
            accuracies.append((file1, acc))

    avg_accuracy = sum(acc for _, acc in accuracies) / len(accuracies) if accuracies else 0

    return avg_accuracy, accuracies

if __name__ == "__main__":
    keypoints_demo = "demo/output/sample_video/input_2D/keypoints.npz"
    keypoints_test = "test/sample_video/input_2D/keypoints.npz"
    accuracy, mismatches = compare_keypoints(keypoints_demo, keypoints_test, rtol=1e-4)
    print(f"\nKeypoint match accuracy: {accuracy:.2f}%")

    base_demo = "demo/output/sample_video"
    base_test = "test/sample_video"
    subfolders = ["pose", "pose2D", "pose3D"]

    for subfolder in subfolders:
        demo_path = os.path.join(base_demo, subfolder)
        test_path = os.path.join(base_test, subfolder)

        print(f"\nComparing folder: {subfolder}")
        avg_acc, all_accuracies = compare_folders(demo_path, test_path)
        print(f"Average pixel-wise accuracy: {avg_acc:.2f}%")