import os
import cv2
import numpy as np

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
    base_demo = "demo/output/sample_video"
    base_test = "test/sample_video"
    subfolders = ["pose", "pose2D", "pose3D"]

    for subfolder in subfolders:
        demo_path = os.path.join(base_demo, subfolder)
        test_path = os.path.join(base_test, subfolder)

        print(f"\nComparing folder: {subfolder}")
        avg_acc, all_accuracies = compare_folders(demo_path, test_path)

        print(f"Average pixel-wise accuracy: {avg_acc:.2f}%")