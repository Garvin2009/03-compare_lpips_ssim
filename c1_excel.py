import lpips
import torch
from PIL import Image
import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt

def compute_ssim(image1_path, image2_path):
    """ 計算 SSIM (結構相似性) 指標 """
    image1 = np.array(Image.open(image1_path).convert('RGB'))
    image2 = np.array(Image.open(image2_path).convert('RGB'))

    # 確保兩個圖像具有相同的尺寸
    assert image1.shape == image2.shape, "Images must have the same dimensions for SSIM."

    # 設定合適的 win_size
    min_dim = min(image1.shape[:2])  # 取得最小邊長
    win_size = min(7, min_dim) if min_dim >= 7 else 3  # 確保 win_size 不超過圖片大小

    print(f"Using win_size={win_size} for SSIM calculation.")

    # 計算 SSIM
    ssim_value = ssim(image1, image2, channel_axis=-1, win_size=win_size)
    return ssim_value

def compute_lpips_distance(image1_path, image2_path, net='alex'):
    """ 計算 LPIPS (感知相似度) 距離 """
    lpips_model = lpips.LPIPS(net=net)

    image1_tensor = lpips.im2tensor(lpips.load_image(image1_path))
    image2_tensor = lpips.im2tensor(lpips.load_image(image2_path))

    # 計算 LPIPS 距離
    with torch.no_grad():
        distance = lpips_model.forward(image1_tensor, image2_tensor)

    return distance.item()

def compare_handwritings(folder_path, my_handwriting_paths):
    """ 比較手寫字圖片相似度，回傳 DataFrame """
    results = []

    # 過濾出圖片檔案
    images_to_compare = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('png', 'jpg', 'jpeg'))
    ]

    for my_handwriting_path in my_handwriting_paths:
        for image_path in images_to_compare:
            lpips_distance = compute_lpips_distance(my_handwriting_path, image_path)
            ssim_value = compute_ssim(my_handwriting_path, image_path)
            results.append((os.path.basename(my_handwriting_path), os.path.basename(image_path), lpips_distance, ssim_value))

    # 建立 DataFrame
    df = pd.DataFrame(results, columns=['Your_Image', 'Other_image', 'LPIPS', 'SSIM'])
    df.sort_values(by='LPIPS', inplace=True)

    return df

# 設定資料夾
folder_path = 'D:/compare_lpips_ssim-main/4E82' //將4E82改名成全部人「陳」字的資料夾名稱，不要用中文
my_handwriting_folder = 'D:/compare_lpips_ssim-main/yours'

# 確保 `yours` 資料夾內有圖片
my_handwriting_images = [
    os.path.join(my_handwriting_folder, f) for f in os.listdir(my_handwriting_folder)
    if f.lower().endswith(('png', 'jpg', 'jpeg'))
]

if not my_handwriting_images:
    raise ValueError("錯誤：'yours' 資料夾內沒有圖片！請確認有 PNG、JPG 或 JPEG 檔案。")

# 執行比對
df = compare_handwritings(folder_path, my_handwriting_images)
print(df)

# 確保 `excel` 資料夾存在
output_folder = 'excel'
os.makedirs(output_folder, exist_ok=True)

# 存成 CSV
output_csv_path = os.path.join(output_folder, 'results.csv')
df.to_csv(output_csv_path, index=False)
print(f"結果已儲存至 {output_csv_path}")
