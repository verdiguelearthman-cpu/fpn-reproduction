import os
import zipfile
import requests
import time


def download_file_with_resume(url, filepath, max_retries=5):
    """
    带有断点续传和自动重试功能的下载函数
    """
    retries = 0
    while retries < max_retries:
        try:
            # 检查本地已下载的文件大小
            downloaded_size = 0
            if os.path.exists(filepath):
                downloaded_size = os.path.getsize(filepath)

            # 设置 HTTP 请求头，请求从已下载的字节处开始
            headers = {}
            if downloaded_size > 0:
                headers['Range'] = f'bytes={downloaded_size}-'
                print(f"Resuming download from {downloaded_size} bytes...")

            # 发起请求
            response = requests.get(url, stream=True, headers=headers, timeout=30)

            # 检查响应状态码
            if response.status_code == 416:  # Range Not Satisfiable，说明已经下载完了
                print(f"File {filepath} already fully downloaded.")
                return True
            elif response.status_code not in [200, 206]:  # 206 是 Partial Content，表示支持断点续传
                print(f"Failed to download. Status code: {response.status_code}")
                return False

            # 获取文件总大小（如果服务器返回了 Content-Length）
            total_size = int(response.headers.get('content-length', 0)) + downloaded_size

            # 以追加模式打开文件，写入数据
            mode = "ab" if downloaded_size > 0 else "wb"
            with open(filepath, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # 检查下载是否完整
            if total_size > 0 and os.path.getsize(filepath) < total_size:
                raise requests.exceptions.ChunkedEncodingError("Download incomplete.")

            print(f"Successfully downloaded {filepath}")
            return True

        except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError) as e:
            retries += 1
            print(f"Network error occurred: {e}. Retrying ({retries}/{max_retries}) in 5 seconds...")
            time.sleep(5)

    print(f"Failed to download {url} after {max_retries} retries.")
    return False


def download_and_extract_coco_val2017(base_dir="."):
    coco_dir = os.path.join(base_dir, "coco")
    os.makedirs(coco_dir, exist_ok=True)

    # COCO 2017 验证集图片下载链接
    val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
    val_images_zip_path = os.path.join(coco_dir, "val2017.zip")
    val_images_extract_path = os.path.join(coco_dir, "val2017")

    # COCO 2017 标注文件下载链接
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    annotations_zip_path = os.path.join(coco_dir, "annotations_trainval2017.zip")
    annotations_extract_path = os.path.join(coco_dir, "annotations")

    # 下载并解压验证集图片
    print(f"Downloading COCO 2017 validation images from {val_images_url}...")
    if download_file_with_resume(val_images_url, val_images_zip_path):
        if not os.path.exists(val_images_extract_path):
            print("Download complete. Extracting images...")
            try:
                with zipfile.ZipFile(val_images_zip_path, "r") as zip_ref:
                    zip_ref.extractall(coco_dir)
                if os.path.exists(os.path.join(coco_dir, "val2017")) and os.path.join(coco_dir,
                                                                                      "val2017") != val_images_extract_path:
                    os.rename(os.path.join(coco_dir, "val2017"), val_images_extract_path)
                print(f"Extracted validation images to {val_images_extract_path}")
            except zipfile.BadZipFile:
                print("Error: The downloaded zip file is corrupted. Please delete it and try again.")
                return
        else:
            print(f"Images already extracted to {val_images_extract_path}")

    # 下载并解压标注文件
    print(f"\nDownloading COCO 2017 annotations from {annotations_url}...")
    if download_file_with_resume(annotations_url, annotations_zip_path):
        if not os.path.exists(annotations_extract_path):
            print("Download complete. Extracting annotations...")
            try:
                with zipfile.ZipFile(annotations_zip_path, "r") as zip_ref:
                    zip_ref.extractall(coco_dir)
                if os.path.exists(os.path.join(coco_dir, "annotations")) and os.path.join(coco_dir,
                                                                                          "annotations") != annotations_extract_path:
                    os.rename(os.path.join(coco_dir, "annotations"), annotations_extract_path)
                print(f"Extracted annotations to {annotations_extract_path}")
            except zipfile.BadZipFile:
                print("Error: The downloaded zip file is corrupted. Please delete it and try again.")
                return
        else:
            print(f"Annotations already extracted to {annotations_extract_path}")

    print("\nCOCO 2017 validation dataset preparation complete!")


if __name__ == "__main__":
    download_and_extract_coco_val2017()