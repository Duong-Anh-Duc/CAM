#!/usr/bin/env python3
"""
Script để tự động download các model files cần thiết cho Hệ thống giám sát tập trung học sinh
"""

import os
import urllib.request
import bz2
import shutil
from pathlib import Path

def download_file(url, destination, description):
    """Download file từ URL với progress indicator"""
    print(f"\nDang download {description}...")
    print(f"   URL: {url}")
    print(f"   Đích: {destination}")
    
    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
            print(f"\r   Progress: {percent:.1f}% ({downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB)", end='')
        
        urllib.request.urlretrieve(url, destination, show_progress)
        print(f"\n[OK] Download thanh cong!")
        return True
    except Exception as e:
        print(f"\n[LOI] Loi khi download: {e}")
        return False

def decompress_bz2(source, destination):
    """Giải nén file .bz2"""
    print(f"\nDang giai nen file...")
    try:
        with bz2.BZ2File(source, 'rb') as src:
            with open(destination, 'wb') as dest:
                shutil.copyfileobj(src, dest)
        print("[OK] Giai nen thanh cong!")
        
        # Xóa file .bz2 sau khi giải nén
        os.remove(source)
        print(f"Da xoa file nen: {source}")
        return True
    except Exception as e:
        print(f"[LOI] Loi khi giai nen: {e}")
        return False

def check_file_exists(filepath):
    """Kiểm tra xem file đã tồn tại chưa"""
    return os.path.exists(filepath)

def get_file_size(filepath):
    """Lấy kích thước file"""
    size = os.path.getsize(filepath)
    return f"{size/1024/1024:.2f}MB"

def main():
    """Main function để download tất cả models cần thiết"""
    print("="*70)
    print("   HỆ THỐNG GIÁM SÁT TẬP TRUNG HỦC SINH - MODEL DOWNLOADER")
    print("="*70)
    
    # Tạo thư mục models nếu chưa có
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Model cần download
    models = [
        {
            "name": "shape_predictor_68_face_landmarks.dat",
            "description": "Dlib Facial Landmarks Model (68 điểm đặc trưng khuôn mặt)",
            "url": "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat",
            "path": models_dir / "shape_predictor_68_face_landmarks.dat",
            "compressed": False,
            "required": True,
            "used_in": "blinkDetect.py - Phát hiện nhấp nháy mắt"
        }
    ]
    
    print("\nKiem tra trang thai models...\n")
    
    models_to_download = []
    
    for model in models:
        status = "[OK] DA CO" if check_file_exists(model["path"]) else "[X] THIEU"
        required_text = "[!] BAT BUOC" if model["required"] else "Tuy chon"
        
        print(f"{status} | {model['name']}")
        print(f"        {required_text} - {model['description']}")
        print(f"        Sử dụng trong: {model['used_in']}")
        
        if check_file_exists(model["path"]):
            size = get_file_size(model["path"])
            print(f"        Kích thước: {size}")
        else:
            models_to_download.append(model)
        print()
    
    if not models_to_download:
        print("[OK] Tat ca models can thiet da co san!")
        print("Ban co the chay ung dung ngay bay gio!")
        return
    
    print(f"\n[!] Can download {len(models_to_download)} model(s)\n")
    
    # Hỏi người dùng có muốn download không
    response = input("Bạn có muốn download các models này không? (y/n): ").lower().strip()
    
    if response not in ['y', 'yes', 'có', 'c']:
        print("\n[X] Da huy download.")
        print("[!] Luu y: Ung dung se KHONG chay duoc neu thieu models bat buoc!")
        return
    
    # Download từng model
    success_count = 0
    for model in models_to_download:
        temp_path = str(model["path"]) + ".tmp"
        
        if download_file(model["url"], temp_path, model["description"]):
            if model["compressed"]:
                if decompress_bz2(temp_path, model["path"]):
                    success_count += 1
            else:
                shutil.move(temp_path, model["path"])
                success_count += 1
        
        if check_file_exists(model["path"]):
            size = get_file_size(model["path"])
            print(f"[OK] File da san sang: {model['path']} ({size})")
    
    print("\n" + "="*70)
    if success_count == len(models_to_download):
        print("DA DOWNLOAD THANH CONG TAT CA MODELS!")
        print("[OK] Ban co the chay ung dung bang lenh: python main.py")
    else:
        print(f"[!] Da download thanh cong {success_count}/{len(models_to_download)} models")
        print("[LOI] Mot so models download that bai. Vui long thu lai.")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Da huy download boi nguoi dung.")
    except Exception as e:
        print(f"\n[LOI] Loi khong xac dinh: {e}")
        import traceback
        traceback.print_exc()
