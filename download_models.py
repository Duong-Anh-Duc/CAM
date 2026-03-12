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
    print(f"\n📥 Đang download {description}...")
    print(f"   URL: {url}")
    print(f"   Đích: {destination}")
    
    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
            print(f"\r   Progress: {percent:.1f}% ({downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB)", end='')
        
        urllib.request.urlretrieve(url, destination, show_progress)
        print(f"\n✅ Download thành công!")
        return True
    except Exception as e:
        print(f"\n❌ Lỗi khi download: {e}")
        return False

def decompress_bz2(source, destination):
    """Giải nén file .bz2"""
    print(f"\n📦 Đang giải nén file...")
    try:
        with bz2.BZ2File(source, 'rb') as src:
            with open(destination, 'wb') as dest:
                shutil.copyfileobj(src, dest)
        print("✅ Giải nén thành công!")
        
        # Xóa file .bz2 sau khi giải nén
        os.remove(source)
        print(f"🗑️  Đã xóa file nén: {source}")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi giải nén: {e}")
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
    models_dir = Path("models")
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
    
    print("\n📋 Kiểm tra trạng thái models...\n")
    
    models_to_download = []
    
    for model in models:
        status = "✅ ĐÃ CÓ" if check_file_exists(model["path"]) else "❌ THIẾU"
        required_text = "⚠️  BẮT BUỘC" if model["required"] else "Tùy chọn"
        
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
        print("✅ Tất cả models cần thiết đã có sẵn!")
        print("🎉 Bạn có thể chạy ứng dụng ngay bây giờ!")
        return
    
    print(f"\n⚠️  Cần download {len(models_to_download)} model(s)\n")
    
    # Hỏi người dùng có muốn download không
    response = input("Bạn có muốn download các models này không? (y/n): ").lower().strip()
    
    if response not in ['y', 'yes', 'có', 'c']:
        print("\n❌ Đã hủy download.")
        print("⚠️  Lưu ý: Ứng dụng sẽ KHÔNG chạy được nếu thiếu models bắt buộc!")
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
            print(f"✅ File đã sẵn sàng: {model['path']} ({size})")
    
    print("\n" + "="*70)
    if success_count == len(models_to_download):
        print("🎉 ĐÃ DOWNLOAD THÀNH CÔNG TẤT CẢ MODELS!")
        print("✅ Bạn có thể chạy ứng dụng bằng lệnh: python main.py")
    else:
        print(f"⚠️  Đã download thành công {success_count}/{len(models_to_download)} models")
        print("❌ Một số models download thất bại. Vui lòng thử lại.")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã hủy download bởi người dùng.")
    except Exception as e:
        print(f"\n❌ Lỗi không xác định: {e}")
        import traceback
        traceback.print_exc()
