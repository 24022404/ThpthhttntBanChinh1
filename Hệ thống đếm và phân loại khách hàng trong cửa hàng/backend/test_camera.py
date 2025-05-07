import cv2
import time
from deepface import DeepFace
import numpy as np

# Kiểm tra camera
print("Đang kiểm tra camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Lỗi: Không thể mở camera. Hãy kiểm tra:")
    print("  - Camera có được kết nối không?")
    print("  - Có ứng dụng khác đang sử dụng camera không?")
    print("  - Bạn đã cấp quyền truy cập camera chưa?")
else:
    print("✅ Camera hoạt động bình thường!")
    
    # Chụp một ảnh thử nghiệm
    ret, frame = cap.read()
    if ret:
        print("✅ Chụp ảnh thành công!")
        
        # Lưu ảnh để kiểm tra
        cv2.imwrite('test_camera.jpg', frame)
        print("✅ Đã lưu ảnh 'test_camera.jpg'")
        
        # Thử phân tích với DeepFace
        print("\nĐang kiểm tra DeepFace...")
        try:
            # Chuyển đổi sang RGB (DeepFace yêu cầu RGB, OpenCV sử dụng BGR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            print("Đang phân tích khuôn mặt... (có thể mất vài giây)")
            results = DeepFace.analyze(
                img_path=rgb_frame,
                actions=['age', 'gender'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            if isinstance(results, dict):
                results = [results]
                
            print(f"✅ Tìm thấy {len(results)} khuôn mặt trong ảnh!")
            for i, face in enumerate(results):
                print(f"\nKhuôn mặt #{i+1}:")
                print(f"  - Giới tính: {face.get('gender', {}).get('dominant', 'Không xác định')}")
                print(f"  - Độ tuổi: {face.get('age', 'Không xác định')}")
                
                # Vẽ khung khuôn mặt lên ảnh
                region = face.get("region", {})
                if region:
                    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
            # Lưu ảnh với các khuôn mặt được đánh dấu
            cv2.imwrite('test_faces.jpg', frame)
            print("✅ Đã lưu ảnh có đánh dấu khuôn mặt 'test_faces.jpg'")
            
        except Exception as e:
            print(f"❌ Lỗi khi sử dụng DeepFace: {e}")
    else:
        print("❌ Không thể chụp ảnh từ camera!")
        
    # Giải phóng camera
    cap.release()

print("\nKiểm tra hoàn tất!")
