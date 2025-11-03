import cv2
import face_recognition
import os
from datetime import datetime

# Buat folder hasil kalau belum ada
os.makedirs("captured_faces", exist_ok=True)

# Buka webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]  # BGR ke RGB

    # Deteksi semua wajah
    face_locations = face_recognition.face_locations(rgb_frame)

    # Gambar kotak bounding
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Tampilkan counter wajah
    cv2.putText(frame, f'Wajah Terdeteksi: {len(face_locations)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Deteksi Wajah - Tekan C untuk Capture', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Simpan wajah yang terdeteksi
        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_img = frame[top:bottom, left:right]
            filename = f"captured_faces/face_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
            cv2.imwrite(filename, face_img)
            print(f" Wajah disimpan: {filename}")
    elif key == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
