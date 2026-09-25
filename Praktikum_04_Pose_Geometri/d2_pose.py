import cv2
from cvzone.PoseModule import PoseDetector

def angleCheck(angle, min_target=70, max_target=110):
    """
    Fungsi untuk mengecek apakah sudut masuk dalam target tertentu.
    Ubah min_target dan max_target sesuai kebutuhan gerakan (misal: squat 70-110 derajat).
    """
    return min_target <= angle <= max_target

# Membuat detector pose
detector = PoseDetector()

# Membuka kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka. Coba index 1 atau 2.")

while True:

    # Membaca gambar dari kamera
    success, img = cap.read()

    if not success:
        print("Gagal membaca kamera.")
        break

    # Mendeteksi pose tubuh
    img = detector.findPose(img)

    # Mengambil posisi landmark tubuh
    lmList, bboxInfo = detector.findPosition(img)

    # Jika tubuh terdeteksi
    if lmList:

        # ==========================================
        # Menghitung sudut lutut kiri
        # ==========================================
        angle_left, img = detector.findAngle(
            lmList[23][:2],  # Pinggul kiri
            lmList[25][:2],  # Lutut kiri
            lmList[27][:2],  # Pergelangan kaki kiri
            img=img
        )

        # ==========================================
        # Menghitung sudut lutut kanan
        # ==========================================
        angle_right, img = detector.findAngle(
            lmList[24][:2],  # Pinggul kanan
            lmList[26][:2],  # Lutut kanan
            lmList[28][:2],  # Pergelangan kaki kanan
            img=img
        )

        # Mengecek apakah sudut lutut kiri sesuai target
        status_kiri = "SESUAI TARGET" if angleCheck(angle_left) else "TIDAK SESUAI"
        color_kiri = (0, 255, 0) if angleCheck(angle_left) else (0, 0, 255)

        # Mengecek apakah sudut lutut kanan sesuai target
        status_kanan = "SESUAI TARGET" if angleCheck(angle_right) else "TIDAK SESUAI"
        color_kanan = (0, 255, 0) if angleCheck(angle_right) else (0, 0, 255)

        # ==========================================
        # Menampilkan sudut lutut kiri
        # ==========================================
        cv2.putText(
            img,
            f"Lutut kiri: {int(angle_left)} ({status_kiri})",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color_kiri,
            2
        )

        # ==========================================
        # Menampilkan sudut lutut kanan
        # ==========================================
        cv2.putText(
            img,
            f"Lutut kanan: {int(angle_right)} ({status_kanan})",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color_kanan,
            2
        )

    # Menampilkan kamera
    cv2.imshow("D2 - Pose Estimation", img)

    # Tekan q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Menutup kamera
cap.release()
cv2.destroyAllWindows()