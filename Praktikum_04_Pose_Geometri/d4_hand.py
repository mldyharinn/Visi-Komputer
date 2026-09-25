import cv2
from cvzone.HandTrackingModule import HandDetector


# ==========================================
# Membuat detector tangan
# ==========================================
detector = HandDetector(
    detectionCon=0.7,
    maxHands=2
)


# ==========================================
# Membuka kamera
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Kamera tidak bisa dibuka. Coba index 1 atau 2."
    )


while True:

    # ==========================================
    # Membaca kamera
    # ==========================================
    success, img = cap.read()

    if not success:
        print("Gagal membaca kamera.")
        break


    # ==========================================
    # Mendeteksi tangan
    # ==========================================
    hands, img = detector.findHands(
        img,
        flipType=True
    )


    # ==========================================
    # Jika ada tangan terdeteksi
    # ==========================================
    if hands:

        total_fingers = 0

        for hand in hands:

            # Menentukan jari yang terbuka
            fingers = detector.fingersUp(hand)

            # Menghitung jumlah jari
            finger_count = sum(fingers)

            # Menambahkan ke total
            total_fingers += finger_count


        # ==========================================
        # Menampilkan jumlah jari
        # ==========================================
        cv2.putText(
            img,
            f"Jumlah jari: {total_fingers}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


    # ==========================================
    # Menampilkan kamera
    # ==========================================
    cv2.imshow(
        "D4 - Hand & Finger Counting",
        img
    )


    # ==========================================
    # Tekan q untuk keluar
    # ==========================================
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# Menutup kamera
# ==========================================
cap.release()
cv2.destroyAllWindows()