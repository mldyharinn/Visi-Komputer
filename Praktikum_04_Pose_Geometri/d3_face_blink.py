import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import math

# Membuat detector Face Mesh
detector = FaceMeshDetector(
    maxFaces=1
)

# Membuka kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Kamera tidak bisa dibuka. Coba index 1 atau 2."
    )

# Threshold EAR
EYE_AR_THRESHOLD = 0.20

# Jumlah frame mata tertutup sebelum dihitung sebagai kedipan
CLOSED_FRAMES_THRESHOLD = 3

# Variabel penghitung
closed_frames = 0
blink_count = 0


while True:

    # Membaca kamera
    success, img = cap.read()

    if not success:
        print("Gagal membaca kamera.")
        break

    # Mendeteksi wajah dan Face Mesh
    img, faces = detector.findFaceMesh(
        img,
        draw=True
    )

    # Jika wajah terdeteksi
    if faces:

        # Mengambil landmark wajah pertama
        face = faces[0]

        # ==========================================
        # Landmark mata kiri
        # ==========================================

        top = face[159]
        bottom = face[145]
        left = face[33]
        right = face[133]

        # Menghitung jarak vertikal
        vertical = math.dist(
            top,
            bottom
        )

        # Menghitung jarak horizontal
        horizontal = math.dist(
            left,
            right
        )

        # Menghitung EAR
        ear = vertical / horizontal

        # ==========================================
        # Deteksi mata tertutup
        # ==========================================

        if ear < EYE_AR_THRESHOLD:

            closed_frames += 1

        else:

            # Jika sebelumnya mata tertutup
            if closed_frames >= CLOSED_FRAMES_THRESHOLD:
                blink_count += 1

            closed_frames = 0

        # ==========================================
        # Menampilkan EAR
        # ==========================================

        cv2.putText(
            img,
            f"EAR: {ear:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Menampilkan jumlah kedipan
        cv2.putText(
            img,
            f"Blink: {blink_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Menampilkan kamera
    cv2.imshow(
        "D3 - Face Mesh & Blink",
        img
    )

    # Tekan q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Menutup kamera
cap.release()
cv2.destroyAllWindows()