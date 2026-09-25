import cv2
from cvzone.HandTrackingModule import HandDetector
import math


# ==========================================
# Membuat detector tangan
# ==========================================
detector = HandDetector(
    detectionCon=0.7,
    maxHands=1
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
    # Jika tangan terdeteksi
    # ==========================================
    if hands:

        hand = hands[0]

        # Landmark tangan
        lmList = hand["lmList"]

        # ==========================================
        # Mengambil titik landmark
        # ==========================================

        wrist = lmList[0]

        thumb_tip = lmList[4]

        index_tip = lmList[8]

        middle_tip = lmList[12]

        ring_tip = lmList[16]

        pinky_tip = lmList[20]


        # ==========================================
        # Menghitung jarak
        # ==========================================

        thumb_index_distance = math.dist(
            thumb_tip,
            index_tip
        )


        # ==========================================
        # Menghitung jari yang terbuka
        # ==========================================

        fingers = detector.fingersUp(hand)

        finger_count = sum(fingers)


        # ==========================================
        # Default gesture
        # ==========================================

        gesture = "UNKNOWN"


        # ==========================================
        # Gesture OK
        # ==========================================

        if thumb_index_distance < 35:

            gesture = "OK"


        # ==========================================
        # Gesture THUMBS UP
        # ==========================================

        elif (
            fingers[0] == 1
            and fingers[1] == 0
            and fingers[2] == 0
            and fingers[3] == 0
            and fingers[4] == 0
        ):

            gesture = "THUMBS UP"


        # ==========================================
        # Rock
        # ==========================================

        elif finger_count == 0:

            gesture = "ROCK"


        # ==========================================
        # Paper
        # ==========================================

        elif finger_count == 5:

            gesture = "PAPER"


        # ==========================================
        # Scissors
        # ==========================================

        elif (
            fingers[1] == 1
            and fingers[2] == 1
            and fingers[3] == 0
            and fingers[4] == 0
        ):

            gesture = "SCISSORS"


        # ==========================================
        # Menampilkan gesture
        # ==========================================

        cv2.putText(
            img,
            f"Gesture: {gesture}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"Jari terbuka: {finger_count}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ==========================================
    # Menampilkan kamera
    # ==========================================

    cv2.imshow(
        "D5 - Gesture Recognition",
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