import cv2
from cvzone.PoseModule import PoseDetector


# ==========================================
# Membuat detector pose
# ==========================================
detector = PoseDetector()


# ==========================================
# Membuka kamera
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Kamera tidak bisa dibuka. Coba index 1 atau 2."
    )


# ==========================================
# MODE
# squat = menghitung squat
# pushup = menghitung push-up
# ==========================================
MODE = "squat"


# ==========================================
# Parameter squat
# ==========================================
KNEE_DOWN = 80
KNEE_UP = 160


# ==========================================
# Parameter push-up
# ==========================================
DOWN_R = 0.85
UP_R = 1.00


# ==========================================
# Debounce
# ==========================================
SAMPLE_OK = 4


# ==========================================
# Variabel squat
# ==========================================
squat_count = 0
squat_state = "up"
squat_ok_frames = 0


# ==========================================
# Variabel push-up
# ==========================================
pushup_count = 0
pushup_state = "up"
pushup_ok_frames = 0


while True:

    # ==========================================
    # Membaca kamera
    # ==========================================
    success, img = cap.read()

    if not success:
        print("Gagal membaca kamera.")
        break


    # ==========================================
    # Mendeteksi pose
    # ==========================================
    img = detector.findPose(img)

    lmList, bboxInfo = detector.findPosition(img)


    # ==========================================
    # Jika tubuh terdeteksi
    # ==========================================
    if lmList:

        # ======================================
        # MODE SQUAT
        # ======================================
        if MODE == "squat":

            # Sudut lutut kiri
            angle_left, _ = detector.findAngle(
                lmList[23][:2],
                lmList[25][:2],
                lmList[27][:2]
            )

            # Sudut lutut kanan
            angle_right, _ = detector.findAngle(
                lmList[24][:2],
                lmList[26][:2],
                lmList[28][:2]
            )

            # Rata-rata sudut kedua lutut
            knee_angle = (
                angle_left + angle_right
            ) / 2


            # -------------------------------
            # Posisi DOWN
            # -------------------------------
            if knee_angle < KNEE_DOWN:

                squat_ok_frames += 1

                if squat_ok_frames >= SAMPLE_OK:
                    squat_state = "down"


            # -------------------------------
            # Posisi UP
            # -------------------------------
            elif knee_angle > KNEE_UP:

                squat_ok_frames += 1

                if squat_ok_frames >= SAMPLE_OK:

                    # Jika sebelumnya DOWN,
                    # berarti satu squat selesai
                    if squat_state == "down":
                        squat_count += 1

                    squat_state = "up"


            else:

                squat_ok_frames = 0


            # -------------------------------
            # Tampilan squat
            # -------------------------------
            cv2.putText(
                img,
                "MODE: SQUAT",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                f"Squat: {squat_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                f"Sudut lutut: {int(knee_angle)}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        # ======================================
        # MODE PUSH-UP
        # ======================================
        else:

            # Titik landmark kiri
            shoulder = lmList[11]
            wrist = lmList[15]
            hip = lmList[23]


            # Menghitung jarak
            shoulder_wrist = detector.findDistance(
                lmList[11][:2],
                lmList[15][:2]
            )[0]

            shoulder_hip = detector.findDistance(
                lmList[11][:2],
                lmList[23][:2]
            )[0]


            # Mencegah pembagian dengan nol
            if shoulder_hip != 0:

                ratio = (
                    shoulder_wrist /
                    shoulder_hip
                )


                # -------------------------------
                # Posisi DOWN
                # -------------------------------
                if ratio < DOWN_R:

                    pushup_ok_frames += 1

                    if pushup_ok_frames >= SAMPLE_OK:
                        pushup_state = "down"


                # -------------------------------
                # Posisi UP
                # -------------------------------
                elif ratio > UP_R:

                    pushup_ok_frames += 1

                    if pushup_ok_frames >= SAMPLE_OK:

                        # Jika sebelumnya DOWN,
                        # berarti satu push-up selesai
                        if pushup_state == "down":
                            pushup_count += 1

                        pushup_state = "up"


                else:

                    pushup_ok_frames = 0


                # -------------------------------
                # Tampilan push-up
                # -------------------------------
                cv2.putText(
                    img,
                    "MODE: PUSH-UP",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    img,
                    f"Push-up: {pushup_count}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    img,
                    f"Ratio: {ratio:.2f}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


    # ==========================================
    # Petunjuk
    # ==========================================
    cv2.putText(
        img,
        "Tekan 'm' untuk ganti mode",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ==========================================
    # Menampilkan kamera
    # ==========================================
    cv2.imshow(
        "D6 - Activity Counter",
        img
    )


    # ==========================================
    # Tombol keyboard
    # ==========================================
    key = cv2.waitKey(1) & 0xFF

    # q = keluar
    if key == ord("q"):
        break

    # m = ganti mode
    elif key == ord("m"):

        if MODE == "squat":
            MODE = "pushup"
        else:
            MODE = "squat"


# ==========================================
# Menutup kamera
# ==========================================
cap.release()
cv2.destroyAllWindows()