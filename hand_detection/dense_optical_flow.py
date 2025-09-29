import cv2
import numpy as np
import os
import urllib.request

#  Haarcascade İndir
def download_haar_cascades():
    haar_files = {
        "haarcascade_frontalface_default.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        "haarcascade_eye.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
    }
    for filename, url in haar_files.items():
        if not os.path.exists(filename):
            print(f"{filename} indiriliyor...")
            urllib.request.urlretrieve(url, filename)
    print("Haarcascade XML dosyaları hazır.")

# ROI Seçici
def select_roi(frame):
    roi = cv2.selectROI("ROI Sec - Elinizi kutucuğa alin", frame, fromCenter=False, showCrosshair=True)
    return roi

# Lucas-Kanade Takibi 
def lucas_kanade_tracking(frame, frame_gray, old_gray, roi, p0, face_cascade, mask_draw,
                          feature_params, lk_params, color=(0, 255, 0)):

    rx, ry, rw, rh = roi

    roi_mask = np.zeros_like(frame_gray, dtype=np.uint8)
    roi_mask[ry:ry + rh, rx:rx + rw] = 255

    # Yüzleri hariç tut
    face_mask = np.ones_like(frame_gray, dtype=np.uint8) * 255
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_mask[y:y + h, x:x + w] = 0

    mask_roi = cv2.bitwise_and(roi_mask, face_mask)

    if p0 is None:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask_roi, **feature_params)

    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) > 5:
                x_min, y_min = np.min(good_new, axis=0).astype(int)
                x_max, y_max = np.max(good_new, axis=0).astype(int)

                w_box, h_box = x_max - x_min, y_max - y_min
                area = w_box * h_box

                if area > 2000 and 0.3 < (w_box / float(h_box + 1e-5)) < 3.5:
                    motion_mag = np.mean(np.linalg.norm(good_new - good_old, axis=1))
                    if motion_mag > 2.0:
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                        cv2.putText(frame, "El", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            for (new, old) in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                mask_draw = cv2.line(mask_draw, (int(a), int(b)), (int(c), int(d)), color, 2)
                frame = cv2.circle(frame, (int(a), int(b)), 3, color, -1)

            img = cv2.add(frame, mask_draw)
            cv2.imshow('Lucas-Kanade Tracking', img)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    return old_gray, p0, mask_draw


# Farneback Takibi
def farneback_tracking(frame, frame_gray, old_gray, roi, face_cascade):
    rx, ry, rw, rh = roi

    roi_mask = np.zeros_like(frame_gray, dtype=np.uint8)
    roi_mask[ry:ry + rh, rx:rx + rw] = 255

    face_mask = np.ones_like(frame_gray, dtype=np.uint8) * 255
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_mask[y:y + h, x:x + w] = 0

    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask_motion = mag > 2

    contours, _ = cv2.findContours(mask_motion.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        area = w_box * h_box
        if area > 2000 and 0.3 < (w_box / float(h_box + 1e-5)) < 3.5:
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
            cv2.putText(frame, "El", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Farneback Tracking', frame)
    return frame_gray.copy()

# Ana Döngü
def main():
    download_haar_cascades()

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    method = None
    roi = None
    p0 = None
    old_gray = None
    mask_draw = None

    print("Yöntem seçmek için 'l' (Lucas) veya 'f' (Farneback) tuşlarına basın.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if method:
            cv2.putText(frame, f"Yontem: {method.capitalize()}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if method in ["lucas", "farneback"] and roi is None:
            roi = select_roi(frame)
            rx, ry, rw, rh = roi
            old_gray = frame_gray.copy()
            mask_draw = np.zeros_like(frame)

        if method == "lucas" and roi is not None:
            old_gray, p0, mask_draw = lucas_kanade_tracking(
                frame, frame_gray, old_gray, roi, p0, face_cascade, mask_draw, feature_params, lk_params
            )

        elif method == "farneback" and roi is not None:
            old_gray = farneback_tracking(frame, frame_gray, old_gray, roi, face_cascade)

        cv2.imshow("Kamera", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('l'):
            method = "lucas"
            roi, p0 = None, None
            print("Lucas-Kanade yöntemi seçildi.")
        elif key == ord('f'):
            method = "farneback"
            roi, p0 = None, None
            print("Farneback yöntemi seçildi.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
