import cv2
import os

det = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320, 320))
rec = cv2.FaceRecognizerSF.create("face_recognition_sface_2021dec.onnx", "")

def get_embedding(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError("Can't read " + path)

    h, w = img.shape[:2]
    if w > 800:
        scale = 800 / w
        img = cv2.resize(img, (800, int(h * scale)))
        h, w = img.shape[:2]

    det.setInputSize((w, h))
    det.setScoreThreshold(0.7)
    _, faces = det.detect(img)

    if faces is None:
        raise ValueError("No face found in " + path)

    aligned = rec.alignCrop(img, faces[0])
    return rec.feature(aligned)

known = {}
for person in os.listdir("known"):
    folder = os.path.join("known", person)
    if not os.path.isdir(folder):
        continue
    known[person] = []
    for file in os.listdir(folder):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
           known[person].append(get_embedding(os.path.join(folder, file)))
        except ValueError as e:
            print("Skipped:", e)

for person, embs in known.items():
    print(person, len(embs), "photos")



THRESHOLD = 0.45

def identify(emb):
    best_score = -1
    best_name = "Unknown"
    best_file=""
    for name, emb_list in known.items():
        for saved in emb_list:
            score = rec.match(emb, saved, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score = score
                best_name = name
               
    if best_score >= THRESHOLD:
        return best_name, best_score
    return "Unknown", best_score

from datetime import datetime

marked = set()          # names already marked today

def mark_attendance(name):
    if name == "Unknown" or name in marked:
        return
    marked.add(name)
    now = datetime.now()
    with open("attendance.csv", "a") as f:
        f.write(f"{name},{now.strftime('%Y-%m-%d')},{now.strftime('%H:%M:%S')}\n")
    print("Marked:", name)
    
cam = cv2.VideoCapture(0)

ok, frame = cam.read()
h, w = frame.shape[:2]
det.setInputSize((w, h))      # tell YuNet the camera's frame size, once
det.setScoreThreshold(0.8)

while True:
    ok, frame = cam.read()
    if not ok:
        break

    _, faces = det.detect(frame)

    if faces is not None:
        for face in faces:
            aligned = rec.alignCrop(frame, face)
            emb = rec.feature(aligned)
            
            name, score = identify(emb)
            
            mark_attendance(name)

            x, y, bw, bh = face[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {score:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()