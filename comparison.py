import cv2

det = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320, 320))
rec = cv2.FaceRecognizerSF.create("face_recognition_sface_2021dec.onnx", "")

def get_embedding(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError("Can't read " + path)

    # shrink big photos so the face is easier to find
    h, w = img.shape[:2]
    if w > 800:
        scale = 800 / w
        img = cv2.resize(img, (800, int(h * scale)))
        h, w = img.shape[:2]

    det.setInputSize((w, h))
    det.setScoreThreshold(0.7)      # a bit more forgiving than 0.9
    _, faces = det.detect(img)

    if faces is None:
        raise ValueError("No face found in " + path)

    aligned = rec.alignCrop(img, faces[0])
    return rec.feature(aligned)
a = get_embedding("me1.jpeg")
b = get_embedding("me2.jpeg")
c = get_embedding("friend.jpeg")

print("me vs me:    ", rec.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))
print("me vs friend:", rec.match(a, c, cv2.FaceRecognizerSF_FR_COSINE))