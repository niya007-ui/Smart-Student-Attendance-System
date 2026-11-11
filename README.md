# Smart-Student-Attendance-System
ace Recognition Based Student Attendance System using OpenCV, YuNet and SFace
# Smart Student Attendance System

A face recognition attendance system built with Python and OpenCV. It uses a webcam to recognize enrolled students and automatically records their attendance in a CSV file.

**[View Demo Screenshots](https://github.com/niya007-ui/Smart-Student-Attendance-System/tree/main/demo)**

## How it works

1. **Detect:** YuNet detects faces in each webcam frame.
2. **Extract:** SFace converts each face into a 128-dimensional embedding (a numerical representation of facial features).
3. **Compare:** The embedding is compared against enrolled face embeddings using cosine similarity. A higher score means a closer match.
4. **Recognize:** If the best match meets the similarity threshold of 0.45, the person is identified. Otherwise, the face is labelled `Unknown`.
5. **Log:** Recognized students are recorded in `attendance.csv` with their name, date, and time. Each student is marked only once per run.

## Setup

1. Clone the repository and install dependencies:

   ```bash
   git clone https://github.com/niya007-ui/Smart-Student-Attendance-System.git
   cd Smart-Student-Attendance-System
   pip install -r requirements.txt
   ```

2. Download the two pretrained models from [OpenCV Zoo](https://github.com/opencv/opencv_zoo) and place them in the project folder:
   - `face_detection_yunet_2023mar.onnx`
   - `face_recognition_sface_2021dec.onnx`

3. Create a `known/` folder with a separate folder for each student. Add 3–5 clear face photos per student.

   ```text
   known/
       alice/
           photo1.jpg
           photo2.jpg
       bob/
           photo1.jpg
   ```

4. Run the program:

   ```bash
   python recognize.py
   ```

   Press `q` to quit. Attendance is saved to `attendance.csv`.

## Limitations

- Recognition depends on lighting, camera quality, and face angle. Clear, well-lit photos generally give better results.
- **False matches:** During testing, an unknown person was matched with an enrolled student at a threshold of 0.36. Increasing it to 0.45 addressed that particular case, but higher thresholds can also reject genuine matches.
- The system does not include liveness detection, so a photograph or phone screen may be mistaken for a real person.
- Face images are personal data, so the `known/` folder is excluded from this repository.

## Planned Improvements

- Average the top three similarity scores per person instead of using only the best match.
- Require consistent recognition across multiple frames before marking attendance.
- Add student enrollment through the webcam.
- Build a Streamlit dashboard to view attendance records.

## Tech

Python, OpenCV (YuNet, SFace), NumPy, ONNX.

[GitHub](https://github.com/niya007-ui)
