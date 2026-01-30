from flask import Flask, render_template, Response, jsonify
import cv2
from camera import VideoCamera, music_rec  # Emotion Detection Camera & Music Recommendation
from drowsiness import detect_drowsiness  # Drowsiness Detection Function

app = Flask(__name__)

# Emotion-Based Playlist Variables
headings = ("Name", "Album", "Artist")
df1 = music_rec()
df1 = df1.head(15)

# Home Page with Buttons for Emotion & Drowsiness Detection
@app.route('/')
def home():
    return render_template('home.html')  # New home page with selection buttons

# Emotion-Based Playlist Page
@app.route('/emotion')
def emotion():
    return render_template('index.html', headings=headings, data=df1)

# Function to Stream Emotion-Based Video Feed
def gen_emotion(camera):
    while True:
        global df1
        frame, df1 = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route to Provide Emotion-Based Video Stream
@app.route('/video_feed')
def video_feed():
    return Response(gen_emotion(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to Return Emotion-Based Playlist JSON Data
@app.route('/t')
def gen_table():
    return df1.to_json(orient='records')

# Drowsiness Detection Page
@app.route('/drowsiness')
def drowsiness():
    return render_template('drowsiness.html')

# Function to Stream Drowsiness Detection Video Feed
def gen_drowsiness():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_drowsiness(frame)  # Apply drowsiness detection function
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to Provide Drowsiness Detection Video Stream
@app.route('/drowsiness_feed')
def drowsiness_feed():
    return Response(gen_drowsiness(), mimetype='multipart/x-mixed-replace; boundary=frame')

#Run Flask App
if __name__ == '__main__':
    app.debug = True
    app.run()
