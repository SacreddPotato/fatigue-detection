import os
from flask import Flask, render_template, Response, request, redirect, url_for
from werkzeug.utils import secure_filename
from camera import VideoCamera

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    source = request.args.get('source', '0')
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            return redirect(url_for('index', source=path))
        if 'url_link' in request.form and request.form['url_link'] != '':
            return redirect(url_for('index', source=request.form['url_link']))
    return render_template('index.html', source=source)

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else: break

@app.route('/video_feed')
def video_feed():
    source = request.args.get('source', '0')
    if source.isdigit(): source = int(source)
    return Response(gen(VideoCamera(source=source)), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)