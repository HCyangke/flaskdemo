#coding=utf-8
# from datetime import timedelta
from datetime import timedelta
from action_recognition import GesterRecognition
import argparse
from flask import Flask, render_template, Response, jsonify, request, session, url_for, redirect
from camera import VideoCamera
import cv2
app = Flask(__name__)

app.secret_key = "fM3PEZwSRcbLkk2Ew82yZFffdAYsNgOddWoANdQo/U3VLZ/qNsOKLsQPYXDPon2t"
app.permanent_session_lifetime = timedelta(days=7)

video_camera = None
global_frame = None
work = None
def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_queue',
        action='store_true',
        help='If true, random choice is not performed.')
    parser.set_defaults(use_queue=False)
    args = parser.parse_args()

    return args
# # 主页
# @app.route('/')
# def index():
#     username = session.get("username")
#     if not username:
#         return redirect(url_for("login"))
#     return render_template("index.html")


# @app.route("/login", methods=["GET", "POST"])
# def login():
#     username = session.get("username")

#     if username:
#         return redirect(url_for("index"))

#     if request.method == "GET":
#         return render_template("login.html")
#     username = request.form.get("username")
#     password = request.form.get("password")
#     if not all([username, password]):
#         return render_template("login.html", errmsg="参数不足")

#     if not username == "admin" and password == "admin":
#         return render_template("login.html", errmsg="用户名或密码错误")
#     session["username"] = username

#     return redirect(url_for("index"))


# @app.route("/logout")
# def logout():
#     session.pop("username", None)
#     return redirect(url_for("login"))


# @app.route('/record_status', methods=['POST'])
# def record_status():
#     global video_camera
#     if video_camera == None:
#         video_camera = VideoCamera()

#     json = request.get_json()

#     status = json['status']

#     if status == "true":
#         video_camera.start_record()
#         return jsonify(result="started")
#     else:
#         video_camera.stop_record()
#         return jsonify(result="stopped")


def video_stream():
    global video_camera
    global global_frame

    if video_camera == None:
        video_camera = VideoCamera(work)

    while True:
        frame = video_camera.get_frame()

        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')


@app.route('/')#video_viewer
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    opts=parse_opts()
    if work == None:
        work=GesterRecognition(opts.use_queue)
    app.run(host='0.0.0.0', port=8053, threaded=True, debug=True, use_reloader=False)
