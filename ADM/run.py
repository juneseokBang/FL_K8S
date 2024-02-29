import logging
from flask import Flask, request, redirect, jsonify
import server
import os
import threading
from options import args_parser

app = Flask(__name__)
fl_server = None

@app.route('/')
def index():
    # IP 주소를 키로, 파일 이름을 값으로 가지는 딕셔너리
    global fl_server

    args = args_parser()

    file_logger = logging.getLogger("File")
    file_logger.setLevel(logging.DEBUG)
    formatter1 = logging.Formatter('%(message)s')
    log_dir = "logs"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_f_name = log_dir + '/test.log'
    file_handler = logging.FileHandler(log_f_name)
    
    file_handler.setFormatter(formatter1)
    file_logger.addHandler(file_handler)

    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')

    fl_server = server.Server(args, file_logger)
    # fl_server = server.Server(args)
    fl_server.boot()
    # exit(1)
    fl_server.run()

    return 'Finish !'



@app.route('/upload_data', methods=['POST'])
def upload_data():
    global completed_uploads
    if 'file' not in request.files:
        return "No file part", 400

    pod_ip = request.headers.get('X-Pod-IP')
    file = request.files['file']
    filename = f'{pod_ip}-received_data.pth'
    file_path = f"./client/{filename}"
    file.save(file_path)

    return "Local Model uploaded", 200



if __name__ == '__main__':
    # fl_run.main()
    app.run(host='0.0.0.0', port=5000)