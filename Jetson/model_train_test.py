import torch
import logging
import argparse
import json
from fl_client import Client
from flask import Flask, request

app = Flask(__name__)
client_instance = None

@app.route('/upload_data', methods=['POST'])
def upload_data():
    global client_instance
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    filename = 'received_data.pth'
    file_path = f"./{filename}"
    file.save(file_path)
    dataset = torch.load(file_path)
    if client_instance is None :
        client_instance = Client(dataset)
    return "File successfully uploaded", 200

@app.route('/upload_config', methods=['POST'])
def upload_config():
    global client_instance
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    filename = 'received_args.json'
    file.save(filename) 

    with open(filename, 'r') as file:
        args_dict = json.load(file)

    fl_config = argparse.Namespace(**args_dict)
    # logging.info(fl_config.local_ep)
    client_instance.set_config(fl_config)
    # logging.info(client_instance.fl_config.local_ep)
    return "Config uploaded and Client instance created", 200

@app.route('/upload_model', methods=['POST'])
def upload_model():
    global client_instance
    if 'file' not in request.files:
        return "File not found in request", 400

    file = request.files['file']
    filename = 'received_model.pth'
    file_path = f"./{filename}"
    file.save(file_path)
    model = torch.load(file_path)
    # logging.info(request.remote_addr)

    client_instance.set_model(model)
    client_instance.set_train()
    return "Model uploaded and processed", 200


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    app.run(host='0.0.0.0', port=5000)
