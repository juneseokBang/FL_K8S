import os
import torch
import logging
import time
import utils
import random
from models import *
# from ADM.adm import *
from adm import *
from threading import Thread
# import ADM.fl_client as fl_client
import fl_client as fl_client
import copy
import updateModel
import dists
import json
import requests

from kubernetes import client, config

class Server(object):
    def __init__(self, args, file_logger=None):
        self.fl_config = args
        self.file_logger = file_logger

        # 쿠버네티스 설정 로드
        config.load_kube_config()

        # 쿠버네티스 API 클라이언트 생성
        v1 = client.CoreV1Api()

        # 'default' 네임스페이스의 파드 리스트 조회
        self.ret = v1.list_namespaced_pod(namespace='default')

    def boot(self):
        logging.info('Booting server...')

        self.load_model()
        self.make_clients()

    def load_model(self):
        IID = self.fl_config.IID
        fl_config = self.fl_config
        howto = self.fl_config.loader
        dataset = self.fl_config.dataset
        logging.info('dataset: {}'.format(dataset))

        generator = utils.get_data(dataset)
        generator.load_data() # 데이터 생성
        data = generator.generate()
        labels = generator.labels

        logging.info('Dataset size: {}'.format(
            sum([len(x) for x in [data[label] for label in labels]])))
        logging.info('Labels ({}): {}'.format(
            len(labels), labels))
        
        if not IID :
            if howto == "shard":
                self.loader = utils.ShardLoader(fl_config, generator)
            elif howto == "bias":
                self.loader = utils.BiasLoader(fl_config, generator)

        else: # IID
            self.loader = utils.Loader(fl_config, generator)

        self.model = get_model(dataset)
        
    
    def make_clients(self):
        IID = self.fl_config.IID
        labels = self.loader.labels
        hybrid = self.fl_config.hybrid
        num_clients = self.fl_config.num_clients

        clients = []

        for idx, i in enumerate(self.ret.items):
            new_client = fl_client.Client(idx, i.status.pod_ip, copy.deepcopy(self.model))
            # clients.append(new_client)
            if not IID:
                if self.fl_config.loader == "shard":
                    shard = self.fl_config.shard
                    new_client.set_shard(shard)
                elif self.fl_config.loader == "bias":
                    dist = dists.uniform(num_clients, len(labels))
                    self.bias = 0.8 # 0.8
                    pref = random.choices(labels, weights=dist)[0]

                    new_client.set_bias(pref, self.bias)

            clients.append(new_client)


        '''
        로컬용 FL
        for client_id in range(num_clients):
            new_client = fl_client.Client(client_id)
            
            if not IID:
                if self.fl_config.loader == "shard":
                    shard = self.fl_config.shard
                    new_client.set_shard(shard)
                elif self.fl_config.loader == "bias":
                    dist = dists.uniform(num_clients, len(labels))
                    self.bias = 0.8 # 0.8
                    pref = random.choices(labels, weights=dist)[0]

                    new_client.set_bias(pref, self.bias)

            
            clients.append(new_client)
        '''

        logging.info('Total clients: {}'.format(len(clients)))

        if not IID and self.fl_config.loader == "bias":
            logging.info('Label distribution: {}'.format(
                [[client.pref for client in clients].count(label) for label in labels]))

        if not IID:
            if not hybrid:
                if self.fl_config.loader == "shard":
                    self.loader.create_shards()
                    [self.set_client_data(client) for client in clients]
                else :
                    [self.set_client_data(client) for client in clients]
            else:
                if self.fl_config.loader == "shard":
                    m = max(int(self.fl_config.IID_ratio * self.fl_config.num_clients), 1)
                    # IID_clients = [client for client in random.sample(
                    #     clients, m)]
                    IID_clients = [client for client in clients[:m]]
                    Non_clients = set(clients) - set(IID_clients)
                    # print(len(IID_clients))
                    # print(len(Non_clients))
                    self.loader.hybird_shards(len(IID_clients), self.fl_config.shard)
                    # [self.set_client_IID(client) for client in clients[:int(num_clients/2)]]
                    # [self.set_client_Non_IID(client) for client in clients[int(num_clients/2):]]
                    [self.set_client_IID(client) for client in IID_clients]
                    [self.set_client_Non_IID(client) for client in Non_clients]

                else :
                    pass
        else: # IID
            [self.set_client_data(client) for client in clients]

        self.clients = clients
        logging.info('Clients: {}'.format(self.clients))

        self.number_sample_list = []
        for c in self.clients:
            self.number_sample_list.append(len(c.data))
        

    def set_client_data(self, client):
        IID = self.fl_config.IID
        loader = self.fl_config.loader
        number_of_sample = 1000
        # number_of_sample = 2500
        if not IID:
            if loader == "shard":
                data = self.loader.get_partition()
            elif loader == "bias":
                data = self.loader.get_partition(number_of_sample, client.pref, self.bias)

        else: # IID
            data = self.loader.get_partition(number_of_sample)
            # print(type(data))
            # print(type(data[0]))
            # exit()
        torch.save(data, 'data.pth')
        
        # 여기에서 데이터를 pod로 보내는 코드 작성
        url = f'http://{client.client_ip}:5000/upload_data'  # 가정: 파드 내 Flask 앱이 '/upload' 엔드포인트에서 파일을 받음
        files = {'file': open('data.pth', 'rb')}
        response = requests.post(url, files=files)
        print(response.text)
        # Send data to client
        client.set_data(data)

    def set_client_IID(self, client):
        IID = True
        data = self.loader.get_hybrid_partition(IID)
        client.set_data(data)

    def set_client_Non_IID(self, client):
        IID = False
        data = self.loader.get_hybrid_partition(IID)
        client.set_data(data)

    def model_send_train(self, client):
        url = f'http://{client.client_ip}:5000/upload_model'
        files = {'file': open('model.pth', 'rb')}
        response = requests.post(url, files=files)
        print(response.text)

    def run(self):
        rounds = self.fl_config.rounds
        target_accuracy = self.fl_config.target_accuracy
        self.optimal_v_n = [1.0 for _ in range(self.fl_config.num_clients)]
        logging.info('Training: {} rounds\n'.format(rounds))

        for round in range(rounds):
            logging.info('**** Round {}/{} ****'.format(round+1, rounds))
            self.round(round)
            
        

    def round(self, round):
        sample_clients = self.selection()
        self.sample_clients = sample_clients
        self.configuration(sample_clients)
        self.adm_configuration(sample_clients)

        # ADM Algorithm 1
        if round > 0:
            self.optimal_v_n, sol_list, optimal_t = block_coordinate_descent(self.parameters,
                                                                        round,
                                                                        self.parameters["t"]) # sol_list는 objective function 값
            
            self.parameters["t"] = optimal_t

        logging.info('v_n: {}'.format(self.optimal_v_n))


        for client in sample_clients:
            client.adm_algorithm_1(self.optimal_v_n[client.client_id])
            client.pt_data_distribution()
            # client.train()

        self.model.to(torch.device('cpu'))
        torch.save(self.model.state_dict(), 'model.pth')
        # self.model_send_train(sample_clients[0])
        threads = [Thread(target=client.model_send_train()) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]
        # logging.info("D_n: {}".format(self.parameters["D_n"]))

        # 여기까지가 로컬 모델 학습.
        # pod의 local model 파일이 생김
        ####################################
        
        reports = []
        for file_name in os.listdir('./client'):
            if file_name.endswith('.pth'):  # .pth 확장자를 가진 파일만 처리
                # 파일 이름에서 IP 주소 추출
                source_ip = file_name.split('-')[0]
                file_path = f'./client/{file_name}'
                
                # 해당 IP 주소를 가진 클라이언트 찾기
                for client in self.sample_clients:
                    if client.client_ip == source_ip:
                        reports.append(client.get_report(file_path))  # 파일 경로를 weight 속성에 저장
                        break

        # reports = [client.get_report() for client in self.sample_clients]
        logging.info('Reports recieved: {}'.format(len(reports)))
        assert len(reports) == len(self.sample_clients)
    
        logging.info('Aggregating updates')
        updated_weights = self.aggregation(reports)
        updateModel.load_weights(self.model, updated_weights)

        testset = self.loader.get_testset()
        batch_size = 1000
        testloader = updateModel.get_testloader(testset, batch_size)
        accuracy = updateModel.test(self.model, testloader)
        logging.info('Global model accuracy: {:.2f}%'.format(100 * accuracy))
        # self.file_logger.info("{0}_{1:.2f}_{2:.3f}_{3}".format(round, 100*accuracy, time.time()-start_time_epochs, self.optimal_v_n))
        self.file_logger.info("{0}_{1}".format(round, 100*accuracy))
        print()

    def selection(self):
        m = max(int(self.fl_config.frac * self.fl_config.num_clients), 1)
        # sample_clients = [client for client in random.sample(
        #     self.clients, m)]
        sample_clients = [client for client in self.clients[:m]]
        
        return sample_clients
    
    def adm_configuration(self, sample_clients):
        number_sample_list=[]
        for client in sample_clients:
            number_sample_list.append(len(client.data))
        constant_parameters = {'sigma' : 0.9, 'D_n': number_sample_list, 'Gamma': 0.4, 'local_iter': 10, 'c_n': 30,
                #   'frequency_n_GHz' : [1.5, 2, 2.5, 3], 
                  'frequency_n_GHz' : [3000], 
                  'weight_size_n_kbit' : 100,
                  'number_of_clients' : self.fl_config.num_clients, 'bandwidth_MHz' : 1, 'channel_gain_n': 1, 
                #   'transmission_power_n' : [0.2, 0.5, 1], 
                  'transmission_power_n' : [1], 
                  'noise_W' : 1e-12,
                  't':500}
        
        self.parameters=init_param_hetero(constant_parameters, self.fl_config.num_clients, constant_parameters["t"])
        

    def configuration(self, sample_clients):
        hybrid = self.fl_config.hybrid
        fl_config = self.fl_config

        args_dict = vars(fl_config)
        json_str = json.dumps(args_dict)

        with open('args.json', 'w') as json_file:
            json_file.write(json_str)

        for client in sample_clients:
            # 여기에서 데이터를 pod로 보내는 코드 작성
            url = f'http://{client.client_ip}:5000/upload_config'  # 가정: 파드 내 Flask 앱이 '/upload' 엔드포인트에서 파일을 받음
            files = {'file': open('args.json', 'rb')}
            response = requests.post(url, files=files)
            print(response.text)
            # client.configure(fl_config, model=copy.deepcopy(self.model))

    def extract_client_updates(self,reports):
        baseline_weights = updateModel.extract_weights(self.model)

        # Extract weights from reports
        weights = [report.weights for report in reports]

        # Calculate updates from weights
        updates = []
        for weight in weights:
            update = []
            for i, (name, weight) in enumerate(weight):
                
                bl_name, baseline = baseline_weights[i]

                # Ensure correct weight is being updated
                assert name == bl_name

                # Calculate update
                delta = weight - baseline
                update.append((name, delta))
            # print(update)
            # exit(1)
            updates.append(update)
        return updates


    def aggregation(self, reports):
        return self.fedavg(reports)
    
    def fedavg(self,reports):
        # num = int((self.fl_config.num_clients / 2))
        # contri_action = [self.fl_config.contri / num] * num
        # contri_action_non = [(1 - self.fl_config.contri) / num] * num
        # contri_action.extend(contri_action_non)
        
        updates = self.extract_client_updates(reports)

        # Extract total number of samples
        total_samples = sum([report.num_samples for report in reports])
        
        # Perform weighted averaging
        avg_update = [torch.zeros(x.size()) for _, x in updates[0]]
        for i, update in enumerate(updates):
            num_samples = reports[i].num_samples
            # if i >= 10:
            #     break
            # contri = contri_action[reports[i].client_id]
            # logging.info("기여도: {}".format(num_samples / total_samples))

            for j, (_, delta) in enumerate(update):
                # Use weighted average by number of samples
                avg_update[j] += delta * (num_samples / total_samples)
                # avg_update[j] += delta * contri

        # Extract baseline model weights
        baseline_weights = updateModel.extract_weights(self.model)

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights
