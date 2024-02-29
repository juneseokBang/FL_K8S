import logging
import server
import os
from options import args_parser

def main():
    args = args_parser()

    file_logger = logging.getLogger("File")
    file_logger.setLevel(logging.DEBUG)
    formatter1 = logging.Formatter('%(message)s')
    log_dir = "logs"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # log_dir = log_dir  + '/'
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    
    # run_num = 0
    # current_num_files = next(os.walk(log_dir))[2]
    # run_num = len(current_num_files)
    # log_f_name = log_dir + str(run_num) + '.log'
    # log_f_name = log_dir + '/test_al1.log'# 망한거
    # log_f_name = log_dir + '/test_NonIID.log'
    # log_f_name = log_dir + '/test_noniid_al1.log'
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

if __name__ == "__main__":
    main()