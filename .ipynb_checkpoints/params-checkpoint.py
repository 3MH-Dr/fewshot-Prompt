import torch
import argparse


def get_params():
    args = argparse.ArgumentParser()
    args.add_argument("-seed", "--seed", default=42, type=int)
    args.add_argument("-form", "--data_form", default="Pre-Train", type=str)  # ["Pre-Train", "In-Train", "Discard"]

    # dataset setting
    args.add_argument("-data", "--dataset", default="NELL-One", type=str)  # ["NELL-One", "Wiki-One"]
    args.add_argument("-path", "--data_path", default="/home/featurize/data/Datasets/NELL", type=str)  # ["./NELL", "./Wiki"]
    args.add_argument("-if", "--is_shuffle", default=False, type=bool)  # whether shuffle the training relations

    # dataloader setting
    args.add_argument("-bfew", "--base_classes_few", default=3, type=int)  # base support num
    args.add_argument("-bnq", "--base_classes_num_query", default=3, type=int)
    args.add_argument("-few", "--few", default=3, type=int)  # novel support num
    args.add_argument("-nq", "--num_query", default=3, type=int)
    args.add_argument("-br", "--base_classes_relation", default=30, type=int)
    args.add_argument("-bs", "--batch_size", default=3, type=int)  # novel relations num
    args.add_argument("-nt", "--num_tasks", default=8, type=int)

    # model setting
    args.add_argument("-dim", "--embed_dim", default=100, type=int)
    args.add_argument("-p", "--dropout_p", default=0.5, type=float)
    args.add_argument("-b", "--beta", default=5, type=float)
    args.add_argument("-m", "--margin", default=1, type=float)

    # Prompt setting
    args.add_argument('--l2p', default=False, type=bool)
    args.add_argument('--coda', default=False, type=bool)
    args.add_argument('--moe', default=False, type=bool)
    
    args.add_argument('--size', default=32, type=int)  # top_k * num_tasks
    args.add_argument('-len', '--length', default=10, type=int)
    args.add_argument('--top_k', default=4, type=int)
    args.add_argument('--embedding_key', default='mean', type=str)
    args.add_argument('--initializer', default='uniform', type=str, )
    args.add_argument('--prompt_key_init', default='uniform', type=str)
    args.add_argument('--use_prompt_mask', default=True, type=bool)
    args.add_argument('-sp', '--shared_prompt_pool', default=False, type=bool)
    args.add_argument('--shared_prompt_key', default=False, type=bool)
    args.add_argument('--batchwise_prompt', default=True, type=bool)
    args.add_argument('--predefined_key', default='', type=str)
    args.add_argument('--pull_constraint', default=True)
    args.add_argument('--pull_constraint_coeff', default=0.5, type=float)

    # training setting
    args.add_argument("-gpu", "--device", default=0, type=int)
    args.add_argument("-lr", "--learning_rate", default=0.001, type=float)
    args.add_argument("-epo", "--epoch", nargs='+', required=True, type=int)  
    args.add_argument("-es_p", "--early_stopping_patience", default=1000, type=int)  # base patience
    args.add_argument("-es_np", "--early_NOVEL_stopping_patience", default=50, type=int)  # [50 for NELL, 300 for Wiki]
    args.add_argument("-prt_epo", "--print_epoch", default=100, type=int)
    args.add_argument("-ckpt_epo", "--checkpoint_epoch", default=10000, type=int)
    args.add_argument("-eval_epo", "--eval_epoch", nargs='+', required=True, type=int)
    args.add_argument("-pt", "--is_prompt_tuning", default=False, type=bool)

    # other setting
    args.add_argument("-spath", "--save_path", type=str)
    args.add_argument("-metric", "--metric", default="MRR", choices=["MRR", "Hits@10", "Hits@5", "Hits@1"])
    args.add_argument("-prefix", "--prefix", default="exp1", type=str)
    args.add_argument("-step", "--step", default="train", type=str, choices=['train', 'test', 'dev'])
    args.add_argument("-log_dir", "--log_dir", default="log", type=str)
    args.add_argument("-state_dir", "--state_dir", default="state", type=str)
    args.add_argument("-eval_ckpt", "--eval_ckpt", default=None, type=str)
    args.add_argument("-eval_by_rel", "--eval_by_rel", default=False, type=bool)
    args.add_argument("-idx", "--rel_index", default=30, type=int)

    args = args.parse_args()
    params = {}
    for k, v in vars(args).items():
        params[k] = v

    if args.dataset == 'NELL-One':
        params['embed_dim'] = 100
    elif args.dataset == 'Wiki-One':
        params['embed_dim'] = 50

    params['device'] = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    return params


data_dir = {
    'train_tasks': '/continual_train_tasks.json',
    'test_tasks': '/test_tasks.json',
    'dev_tasks': '/con_base100_n100_dev_tasks.json',  # continual testing evaluation
    'few_shot_dev_tasks': '/dev_tasks.json',

    'rel2candidates': '/rel2candidates.json',
    'e1rel_e2': '/e1rel_e2.json',
    'ent2ids': '/ent2ids',
    'ent2vec': '/ent2vec.npy',
}
