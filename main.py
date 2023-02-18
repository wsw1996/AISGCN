import argparse
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np
from data import *
import pickle
import torch.distributions.multivariate_normal as torchdist
from model import *
from metrics import *
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("run_wswmodel_1_4")

# 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# 自定义目录存放日志文件
log_path = './wswLogs/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
# 日志文件名按照程序运行时间设置
log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
# 记录正常的 print 信息
sys.stdout = Logger(log_file_name)
# 记录 traceback 异常信息
sys.stderr = Logger(log_file_name)


def get_dirs(args):
	net_dir = str(args.model)+'/'
	data_dir = 'data/%02d/'%(args.zone)
	# print(data_dir)
	assert(os.path.isdir(data_dir))
	if not os.path.isdir(net_dir):
		os.makedirs(net_dir)
	return net_dir, data_dir


def load_data(data_dir, args):
    dataset_dir = 'data/%02d/' % (args.zone)
    train_dir = dataset_dir + 'train/'
    val_dir = dataset_dir + 'val/'
    test_dir = dataset_dir + 'test/'

    if args.split_data or len(os.listdir(train_dir)) == 0:
        data = trajectory_dataset(data_dir, args.obs_length, args.pred_length, args.feature_size)
        data_size = len(data)
        print("data size total--::", data_size)
        valid_size = int(np.floor(0.1 * data_size))
        test_size = valid_size
        train_size = data_size - valid_size - test_size
        traindataset, validdataset, testdataset = random_split(data, [train_size, valid_size, test_size])
        torch.save(traindataset, train_dir + "%02d_%02d.pt" % (args.obs_length, args.pred_length))
        torch.save(validdataset, val_dir + "%02d_%02d.pt" % (args.obs_length, args.pred_length))
        torch.save(testdataset, test_dir + "%02d_%02d.pt" % (args.obs_length, args.pred_length))
    else:
        traindataset = torch.load(train_dir + "%02d_%02d.pt" % (args.obs_length, args.pred_length))
        validdataset = torch.load(val_dir + "%02d_%02d.pt" % (args.obs_length, args.pred_length))
        testdataset = torch.load(test_dir + "%02d_%02d.pt" % (args.obs_length, args.pred_length))
    return traindataset, validdataset, testdataset

def graph_loss(V_pred,V_target):
    return bivariate_loss(V_pred,V_target)

def train(epoch):
    global metrics, loader_train
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_train):
        batch_count += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        optimizer.zero_grad()
        # Forward
        # V_obs = batch,seq,node,feat    ----- 128，8，57，2
        # V_obs_tmp = batch,feat,seq,node
        # V起初是[8,57,2] 高8层 57行 2列， 现在是高2层 8行 57列
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)  ##  v现在是128*高2层 8行 57列, A是邻接矩阵128*8*57*57
        # V_obs_tmp[batch_size,input_channel,obs_seq_len,行人数]
        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())  # ---------#经过模型得到V:128*5*12*57   A：8,57,57   (邻接矩阵维度一直没有改变)
        # 128*5*12*57

        V_pred = V_pred.permute(0, 2, 3, 1)  # 128*5*12*57————>[128,12,57,5]

        V_tr = V_tr.squeeze()  # [128,8,57,2]
        A_tr = A_tr.squeeze()  # [8,57,57]
        V_pred = V_pred.squeeze()  # [128,12,57,5]

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr)  # 越小越好！！概率密度函数的负对数

            if is_fst_loss:  # 开始训练是true
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            loss.backward()

            if args.clip_grad is not None:  # 默认是None
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            # Metrics
            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    metrics['train_loss'].append(loss_batch / batch_count)


def vald(epoch):
    global metrics, loader_val, constant_metrics
    model.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_val):
        batch_count += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())

        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            # Metrics
            loss_batch += loss.item()
            print('VALD:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    metrics['val_loss'].append(loss_batch / batch_count)

    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')  # OK


def test(KSTEPS=20):
    global loader_test, model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step = 0
    for batch in loader_test:
        step += 1
        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        num_of_objs = obs_traj_rel.shape[1]  # 行人数N

        # Forward
        # V_obs = batch,seq,node,feat
        # V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        # print(V_pred.shape)
        # torch.Size([1, 5, 12, 2])    [1,5,12,57]
        # torch.Size([12, 2, 5])
        V_pred = V_pred.permute(0, 2, 3, 1)  # 【1，12，2，5】  [1,12,57,5]
        # torch.Size([1, 12, 2, 5])>>seq,node,feat
        # V_pred= torch.rand_like(V_tr).cuda()

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()  # [12,57,5]
        num_of_objs = obs_traj_rel.shape[1]  ##[2785*57,2,8]    他这个指定的就是行人数
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]  # 这个维度应该是：预测长度、行人数、5
        # print(V_pred.shape)

        # For now I have my bi-variate parameters
        # normx =  V_pred[:,:,0:1]
        # normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:, :, 2])  # sx
        sy = torch.exp(V_pred[:, :, 3])  # sy
        corr = torch.tanh(V_pred[:, :, 4])  # corr

        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).cuda()  # [:,:,2，2]
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]  # 包括V_pred[:,:,0]  和  V_pred[:,:,1]    [:,:,2]

        mvnormal = torchdist.MultivariateNormal(mean, cov)
        # MultivariateNormal(loc:torch.size([12,2,2]), covariance_matrix:torch.size([12,2,2,2)) )
        #

        ### Rel to abs
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len

        # Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                V_x[0, :, :].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                V_x[-1, :, :].copy())  # 得到的是一个三位的

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()

            # V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                       V_x[-1, :, :].copy())  # #[12,57,5]
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

            # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])
                obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    return ade_, fde_, raw_data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=True, action="store_true", help="train model")
    parser.add_argument("--test", default=False, action="store_true", help="evaluate model")
    parser.add_argument("--zone", type=int, default=11, help="UTM zone")
    parser.add_argument("--model", type=str, default='stgcnn',
                        choices=['spatial_temporal_model', 'spatial_model', 'temporal_model', 'vanilla_lstm'],
                        help="model type")
    parser.add_argument("--split_data", action="store_true", help="split data into train, valid, test")
    parser.add_argument('--feature_size', type=int, default=2, help="feature size")
    parser.add_argument('--obs_length', type=int, default=5, help="sequence length")
    parser.add_argument('--pred_length', type=int, default=5, help="prediction length")

    # Model specific parameters
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--n_stgcnn', type=int, default=1, help='Number of ST-GCNN layers')
    parser.add_argument('--n_txpcnn', type=int, default=4, help='Number of TXPCNN layers')
    parser.add_argument('--kernel_size', type=int, default=3)

    # Training specifc parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=250,
                        help='number of epochs default 250')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gadient clipping')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=150,
                        help='number of steps to drop the lr')
    parser.add_argument('--use_lrschd', action="store_true", default=True,
                        help='Use lr rate scheduler')
    parser.add_argument('--tag', default='wswmodel_1_4',
                        help='personal tag for the model ')

    args = parser.parse_args()
    print("-" * 50)
    print("Parameters:")
    for k, v in vars(args).items():
        print(k, ":", v)
    print("-" * 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using {}".format(device))
    print("init----")
    net_dir, data_dir = get_dirs(args)
    traindataset, validdataset, testdataset = load_data(data_dir, args)

    loader_train = DataLoader(traindataset, batch_size=1, shuffle=True, num_workers=0)
    loader_val = DataLoader(validdataset, batch_size=1, shuffle=False, num_workers=0)

    # Defining the model

    model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                          output_feat=args.output_size, seq_len=args.obs_length,
                          kernel_size=args.kernel_size, pred_seq_len=args.pred_length).cuda()

    # 开始的输入图形数据是batch_size,in_channel,obs_len,行人数
    # 经过模型最后变成 v:128*5*12*57   A：8,57,57 A只是帮助在图运算的过程中，所以经过gcn之后没有变化

    # Training settings

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if args.use_lrschd:  # 默认是false
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)

    checkpoint_dir = './checkpoint/' + args.tag + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    # Training
    metrics = {'train_loss': [], 'val_loss': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}

    if args.train:

        print('Training started ...')
        for epoch in range(args.num_epochs):
            train(epoch)
            vald(epoch)

            writer.add_scalar('trainloss', np.array(metrics['train_loss'])[epoch], epoch)
            writer.add_scalar('valloss', np.array(metrics['val_loss'])[epoch], epoch)

            if args.use_lrschd:
                scheduler.step()
            print('*' * 30)
            print('Epoch:', args.tag, ":", epoch)
            for k, v in metrics.items():
                if len(v) > 0:
                    print(k, v[-1])

            print(constant_metrics)
            print('*' * 30)

            with open(checkpoint_dir + 'metrics.pkl', 'wb') as fp:
                pickle.dump(metrics, fp)

            with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
                pickle.dump(constant_metrics, fp)

    if args.test:
        paths = ['./checkpoint/wswmodel_1_4']
        KSTEPS = 20

        for feta in range(len(paths)):
            ade_ls = []
            fde_ls = []
            path = paths[feta]
            exps = glob.glob(path)
            print('Model being tested are:', exps)

            for exp_path in exps:
                print("*" * 50)
                print("Evaluating model:", exp_path)

                model_path = exp_path + '/val_best.pth'
                args_path = exp_path + '/args.pkl'
                with open(args_path, 'rb') as f:
                    args = pickle.load(f)  # 反序列化对象，将文件中的数据解析为一个python对象。file中有read()接口和 readline() 接口

                stats = exp_path + '/constant_metrics.pkl'
                with open(stats, 'rb') as f:
                    cm = pickle.load(f)
                print("Stats:", cm)

            loader_test = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0)
            # Defining the model
            model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                                  output_feat=args.output_size, seq_len=args.obs_length,
                                  kernel_size=args.kernel_size, pred_seq_len=args.pred_length).cuda()
            model.load_state_dict(torch.load(model_path))

            ade_ = 999999
            fde_ = 999999
            print("Testing ....")
            ad, fd, raw_data_dic_ = test()
            ade_ = min(ade_, ad)
            fde_ = min(fde_, fd)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            print("ADE:{:.5f}, FDE:{:.5f}".format(ade_,fde_))






