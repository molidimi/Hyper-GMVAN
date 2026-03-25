# coding=utf-8
import os.path
import torch
# 兼容包内/脚本两种运行方式
try:
    from . import gol  # 全局配置模块，包含超参数和全局变量设置
except Exception:
    import gol
import numpy as np
from torch.utils.data import DataLoader  # 张量计算和数据加载
from pprint import pformat
from datetime import datetime
import glob

try:
    from .model import HyperGMVAN as DiffDGMN
    from .dataset import GraphData, collate_eval, collate_edge, getDatasets, NDCG_at_k, ACC_at_k, MRR
except Exception:
    from model import HyperGMVAN as DiffDGMN
    from dataset import GraphData, collate_eval, collate_edge, getDatasets, NDCG_at_k, ACC_at_k, MRR


def get_next_log_number(log_dir, dataset):
    # 获取指定目录下特定数据集的所有日志文件
    pattern = os.path.join(log_dir, f'GT_kan_{dataset}_run_*.log')
    existing_logs = glob.glob(pattern)

    # 如果没有现有日志文件，从1开始
    if not existing_logs:
        return 1

    # 提取现有日志文件的编号
    numbers = []
    for log in existing_logs:
        try:
            num = int(log.split('_run_')[-1].split('_')[0])
            numbers.append(num)
        except:
            continue

    # 返回最大编号+1
    return max(numbers) + 1 if numbers else 1


def setup_logging():
    # 确保在本地创建日志目录
    local_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_logs', 'GT_kan', gol.dataset)
    os.makedirs(local_log_dir, exist_ok=True)

    # 获取下一个运行编号
    run_number = get_next_log_number(local_log_dir, gol.dataset)

    # 生成日志文件名：数据集名_run_编号_时间戳.log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = f'GT_kan_{gol.dataset}_run_{run_number:03d}_{timestamp}.log'
    log_path = os.path.join(local_log_dir, log_name)

    # 配置日志系统
    import logging
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%m/%d %H:%M')

    # 文件处理器 - 写入本地文件
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # 获取根日志记录器并配置
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 添加新的处理器
    root_logger.addHandler(file_handler)

    # 同时输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 记录实验基本信息
    logging.info(f'Experiment started - Run #{run_number:03d}')
    logging.info(f'Local log file: {log_path}')
    logging.info(f'Dataset: {gol.dataset}')
    logging.info(f'Device: {gol.device}')
    logging.info(f'Configuration: {gol.conf}')

    return log_path


#                   数据集类，   DataLoader组批函数            数据加载函数    评价指标函数

# 断点保存/加载（兼容老/新命名）
def save_checkpoint(model, optimizer, epoch, best_val_epoch, best_val_score, ckpt_epoch_tmpl, ckpt_latest_path):
    try:
        state = {
            'epoch': int(epoch),
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_epoch': int(best_val_epoch),
            'best_val_score': float(best_val_score),
            'rng_cpu': torch.get_rng_state(),
            'rng_np': np.random.get_state(),
            'rng_py': __import__('random').getstate(),
        }
        # 兼容旧键名
        state['model_state'] = state['model']
        state['optimizer_state'] = state['optimizer']
        state['torch_rng_state'] = state['rng_cpu']
        state['numpy_rng_state'] = state['rng_np']
        state['python_rng_state'] = state['rng_py']
        if torch.cuda.is_available():
            state['rng_cuda'] = torch.cuda.get_rng_state_all()
            state['cuda_rng_state_all'] = state['rng_cuda']
        epoch_path = ckpt_epoch_tmpl.format(epoch=epoch)
        torch.save(state, epoch_path)
        torch.save(state, ckpt_latest_path)
        gol.pLog(f'Saved checkpoint to: {epoch_path} and latest: {ckpt_latest_path}')
    except Exception as e:
        gol.pLog(f'Failed to save checkpoint: {str(e)}')


def load_checkpoint(path, model, optimizer=None, map_location=None):
    # 优先尝试安全模式；若失败则回退到常规加载（兼容旧检查点）
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as _e:
        gol.pLog(f'Weights-only load failed, fallback to unsafe load (trusted local ckpt). Reason: {str(_e)}')
        ckpt = torch.load(path, map_location=map_location)

    # 解析多种格式
    model_state = None
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            model_state = ckpt['model']
        elif 'model_state' in ckpt:
            model_state = ckpt['model_state']
        elif 'state_dict' in ckpt:
            model_state = ckpt['state_dict']
        else:
            try:
                if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                    model_state = ckpt
            except Exception:
                pass
    else:
        model_state = ckpt

    if model_state is None:
        raise KeyError('No model state found in checkpoint')

    model.load_state_dict(model_state)

    if optimizer is not None and isinstance(ckpt, dict):
        opt_state = ckpt.get('optimizer', ckpt.get('optimizer_state', None))
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)

    try:
        if isinstance(ckpt, dict):
            torch_rng = ckpt.get('rng_cpu', ckpt.get('torch_rng_state', None))
            if torch_rng is not None:
                torch.set_rng_state(torch_rng)
            if torch.cuda.is_available():
                cuda_rng = ckpt.get('rng_cuda', ckpt.get('cuda_rng_state_all', None))
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
            np_rng = ckpt.get('rng_np', ckpt.get('numpy_rng_state', None))
            if np_rng is not None:
                np.random.set_state(np_rng)
            py_rng = ckpt.get('rng_py', ckpt.get('python_rng_state', None))
            if py_rng is not None:
                __import__('random').setstate(py_rng)
    except Exception as e:
        gol.pLog(f'Warning: failed to restore RNG state: {str(e)}')

    start_epoch = 0
    best_val_epoch = 0
    best_val_score = 0.0
    if isinstance(ckpt, dict):
        start_epoch = int(ckpt.get('epoch', ckpt.get('start_epoch', -1))) + 1
        best_val_epoch = int(ckpt.get('best_val_epoch', 0))
        best_val_score = float(ckpt.get('best_val_score', 0.0))
    return start_epoch, best_val_epoch, best_val_score

# 定义模型评估函数，计算在评估集上的推荐指标
def eval_model(model: DiffDGMN, eval_set: GraphData):
    Ks = [1, 2, 5, 10, 20]  # 记录在TOP-k下的表现
    result = {'Recall': np.zeros(len(Ks)), 'NDCG': np.zeros(len(Ks)),
              'MRR': 0., 'ACC': np.zeros(len(Ks))}  # 增加 ACC@K 输出
    eval_loader = DataLoader(eval_set, batch_size=gol.TEST_BATCH_SZ, shuffle=True, collate_fn=collate_eval)  # 评估数据加载器
    #   评估数据加载器                         测试批大小                       随机打乱用户顺序，使用自定义的聚合函数将批数据整理成张量
    #   此处的DataLoader数据较杂，包含用户ID、候选POI标签、序列图结构等

    with torch.no_grad():
        model.eval()  # 设置模型为评估模式
        tot_cnt = 0  # 累计评估总用户数
        for idx, batch in enumerate(eval_loader):  # 批量评估循环
            u, pos_list, exclude_mask, seqs, seq_graph, cur_time = batch
            # 将必要字段挂到图上，保持与 forward 的输入一致
            try:
                seq_graph.user = u.to(seq_graph.x.device, non_blocking=True)
            except Exception:
                # 若无 x 字段，退回到 gol.device
                seq_graph.user = u.to(gol.device, non_blocking=True)

            if isinstance(seqs, (list, tuple)):
                seq_graph.seq = torch.cat(list(seqs), dim=0).to(seq_graph.user.device, non_blocking=True)
            else:
                seq_graph.seq = seqs.to(seq_graph.user.device, non_blocking=True)

            # 前向计算，输出每个用户对所有POI的预测评分向量
            item_score = model(seq_graph)
           
            # 处理序列图中的特殊值：seq_graph.mean_interv[...] = ...用平均值填充了序列图中mean_interv为NaN的项

            item_score[exclude_mask] = -1e10  # 筛选无效候选，将每个用户已访问过的POI对应的评分置为一个极小值-1e10，从而在排序时这些项目不会被选中
            item_score = item_score.cpu()

            for score, label in zip(item_score, pos_list):  # 计算TOP-k指标
                ranked_idx = np.argsort(-score)[: max(Ks)]  # 对评分倒序排序，取出前K最大评分的POI索引ranked_idx（这里K取最大值20）
                rank_results = label[
                    ranked_idx]  # 再用label[ranked_idx]取出这些位置对应的真实标签值，得到rank_results数组——其中1表示命中真实下一个POI，0表示未命中

                recall, ndcg, acc = [], [], []  # 初始化临时列表，计算不同k下的指标
                for K in Ks:
                    hit_at_k = ACC_at_k(rank_results, K, 1)  # 单正样本下，ACC@K 与 Recall@K 数值相同
                    recall.append(hit_at_k)
                    acc.append(hit_at_k)
                    ndcg.append(NDCG_at_k(rank_results, K))  # 函数根据rank_results前K项的位置给命中项按对数衰减累加，得到归一化折损累计增益
                mrr = MRR(rank_results)  # 计算平均倒数排名，函数会根据rank_results前K项的位置给命中项按对数衰减累加，得到归一化折损累计增益

                result['Recall'] += recall
                result['NDCG'] += ndcg
                result['MRR'] += mrr
                result['ACC'] += acc
                tot_cnt += 1

    # 汇总平均，循环结束后，用累计值除以总用户数tot_cnt求出平均指标
    result['Recall'] /= tot_cnt
    result['NDCG'] /= tot_cnt
    result['MRR'] /= tot_cnt
    result['ACC'] /= tot_cnt
    return result  # 用于显示验证/测试集的指标


def save_experiment_summary(run_number, test_result, best_epoch, total_time, log_dir):
    """保存实验结果摘要到单独的CSV文件"""
    import csv
    import os

    summary_file = os.path.join(log_dir, 'experiments_summary.csv')
    is_new_file = not os.path.exists(summary_file)

    with open(summary_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(['Run', 'Date', 'Dataset', 'Best_Epoch', 'Recall@5', 'NDCG@5', 'MRR',
                             'Total_Time', 'Device', 'Learning_Rate', 'Batch_Size', 'Dropout'])

        writer.writerow([
            f'{run_number:03d}',
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            gol.dataset,
            best_epoch,
            test_result['Recall'][2],  # @5
            test_result['NDCG'][2],  # @5
            test_result['MRR'],
            f'{total_time:.2f}s',
            str(gol.device),
            gol.conf['lr'],
            gol.BATCH_SZ,
            gol.conf['dp'] if gol.conf['dropout'] else 0
        ])


def train_eval(model: DiffDGMN, datasets, start_epoch: int = 0, opt_state_dict=None, best_val_epoch_init: int = 0, best_val_score_init: float = 0.0, ckpt_epoch_tmpl: str = None, ckpt_latest_path: str = None):
    global w_best_path, w_last_path
    trn_set, val_set, tst_set = datasets  # datasets是一个包含训练集、验证集、测试集的元组，此处进行解包
    trn_loader = DataLoader(trn_set, batch_size=gol.BATCH_SZ, shuffle=True, collate_fn=collate_edge)  # 训练数据加载器
    opt = torch.optim.AdamW(model.parameters(), lr=gol.conf['lr'], weight_decay=gol.conf['decay'])  # 定义优化器
    if opt_state_dict is not None:
        try:
            opt.load_state_dict(opt_state_dict)
            gol.pLog('Optimizer state restored from checkpoint.')
        except Exception as e:
            gol.pLog(f'Warning: failed to load optimizer state: {str(e)}')
    batch_num = len(trn_set) // gol.BATCH_SZ  # 每epoch=训练集样本数整除批大小
    best_val_epoch, best_val_score = best_val_epoch_init, best_val_score_init  # 跟踪验证集上 NDCG@5 最高的 epoch 及对应值
    ave_tot, ave_rec, ave_fis = 0., 0., 0.  # 累积每epoch的平均总损失、推荐损失、散度损失
    tst_result = None  # 保存最佳模型对应的测试结果

    # pos_lbl是正样本POI的索引（每个用户下一个要预测的POI），neg是负样本

    for epoch in range(start_epoch, gol.EPOCH):  # 遍历预定的最大epoch
        with torch.no_grad():
            R_V_all = model.compute_R_V_all()  # 内部会用到 self.G_D（若你的B模块需要）
            model.set_R_V_all(R_V_all)
        model.train()
        for idx, batch in enumerate(trn_loader):
            loss_rec, loss_div = model.getTrainLoss(batch)  # 返回两个标量：推荐损失和散度损失
            tot_loss = loss_rec + gol.conf['zeta'] * loss_div  # 计算加权总损失，按权重融合两种损失，默认zeta=0.2，

            # 梯度更新
            opt.zero_grad()  # 梯度清零
            tot_loss.backward()  # 反向传播
            # 梯度裁剪，稳定训练
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()  # 参数更新
            if idx % (batch_num // 2) == 0:  # 若经过一半批次，打印输出当前批次及损失值
                gol.pLog(
                    f'Batch {idx} / {batch_num}, Total_loss: {tot_loss.item():.5f}' + f' = Recloss: {loss_rec.item():.5f} + Divloss: {loss_div.item():.5f}')

            # 累计损失后求平均
            ave_tot += tot_loss.item()
            ave_rec += loss_rec.item()
            ave_fis += loss_div.item()

        ave_tot /= batch_num
        ave_rec /= batch_num
        ave_fis /= batch_num
        # validation phase:
        val_results = eval_model(model, val_set)

        # 日志记录：输出当前epoch的平均损失和验证集关键指标（含@1/@5/@10/@20）
        gol.pLog(
            f'Avg Epoch {epoch} / {gol.EPOCH}, Total_Loss: {ave_tot:.5f}' + f' = Recloss: {ave_rec:.5f} + Divloss: {ave_fis:.5f}')
        r, n, a = val_results["Recall"], val_results["NDCG"], val_results.get("ACC", val_results["Recall"])    
        gol.pLog(
            f'Valid Recall@[1,5,10,20]: {r[0]:.5f}, {r[2]:.5f}, {r[3]:.5f}, {r[4]:.5f}')
        gol.pLog(
            f'Valid NDCG@[1,5,10,20]: {n[0]:.5f}, {n[2]:.5f}, {n[3]:.5f}, {n[4]:.5f}')
        gol.pLog(
            f'Valid ACC@[1,5,10,20]: {a[0]:.5f}, {a[2]:.5f}, {a[3]:.5f}, {a[4]:.5f}')
        # 保存检查点（按配置）
        if ckpt_epoch_tmpl is not None and ckpt_latest_path is not None:
            if ((epoch + 1) % getattr(gol, 'CKPT_EVERY', 1)) == 0:
                save_checkpoint(model, opt, epoch, best_val_epoch, best_val_score, ckpt_epoch_tmpl, ckpt_latest_path)

        # 早停检查
        if epoch - best_val_epoch == gol.patience:
            gol.pLog(f'Stop training after {gol.patience} epochs without valid improvement.')
            break

        # 每轮均在测试集上完整评估并输出，便于进度检查
        tst_result_epoch = eval_model(model, tst_set)
        gol.pLog(
            f'Test top@k at epoch {epoch} (k = {1, 2, 5, 10, 20}):\n {pformat(tst_result_epoch)}'
        )

        # 模型保存与测试（以 NDCG@5 作为最佳判据）；复用当轮测试结果，避免重复评估
        if val_results["NDCG"][2] > best_val_score or epoch == 0:
            best_val_epoch, best_val_score = epoch, val_results["NDCG"][2]
            tst_result = tst_result_epoch  # 记录当前最佳对应的测试结果
            gol.pLog(
                f'New test top@k result at k = {1, 2, 5, 10, 20}:\n {pformat(tst_result)}')
            # 始终保存当前最佳权重（state_dict）
            try:
                state = model.state_dict()
                torch.save(state, w_best_path)
                gol.pLog(f'Saved best weights to: {w_best_path}')
            except Exception as e:
                gol.pLog(f'Failed to save best weights: {str(e)}')
        gol.pLog(f'Best valid NDCG@5 at epoch {best_val_epoch}\n')  # 每轮末尾，打印当前最佳验证 NDCG@5 及其所在epoch

    # 训练结束后保存最后一轮权重（state_dict）
    try:
        torch.save(model.state_dict(), w_last_path)
        gol.pLog(f'Saved last weights to: {w_last_path}')
    except Exception as e:
        gol.pLog(f'Failed to save last weights: {str(e)}')

    return tst_result, best_val_epoch


if __name__ == '__main__':
    import time

    start_time = time.time()

    # 设置日志文件
    log_path = setup_logging()
    log_dir = os.path.dirname(log_path)
    run_number = int(log_path.split('_run_')[-1].split('_')[0])

    # 创建权重/检查点路径（使用运行编号）
    w_best_path = os.path.join(log_dir, f'best_weight_run_{run_number:03d}.pth')
    w_last_path = os.path.join(log_dir, f'last_weight_run_{run_number:03d}.pth')
    ckpt_epoch_tmpl = os.path.join(log_dir, f'ckpt_run_{run_number:03d}_ep' + '{epoch:04d}.pt')
    ckpt_latest_path = os.path.join(log_dir, f'ckpt_run_{run_number:03d}_latest.pt')

    try:
        n_user, n_poi, datasets, G_D = getDatasets(gol.DATA_PATH, gol.dataset)
        # 获得用户数n_user、POI数n_poi，以及划分后的训练集trn_set、验证集val_set、测试集tst_set
        POI_model = DiffDGMN(n_user, n_poi, G_D)  # 模型初始化

        # 断点续训优先
        start_epoch = 0
        best_val_epoch_init, best_val_score_init = 0, 0.0
        opt_state = None
        if getattr(gol, 'RESUME', False):
            resume_path = gol.RESUME_PATH
            if resume_path is None:
                # 先找 latest（新旧命名）
                candidates = []
                candidates += glob.glob(os.path.join(log_dir, 'ckpt_run_*_latest.pt'))
                old_latest = os.path.join(log_dir, 'latest.pt')
                if os.path.exists(old_latest):
                    candidates.append(old_latest)
                if candidates:
                    candidates.sort(key=lambda p: os.path.getmtime(p))
                    resume_path = candidates[-1]
                else:
                    # 再找 epoch 快照（新旧命名）
                    ep_cands = []
                    ep_cands += glob.glob(os.path.join(log_dir, 'ckpt_run_*_ep*.pt'))
                    ep_cands += glob.glob(os.path.join(log_dir, 'ckpt_run_*_epoch_*.pt'))
                    if ep_cands:
                        ep_cands.sort(key=lambda p: os.path.getmtime(p))
                        resume_path = ep_cands[-1]
            if resume_path is not None and os.path.exists(resume_path):
                start_epoch, best_val_epoch_init, best_val_score_init = load_checkpoint(resume_path, POI_model, None, map_location=gol.device)
                try:
                    _ck = torch.load(resume_path, map_location='cpu')
                    if isinstance(_ck, dict):
                        opt_state = _ck.get('optimizer', _ck.get('optimizer_state', None))
                    gol.pLog(f'Resuming from checkpoint: {resume_path} (start_epoch={start_epoch})')
                except Exception as e:
                    gol.pLog(f'Warning: failed reading optimizer state from ckpt: {str(e)}')
            else:
                gol.pLog('Resume requested but no checkpoint found, starting fresh.')

        # 仅加载权重（不恢复训练状态）
        elif gol.LOAD:
            load_path = w_best_path if os.path.exists(w_best_path) else (w_last_path if os.path.exists(w_last_path) else None)
            if load_path is not None:
                POI_model.load_state_dict(torch.load(load_path, map_location=gol.device))  # 从预训练权重文件加载参数到模型
                gol.pLog(f'Loaded weights from: {load_path}')
        POI_model = POI_model.to(gol.device)  # 模型在计算设备上切换，CPU or GPU

        # 训练日志记录
        gol.pLog(f'Dropout probability: {gol.conf["dp"] if gol.conf["dropout"] else 0}')
        # 计算并输出模型参数总数
        num_params = 0
        for param in POI_model.parameters():
            num_params += param.numel()
        gol.pLog(f'The Number of Parameters for the Diff-DGMN Model is {num_params}')
        gol.pLog(f'-------------------Start Training---------------------\n')

        test_result, best_epoch = train_eval(
            POI_model,
            datasets,
            start_epoch=start_epoch,
            opt_state_dict=opt_state,
            best_val_epoch_init=best_val_epoch_init,
            best_val_score_init=best_val_score_init,
            ckpt_epoch_tmpl=ckpt_epoch_tmpl,
            ckpt_latest_path=ckpt_latest_path
        )
        gol.pLog(f'\n Training on {gol.dataset.upper()} Finished, Best Valid at epoch {best_epoch}')
        gol.pLog(f'***Best Test Top@k Result at k = {1, 2, 5, 10, 20} is***\n{pformat(test_result)}')

        # 保存实验结果摘要
        total_time = time.time() - start_time
        save_experiment_summary(run_number, test_result, best_epoch, total_time, log_dir)
        gol.pLog(f'\nExperiment summary saved to: {os.path.join(log_dir, "experiments_summary.csv")}')

    except Exception as e:
        # 记录错误信息
        gol.pLog(f'\nError occurred: {str(e)}')
        import traceback

        gol.pLog(traceback.format_exc())
    finally:
        # 确保实验时长被记录
        total_time = time.time() - start_time
        gol.pLog(f'\nTotal experiment time: {total_time:.2f} seconds')