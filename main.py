# coding=utf-8
import os.path
import torch
# Compatible with both package and script execution.
try:
    from . import gol  # global configuration (hyperparameters, global variables)
except Exception:
    import gol
import numpy as np
from torch.utils.data import DataLoader  # tensor computation + data loading
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
    # Collect existing log files for a given dataset
    pattern = os.path.join(log_dir, f'GT_kan_{dataset}_run_*.log')
    existing_logs = glob.glob(pattern)

    # If no logs exist, start from 1
    if not existing_logs:
        return 1

    # Extract run numbers from filenames
    numbers = []
    for log in existing_logs:
        try:
            num = int(log.split('_run_')[-1].split('_')[0])
            numbers.append(num)
        except:
            continue

    # Return (max_run_number + 1)
    return max(numbers) + 1 if numbers else 1


def setup_logging():
    # Ensure the local log directory exists
    local_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_logs', 'GT_kan', gol.dataset)
    os.makedirs(local_log_dir, exist_ok=True)

    # Get the next run number
    run_number = get_next_log_number(local_log_dir, gol.dataset)

    # Generate log filename: dataset_run_<id>_<timestamp>.log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = f'GT_kan_{gol.dataset}_run_{run_number:03d}_{timestamp}.log'
    log_path = os.path.join(local_log_dir, log_name)

    # Configure logging
    import logging
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%m/%d %H:%M')

    # File handler (write to local file)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handlers
    root_logger.addHandler(file_handler)

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Record basic experiment info
    logging.info(f'Experiment started - Run #{run_number:03d}')
    logging.info(f'Local log file: {log_path}')
    logging.info(f'Dataset: {gol.dataset}')
    logging.info(f'Device: {gol.device}')
    logging.info(f'Configuration: {gol.conf}')

    return log_path


# Dataset / collate / loader / metrics helpers

# Checkpoint save/load (compatible with legacy/new naming)
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
        # Legacy key compatibility
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
    # Prefer safe weights-only load; fall back to full load for legacy checkpoints.
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as _e:
        gol.pLog(f'Weights-only load failed, fallback to unsafe load (trusted local ckpt). Reason: {str(_e)}')
        ckpt = torch.load(path, map_location=map_location)

    # Parse multiple checkpoint formats
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

# Evaluate the model on validation/test sets.
def eval_model(model: DiffDGMN, eval_set: GraphData):
    Ks = [1, 2, 5, 10, 20]  # report metrics at Top-K
    result = {'Recall': np.zeros(len(Ks)), 'NDCG': np.zeros(len(Ks)),
              'MRR': 0., 'ACC': np.zeros(len(Ks))}  # include ACC@K (same as Recall@K for single-positive)
    eval_loader = DataLoader(eval_set, batch_size=gol.TEST_BATCH_SZ, shuffle=True, collate_fn=collate_eval)
    # Note: the evaluation batch includes user ids, candidate labels/masks, and graph structures.

    with torch.no_grad():
        model.eval()  # evaluation mode
        tot_cnt = 0  # total number of evaluated users
        for idx, batch in enumerate(eval_loader):  # batch evaluation loop
            u, pos_list, exclude_mask, seqs, seq_graph, cur_time = batch
            # Attach required fields to the graph to match forward() inputs
            try:
                seq_graph.user = u.to(seq_graph.x.device, non_blocking=True)
            except Exception:
                # If x is absent, fall back to gol.device
                seq_graph.user = u.to(gol.device, non_blocking=True)

            if isinstance(seqs, (list, tuple)):
                seq_graph.seq = torch.cat(list(seqs), dim=0).to(seq_graph.user.device, non_blocking=True)
            else:
                seq_graph.seq = seqs.to(seq_graph.user.device, non_blocking=True)

            # Forward pass: predict scores over all POIs
            item_score = model(seq_graph)

            # Mask invalid candidates: set visited POIs to a very small value so they are not ranked.
            item_score[exclude_mask] = -1e10
            item_score = item_score.cpu()

            for score, label in zip(item_score, pos_list):  # compute Top-K metrics
                ranked_idx = np.argsort(-score)[: max(Ks)]  # top-K indices (K = max(Ks))
                rank_results = label[
                    ranked_idx]  # 1 indicates a hit (true next POI), 0 indicates miss

                recall, ndcg, acc = [], [], []  # metrics at different K
                for K in Ks:
                    hit_at_k = ACC_at_k(rank_results, K, 1)  # single-positive: ACC@K == Recall@K
                    recall.append(hit_at_k)
                    acc.append(hit_at_k)
                    ndcg.append(NDCG_at_k(rank_results, K))  # log-discounted gain within Top-K
                mrr = MRR(rank_results)  # mean reciprocal rank

                result['Recall'] += recall
                result['NDCG'] += ndcg
                result['MRR'] += mrr
                result['ACC'] += acc
                tot_cnt += 1

    # Average over evaluated users
    result['Recall'] /= tot_cnt
    result['NDCG'] /= tot_cnt
    result['MRR'] /= tot_cnt
    result['ACC'] /= tot_cnt
    return result


def save_experiment_summary(run_number, test_result, best_epoch, total_time, log_dir):
    """Save a compact experiment summary to a CSV file."""
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
    trn_set, val_set, tst_set = datasets  # unpack (train, valid, test)
    trn_loader = DataLoader(trn_set, batch_size=gol.BATCH_SZ, shuffle=True, collate_fn=collate_edge)
    opt = torch.optim.AdamW(model.parameters(), lr=gol.conf['lr'], weight_decay=gol.conf['decay'])
    if opt_state_dict is not None:
        try:
            opt.load_state_dict(opt_state_dict)
            gol.pLog('Optimizer state restored from checkpoint.')
        except Exception as e:
            gol.pLog(f'Warning: failed to load optimizer state: {str(e)}')
    batch_num = len(trn_set) // gol.BATCH_SZ  # num batches per epoch (floor)
    best_val_epoch, best_val_score = best_val_epoch_init, best_val_score_init  # track best valid NDCG@5
    ave_tot, ave_rec, ave_fis = 0., 0., 0.  # running averages (total / rec / divergence)
    tst_result = None  # test results for the best validation checkpoint

    # Training samples provide positive POIs; negatives are sampled in the dataset.

    for epoch in range(start_epoch, gol.EPOCH):  # epochs
        with torch.no_grad():
            R_V_all = model.compute_R_V_all()  # precompute candidate geographic representations
            model.set_R_V_all(R_V_all)
        model.train()
        for idx, batch in enumerate(trn_loader):
            loss_rec, loss_div = model.getTrainLoss(batch)  # (recommendation loss, divergence loss)
            tot_loss = loss_rec + gol.conf['zeta'] * loss_div  # weighted sum

            # Gradient update
            opt.zero_grad()
            tot_loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            if idx % (batch_num // 2) == 0:  # log mid-epoch progress
                gol.pLog(
                    f'Batch {idx} / {batch_num}, Total_loss: {tot_loss.item():.5f}' + f' = Recloss: {loss_rec.item():.5f} + Divloss: {loss_div.item():.5f}')

            # Accumulate losses for epoch averages
            ave_tot += tot_loss.item()
            ave_rec += loss_rec.item()
            ave_fis += loss_div.item()

        ave_tot /= batch_num
        ave_rec /= batch_num
        ave_fis /= batch_num
        # Validation phase
        val_results = eval_model(model, val_set)

        # Log epoch losses and validation metrics (@1/@5/@10/@20)
        gol.pLog(
            f'Avg Epoch {epoch} / {gol.EPOCH}, Total_Loss: {ave_tot:.5f}' + f' = Recloss: {ave_rec:.5f} + Divloss: {ave_fis:.5f}')
        r, n, a = val_results["Recall"], val_results["NDCG"], val_results.get("ACC", val_results["Recall"])    
        gol.pLog(
            f'Valid Recall@[1,5,10,20]: {r[0]:.5f}, {r[2]:.5f}, {r[3]:.5f}, {r[4]:.5f}')
        gol.pLog(
            f'Valid NDCG@[1,5,10,20]: {n[0]:.5f}, {n[2]:.5f}, {n[3]:.5f}, {n[4]:.5f}')
        gol.pLog(
            f'Valid ACC@[1,5,10,20]: {a[0]:.5f}, {a[2]:.5f}, {a[3]:.5f}, {a[4]:.5f}')
        # Save checkpoint snapshots (if enabled)
        if ckpt_epoch_tmpl is not None and ckpt_latest_path is not None:
            if ((epoch + 1) % getattr(gol, 'CKPT_EVERY', 1)) == 0:
                save_checkpoint(model, opt, epoch, best_val_epoch, best_val_score, ckpt_epoch_tmpl, ckpt_latest_path)

        # Early-stopping check
        if epoch - best_val_epoch == gol.patience:
            gol.pLog(f'Stop training after {gol.patience} epochs without valid improvement.')
            break

        # Evaluate on test set each epoch (useful for progress monitoring)
        tst_result_epoch = eval_model(model, tst_set)
        gol.pLog(
            f'Test top@k at epoch {epoch} (k = {1, 2, 5, 10, 20}):\n {pformat(tst_result_epoch)}'
        )

        # Model selection by valid NDCG@5; reuse the current epoch's test result to avoid re-evaluation
        if val_results["NDCG"][2] > best_val_score or epoch == 0:
            best_val_epoch, best_val_score = epoch, val_results["NDCG"][2]
            tst_result = tst_result_epoch  # store test results for the best valid checkpoint
            gol.pLog(
                f'New test top@k result at k = {1, 2, 5, 10, 20}:\n {pformat(tst_result)}')
            # Always save best weights (state_dict)
            try:
                state = model.state_dict()
                torch.save(state, w_best_path)
                gol.pLog(f'Saved best weights to: {w_best_path}')
            except Exception as e:
                gol.pLog(f'Failed to save best weights: {str(e)}')
        gol.pLog(f'Best valid NDCG@5 at epoch {best_val_epoch}\n')

    # Save last-epoch weights (state_dict)
    try:
        torch.save(model.state_dict(), w_last_path)
        gol.pLog(f'Saved last weights to: {w_last_path}')
    except Exception as e:
        gol.pLog(f'Failed to save last weights: {str(e)}')

    return tst_result, best_val_epoch


if __name__ == '__main__':
    import time

    start_time = time.time()

    # Set up logging
    log_path = setup_logging()
    log_dir = os.path.dirname(log_path)
    run_number = int(log_path.split('_run_')[-1].split('_')[0])

    # Create weight/checkpoint paths (scoped by run id)
    w_best_path = os.path.join(log_dir, f'best_weight_run_{run_number:03d}.pth')
    w_last_path = os.path.join(log_dir, f'last_weight_run_{run_number:03d}.pth')
    ckpt_epoch_tmpl = os.path.join(log_dir, f'ckpt_run_{run_number:03d}_ep' + '{epoch:04d}.pt')
    ckpt_latest_path = os.path.join(log_dir, f'ckpt_run_{run_number:03d}_latest.pt')

    try:
        n_user, n_poi, datasets, G_D = getDatasets(gol.DATA_PATH, gol.dataset)
        # Load user/POI counts and dataset splits, and build the global distance graph
        POI_model = DiffDGMN(n_user, n_poi, G_D)  # model init

        # Resume-from-checkpoint has priority
        start_epoch = 0
        best_val_epoch_init, best_val_score_init = 0, 0.0
        opt_state = None
        if getattr(gol, 'RESUME', False):
            resume_path = gol.RESUME_PATH
            if resume_path is None:
                # First, search for "latest" checkpoints (new/legacy naming)
                candidates = []
                candidates += glob.glob(os.path.join(log_dir, 'ckpt_run_*_latest.pt'))
                old_latest = os.path.join(log_dir, 'latest.pt')
                if os.path.exists(old_latest):
                    candidates.append(old_latest)
                if candidates:
                    candidates.sort(key=lambda p: os.path.getmtime(p))
                    resume_path = candidates[-1]
                else:
                    # Then, search for epoch snapshots (new/legacy naming)
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

        # Only load weights (do not restore optimizer/RNG/training state)
        elif gol.LOAD:
            load_path = w_best_path if os.path.exists(w_best_path) else (w_last_path if os.path.exists(w_last_path) else None)
            if load_path is not None:
                POI_model.load_state_dict(torch.load(load_path, map_location=gol.device))  # load pretrained weights
                gol.pLog(f'Loaded weights from: {load_path}')
        POI_model = POI_model.to(gol.device)  # move model to device

        # Training logs
        gol.pLog(f'Dropout probability: {gol.conf["dp"] if gol.conf["dropout"] else 0}')
        # Log parameter count
        num_params = 0
        for param in POI_model.parameters():
            num_params += param.numel()
        gol.pLog(f'The Number of Parameters for the Hyper-GMVAN Model is {num_params}')
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

        # Save experiment summary
        total_time = time.time() - start_time
        save_experiment_summary(run_number, test_result, best_epoch, total_time, log_dir)
        gol.pLog(f'\nExperiment summary saved to: {os.path.join(log_dir, "experiments_summary.csv")}')

    except Exception as e:
        # Log error info
        gol.pLog(f'\nError occurred: {str(e)}')
        import traceback

        gol.pLog(traceback.format_exc())
    finally:
        # Always record total runtime
        total_time = time.time() - start_time
        gol.pLog(f'\nTotal experiment time: {total_time:.2f} seconds')