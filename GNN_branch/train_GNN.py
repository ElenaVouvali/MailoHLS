#-----------------------------------------------------------
#                       train.py
#-----------------------------------------------------------

from config import FLAGS
from saver import saver
from utils import MLP, OurTimer, MLP_multi_objective, plot_loss_trend, _get_y_with_target, create_dir_if_not_exists, plot_lr_trend
# from data import MyOwnDataset, get_kernel_samples, split_dataset, split_dataset_resample, split_train_test_kernel
# import data
from mlir_data import (
    MyOwnDataset,
    get_kernel_samples,
    split_dataset,
    split_dataset_resample,
    split_train_val_test_kernel,
)
import mlir_data as data
SAVE_DIR = data.SAVE_DIR

from model import Net

from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, \
    mean_absolute_percentage_error, classification_report, confusion_matrix

import torch
import pytorch_warmup as warmup
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import torch.nn as nn
import shutil
import numpy as np
from scipy.stats import kendalltau

from tqdm import tqdm
from os.path import join, exists, basename

from collections import OrderedDict, defaultdict

import pandas as pd


def _as_int_seed(seed):
    if isinstance(seed, (int, np.integer)):
        return int(seed)
    if isinstance(seed, str):
        return int(seed.strip())
    raise TypeError(f"random seed must be int-like, got {type(seed)}: {seed}")


def gen_dataset(li):
    if FLAGS.tiny_overfit:
        train_workers = FLAGS.tiny_overfit_workers
        eval_workers = FLAGS.tiny_overfit_workers
    else:
        train_workers = FLAGS.num_workers
        eval_workers = FLAGS.eval_num_workers

    def make_loader(ds, shuffle, workers):
        kwargs = dict(
            batch_size=FLAGS.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=False,
        )
        if shuffle:
            kwargs["generator"] = torch.Generator().manual_seed(
                _as_int_seed(FLAGS.random_seed)
            )
        if workers > 0:
            kwargs["prefetch_factor"] = FLAGS.prefetch_factor
            kwargs["persistent_workers"] = FLAGS.persistent_workers
        return DataLoader(ds, **kwargs)

    train_loader = make_loader(li[0], shuffle=True,  workers=train_workers)
    val_loader   = make_loader(li[1], shuffle=False, workers=eval_workers)
    test_loader  = make_loader(li[2], shuffle=False, workers=eval_workers)

    loader = test_loader if len(test_loader.dataset) > 0 else train_loader

    num_features = loader.dataset[0].num_features
    saver.info(f'num features for training: {num_features}')
    edge_dim = loader.dataset[0].edge_attr.shape[1]
    saver.info(f'size of the edge attribute is {edge_dim}')

    return train_loader, val_loader, test_loader, num_features, edge_dim


def _take_random_fraction(dataset, frac, seed, *, as_subset=False):
    n = len(dataset)
    k = max(1, int(round(n * frac)))
    seed = _as_int_seed(seed)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False).tolist()

    if as_subset:
        from torch.utils.data import Subset
        return Subset(dataset, indices)

    # Build the sampled file list from the dataset
    if hasattr(dataset, 'data_files'):
        files_all = dataset.data_files
    else:
        files_all = dataset.processed_file_names  # uses glob on SAVE_DIR

    sampled_files = [files_all[i] for i in indices]

    # Recreate a MyOwnDataset that only sees the sampled files
    return MyOwnDataset(
        transform=getattr(dataset, 'transform', None),
        pre_transform=getattr(dataset, 'pre_transform', None),
        data_files=sampled_files,
    )


def _take_random_n(dataset, n, seed):
    n_total = len(dataset)
    k = min(int(n), n_total)
    seed = _as_int_seed(seed)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=k, replace=False).tolist()
    return Subset(dataset, indices)


def _filter_dataset_by_kernel(dataset, kernel_name):
    """
    Robust tiny-overfit filtering for the new compact dataset.

    Works even if dataset.processed_file_names / records are not in the
    exact format we expect, because it filters by actual dataset samples.
    """
    selected_indices = []

    # Fast path: if compact records are present as dicts, use them directly
    records = getattr(dataset, 'records', None)
    if records is not None and len(records) > 0 and isinstance(records[0], dict):
        for i, rec in enumerate(records):
            gname = rec['graph_name']
            if gname.startswith(kernel_name):
                selected_indices.append(i)
    else:
        # Fallback path: inspect actual samples
        for i in range(len(dataset)):
            d = dataset[i]
            k = d.kernel[0] if isinstance(d.kernel, (list, tuple)) else d.kernel
            if k == kernel_name:
                selected_indices.append(i)

    if len(selected_indices) == 0:
        raise RuntimeError(f'No records found for kernel={kernel_name}')

    return Subset(dataset, selected_indices)


def process_split_data(dataset):
    dataset_dict = defaultdict(list)

    if FLAGS.tiny_overfit:
        kernel_ds = _filter_dataset_by_kernel(dataset, FLAGS.tiny_overfit_kernel)
        tiny_ds = _take_random_n(kernel_ds, FLAGS.tiny_overfit_num_samples, FLAGS.random_seed)

        saver.log_info(
            f"[tiny_overfit] kernel={FLAGS.tiny_overfit_kernel} "
            f"num_samples={len(tiny_ds)}"
        )

        dataset_dict['train'] = tiny_ds
        dataset_dict['val'] = tiny_ds
        dataset_dict['test'] = tiny_ds
        return dataset_dict

    # Full training: use the whole compact dataset directly
    dataset_dict['train'] = dataset
    dataset_dict['val'] = None
    dataset_dict['test'] = None

    if not FLAGS.all_kernels:
        dataset_dict['train'] = get_kernel_samples(dataset)
    elif FLAGS.test_kernels is not None or FLAGS.val_kernels is not None:
        dataset_dict = split_train_val_test_kernel(dataset)

    return dataset_dict


def get_train_val_count(num_graphs, val_ratio, test_ratio):
    if FLAGS.test_kernels is not None:
        r1 = int(num_graphs * (1.0 - val_ratio))
        r2 = int(num_graphs * (val_ratio))
    else:
        r1 = int(num_graphs * (1.0 - val_ratio - test_ratio))
        r2 = int(num_graphs * (val_ratio))

    return r1, r2


def check_feature_extract(model, key_word, gnn_layer=None):
    '''"
        checks that all parameters except for the ones that have "key_word" are fixed
        as a result, only "key_word" params will be updated
    '''
    for name, param in model.named_parameters():
        if key_word not in name:
            if not gnn_layer:
                assert param.requires_grad == False
            else:
                if 'conv_first' in name or any([f'conv_layers.{d}' in name for d in range(gnn_layer-1)]):
                    assert param.requires_grad == False


def model_update(model, losses_list, loss, epoch, plot_test, tag):
    '''
        Tracks the current loss, saves the best model so far and optionally triggers plotting
    '''
    saver.writer.add_scalar(f'{tag}/{tag}', loss, epoch)
    if not losses_list or loss < min(losses_list):
        if FLAGS.save_model:
            saver.log_info((f'Saved {tag} model at epoch {epoch}'))
            torch.save(model.state_dict(), join(saver.model_logdir, f"{tag}_model_state_dict.pth"))
        plot_test = True
    losses_list.append(loss)

    return plot_test


def log_loss(loss_dict, gae_loss, tag):
    saver.log_info((f'{tag} GAE loss: {gae_loss}'))
    saver.log_info((f'{tag} loss breakdown {loss_dict}'))


def set_target_list():
    '''
        Creates loss_dict per target (zero initialized)
    '''
    _target_list = FLAGS.target
    if not isinstance(FLAGS.target, list):
        _target_list = [FLAGS.target]
    if FLAGS.task =='regression':
        target_list = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t for t in _target_list]
    else:
        target_list = [_target_list[0]]

    loss_dict = {}
    for t in target_list:
        loss_dict[t] = 0.0

    return target_list, loss_dict


def update_total_loss(loss, data, target_list, loss_dict, loss_dict_, out_dict, total_loss, correct):
    '''
        Accumulates training metrics per batch and updates the loss_dict for each target
    '''
    if FLAGS.task == 'regression':
        # MSELoss is a batch mean. Weight by the number of graphs so the final
        # epoch/validation loss is independent of the smaller last batch.
        batch_weight = int(data.num_graphs)
        total_loss += loss.item() * batch_weight
        for t in target_list:
            model_key = 'perf' if FLAGS.encode_log and t == 'actual_perf' else t
            loss_dict[t] += loss_dict_[model_key].item() * batch_weight
        return loss_dict, total_loss
    else:
        loss_, pred = torch.max(out_dict[FLAGS.target[0]], 1)
        labels = _get_y_with_target(data, FLAGS.target[0])
        correct += (pred == labels).sum().item()
        total_loss += labels.size(0)
        return pred, correct, total_loss


def inference_loss_function(pred, true):
    return (pred - true) ** 2


def _inverse_log2_target(value):
    """Convert a model output back to the positive physical QoR domain."""
    if str(FLAGS.norm_method) != 'log2':
        raise RuntimeError(
            "Physical-unit evaluation currently requires --norm_method log2."
        )
    return max(0.0, float(np.exp2(value) - float(FLAGS.epsilon)))


def _kernel_at(data, index):
    kernels = getattr(data, 'kernel', None)
    if isinstance(kernels, (list, tuple)):
        return str(kernels[index])
    if isinstance(kernels, str) and int(data.num_graphs) == 1:
        return kernels
    raise RuntimeError("Could not align a prediction with its kernel name.")


def update_csv_dict(csv_dict, data, i, target_name, target_value, out_value):
    '''
        Collects per graph "actual VS predicted" rows into a dict we can later write into a CSV
    '''
    if csv_dict is not None:
        gname = _get_y_with_target(data, 'gname')[i]
        pragma = _get_y_with_target(data, 'pragmas')[i][0].item()
        pragma = '-'.join([str(j.item()) for j in _get_y_with_target(data, 'pragmas')[i]])
        if True or 'blocked' in gname:
            if f'{gname}-{pragma}' not in csv_dict:
                csv_dict[f'{gname}-{pragma}'] = {'gname': gname, 'pragma': pragma}
            csv_dict[f'{gname}-{pragma}'][f'acutal-{target_name}'] = target_value
            csv_dict[f'{gname}-{pragma}'][f'predicted-{target_name}'] = out_value
            l = csv_dict['header']
            if f'acutal-{target_name}' not in l:
                l.extend([f'acutal-{target_name}', f'predicted-{target_name}'])
                csv_dict['header'] = l


def _print_sanity_rows(csv_dict, n=10):
    if csv_dict is None or n <= 0:
        return

    rows = []
    for k, v in csv_dict.items():
        if k == 'header':
            continue

        rows.append({
            'gname': v.get('gname'),
            'pragma': v.get('pragma'),
            'actual_perf': v.get('acutal-perf'),
            'pred_perf': v.get('predicted-perf'),
            'actual_area': v.get('acutal-area'),
            'pred_area': v.get('predicted-area'),
        })

    if not rows:
        saver.log_info('[sanity] no rows collected')
        return

    df = pd.DataFrame(rows).head(n).copy()

    saver.log_info('[sanity] first predictions vs targets:')
    saver.log_info(df.round(4).to_string(index=False))

    pred_cols = ['pred_perf', 'pred_area']
    has_nan = df[pred_cols].isna().any().any()
    saver.log_info(f'[sanity] NaNs present? {has_nan}')
    saver.log_info(
        f"[sanity] unique pred_perf={df['pred_perf'].nunique()} | "
        f"unique pred_area={df['pred_area'].nunique()}"
    )


def train_main(dataset, pragma_dim = None, val_ratio=FLAGS.val_ratio, test_ratio=FLAGS.val_ratio, resample=-1):
    saver.info(f'Reading dataset from {SAVE_DIR}')

    dataset_dict = process_split_data(dataset)
    num_graphs = len(dataset_dict['train'])
    r1, r2 = get_train_val_count(num_graphs, val_ratio, test_ratio)

    if FLAGS.tiny_overfit:
        tiny_ds = dataset_dict['train']
        li = [tiny_ds, tiny_ds, tiny_ds]
        saver.log_info(
            f"[tiny_overfit] using identical subset for train/val/test "
            f"(n={len(tiny_ds)})"
        )
    else:
        if dataset_dict.get('val') is not None:
            li = [
                dataset_dict['train'],
                dataset_dict['val'],
                dataset_dict['test'] or MyOwnDataset(data_files=[]),
            ]
        elif resample == -1:
            li = split_dataset(dataset_dict['train'], r1, r2, dataset_test=dataset_dict['test'])
        else:
            li = split_dataset_resample(
                dataset_dict['train'],
                1.0 - val_ratio - test_ratio,
                val_ratio,
                test_ratio,
                test_id=resample
            )

    train_loader, val_loader, test_loader, num_features, edge_dim = gen_dataset(li)
    model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=pragma_dim).to(FLAGS.device)
    saver.log_info(f"Model first param device: {next(model.parameters()).device}")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    else:
        print("CPU training")

    if FLAGS.load_pretrained and FLAGS.model_path is not None:
        model_path = FLAGS.model_path[0] if isinstance(FLAGS.model_path, list) else FLAGS.model_path
        saver.info(f'loading model from {model_path}')
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        )

    if FLAGS.feature_extract:
        feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)

    saver.log_model_architecture(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    num_steps = len(train_loader) * FLAGS.epoch_num
    # warmup_steps = len(train_loader) * 10
    warmup_steps = int(0.1 * num_steps)

    if FLAGS.scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[FLAGS.epoch_num // 3], gamma=0.1)
    elif FLAGS.scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)
    else:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    if FLAGS.warmup == 'linear':
        # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_steps)
    elif FLAGS.warmup == 'exponential':
        warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
    elif FLAGS.warmup == 'radam':
        warmup_scheduler = warmup.RAdamWarmup(optimizer)
    else:
        warmup_scheduler = warmup.LinearWarmup(optimizer, 1)

    ckpt_path = join(saver.model_logdir, "last_ckpt.pt")
    start_epoch = 0

    train_losses, val_losses, test_losses, total_lrs = [], [], [], []
    gae_train_losses, gae_val_losses, gae_test_losses = [], [], []
    plot_test = False

    if FLAGS.resume_training and exists(ckpt_path):
        st = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(st["model"])
        model.to(FLAGS.device)
        optimizer.load_state_dict(st["optim"])
        if FLAGS.scheduler is not None and st.get("scheduler") is not None:
            lr_scheduler.load_state_dict(st["scheduler"])
        for s in optimizer.state.values():
            for k, v in s.items():
                if torch.is_tensor(v):
                    s[k] = v.to(FLAGS.device)
        start_epoch = st.get("epoch", -1) + 1
        train_losses = st.get("train_losses", train_losses)
        val_losses   = st.get("val_losses", val_losses)
        test_losses  = st.get("test_losses", test_losses)
        saver.log_info(f"Resuming from checkpoint at epoch {start_epoch}")


    for epoch in range(start_epoch, FLAGS.epoch_num):
        plot_test = False
        timer = OurTimer()
        if FLAGS.feature_extract:
            check_feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)
        saver.log_info(f'Test batch ID (resample): {resample} - Epoch {epoch} train')
        loss, loss_dict_train, gae_loss_train, lrs = train(epoch, model, train_loader, optimizer, lr_scheduler, warmup_scheduler)
        plot_test = model_update(model, train_losses, loss, epoch, plot_test, 'train')
        total_lrs.extend(lrs)
        if len(val_loader) > 0:
            saver.log_info(f'Epoch {epoch} val')
            val, loss_dict_val, gae_loss_val, _ = test(val_loader, 'val', model, epoch)
            plot_test = model_update(model, val_losses, val, epoch, plot_test, 'val')

        # log_loss(loss_dict_train, gae_loss_train, "Train")
        if len(val_loader) > 0:
        #     log_loss(loss_dict_val, gae_loss_val, "Val")
            saver.log_info(('Epoch: {:03d}, Train Loss: {:.4f}, '
                            'Val loss: {:.4f}, Time: {}'.format(
                            epoch, loss, val, timer.time_and_clear())))
        #     gae_val_losses.append(gae_loss_val)
        else:
            saver.log_info(('Epoch: {:03d}, Train loss: {:.4f}, '
                            'Time: {}'.format(
                epoch, loss, timer.time_and_clear())))
        gae_train_losses.append(gae_loss_train)

        if (epoch + 1) % 2 == 0 or (epoch + 1) == FLAGS.epoch_num:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": (lr_scheduler.state_dict() if FLAGS.scheduler is not None else None),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "test_losses": test_losses,
            }, ckpt_path)
            saver.log_info(f"Checkpoint saved at epoch {epoch} -> {ckpt_path}")

        if len(train_losses) > 50:
            selection_losses = val_losses if val_losses else train_losses
            if len(selection_losses) >= 50 and len(set(selection_losses[-50:])) == 1:
                break

    epochs = range(epoch+1)
    plot_loss_trend(epochs, train_losses, val_losses, [], saver.get_log_dir(), file_name='losses.png')
    if FLAGS.gae_T or FLAGS.gae_P:
        plot_loss_trend(epochs, gae_train_losses, gae_val_losses, [], saver.get_log_dir(), file_name='gae_losses.png')

    # Restore the validation-selected checkpoint before reporting validation
    # and exposing the held-out test set exactly once.
    selection_tag = 'val' if val_losses else 'train'
    selection_epoch = (
        val_losses.index(min(val_losses))
        if val_losses
        else train_losses.index(min(train_losses))
    )
    selection_path = join(
        saver.model_logdir,
        f'{selection_tag}_model_state_dict.pth',
    )
    if len(val_loader) > 0 or len(test_loader) > 0:
        if not exists(selection_path):
            raise RuntimeError(
                f'Missing {selection_tag}-selected checkpoint: {selection_path}'
            )
        model.load_state_dict(
            torch.load(
                selection_path,
                map_location=torch.device('cpu'),
                weights_only=False,
            )
        )
        model.to(FLAGS.device)

    if len(val_loader) > 0:
        saver.log_info(
            f'Final validation report for best epoch {selection_epoch}'
        )
        test(
            val_loader,
            'best_val',
            model,
            selection_epoch,
            test_losses=[],
        )

    if len(test_loader) > 0:
        saver.log_info(
            f'Final test using the best {selection_tag} checkpoint from '
            f'epoch {selection_epoch}'
        )
        test_loss, loss_dict_test, gae_loss_test, _ = test(
            test_loader,
            'test',
            model,
            selection_epoch,
            plot_test=True,
            test_losses=[],
        )
        test_losses.append(test_loss)
        saver.writer.add_scalar('test/final', test_loss, selection_epoch)
        log_loss(loss_dict_test, gae_loss_test, 'Final test')
        saver.log_info(f'Final held-out test loss: {test_loss:.4f}')

    if len(test_loader) > 0:
        saver.log_info('The test set was evaluated exactly once.')
    if len(val_loader) > 0:
        saver.log_info(f'min val loss at epoch: {val_losses.index(min(val_losses))}')
    if FLAGS.scheduler is not None:
        plot_lr_trend(total_lrs, FLAGS.epoch_num + 1, saver.get_log_dir())
    saver.log_info(f'min train loss at epoch: {train_losses.index(min(train_losses))}')


def train(epoch, model, train_loader, optimizer, lr_scheduler, warmup_scheduler):
    model.train()
    lrs = []
    total_loss, correct, i = 0, 0, 0
    target_list, loss_dict = set_target_list()
    for data in tqdm(train_loader):
        if FLAGS.scheduler is not None:
            lr = optimizer.param_groups[0]['lr']
            lrs.append(lr)
            if i == 0:
                saver.log_info(f"epoch = {epoch}, learning rate = {lr}")
        data = data.to(FLAGS.device)
        out_dict, loss, loss_dict_, gae_loss = model(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if FLAGS.scheduler is not None:
            # lr_scheduler.step()
            # lr_scheduler.step(lr_scheduler.last_epoch+1)
        if FLAGS.warmup is not None and FLAGS.scheduler is not None:
            with warmup_scheduler.dampening():
                lr_scheduler.step()
            # warmup_scheduler.dampen()
        # lr = optimizer.param_groups[0]['lr']

        total_loss_dict = update_total_loss(loss, data, target_list, loss_dict, loss_dict_, out_dict, total_loss, correct)
        if FLAGS.task == 'regression': loss_dict, total_loss = total_loss_dict
        else: pred, correct, total_loss = total_loss_dict

        saver.writer.add_scalar('loss/loss', loss, epoch * len(train_loader) + i)
        i += 1

    if FLAGS.scheduler is not None and epoch < 2:
        create_dir_if_not_exists(join(saver.get_log_dir(), 'lrs'))
    if FLAGS.task == 'regression':
        example_count = len(train_loader.dataset)
        return (
            total_loss / example_count,
            {key: v / example_count for key, v in loss_dict.items()},
            gae_loss,
            lrs,
        )
    else:
        return 1 - correct / total_loss, {key: v / len(train_loader) for key, v in loss_dict.items()}, gae_loss, lrs


def inference(dataset, init_pragma_dict=None, model_path=FLAGS.model_path,
              val_ratio=FLAGS.val_ratio, test_ratio=FLAGS.val_ratio,
              resample=-1, model_id=0, is_train_set=False, is_val_set=False):

    dataset_dict = process_split_data(dataset)
    num_graphs = len(dataset_dict['train'])
    r1, r2 = get_train_val_count(num_graphs, val_ratio, test_ratio)

    if FLAGS.tiny_overfit:
        tiny_ds = dataset_dict['train']
        li = [tiny_ds, tiny_ds, tiny_ds]
        saver.log_info(
            f"[tiny_overfit][inference] using identical subset for train/val/test "
            f"(n={len(tiny_ds)})"
        )
    else:
        if dataset_dict.get('val') is not None:
            li = [
                dataset_dict['train'],
                dataset_dict['val'],
                dataset_dict['test'] or MyOwnDataset(data_files=[]),
            ]
        elif resample == -1:
            li = split_dataset(
                dataset_dict['train'],
                r1,
                r2,
                dataset_test=dataset_dict['test']
            )
        else:
            li = split_dataset_resample(
                dataset_dict['train'],
                1.0 - val_ratio - test_ratio,
                val_ratio,
                test_ratio,
                test_id=resample
            )

    train_loader, val_loader, test_loader, num_features, edge_dim = gen_dataset(li)
    test_set = test_loader
    if is_train_set:
        test_set = train_loader
        saver.info('running inference on train set')
    elif is_val_set:
        test_set = val_loader
        saver.info('running inference on val set')

    if init_pragma_dict is None:
        init_pragma_dict = {'all': [1, 21]}
    model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=init_pragma_dict).to(FLAGS.device)

    if model_path is not None:
        saver.info(f'loading model from {model_path}')
        state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        model.load_state_dict(state)
        shutil.copy(model_path, join(saver.logdir, f"{(basename(model_path)).split('.')[0]}-{model_id}.pth"))
    else:
        saver.error('model path should be set during inference')
        raise RuntimeError()

    if model_id == 0:
        saver.log_model_architecture(model)

    data_list = []

    if FLAGS.task == 'regression':
        csv_dict = {'header': ['gname', 'pragma']}
        test_loss, loss_dict, gae_loss, MSE_loss = test(
            test_set,
            'test',
            model,
            0,
            plot_test=True,
            csv_dict=csv_dict,
            data_list=data_list,
            is_train_set=is_train_set,
            is_val_set=is_val_set
        )

        loss_dict = {k: round(v, 4) for k, v in loss_dict.items()}
        saver.log_info(f'{loss_dict}')
        saver.log_info('Test loss: {:.7f}, MSE loss: {:.7f}'.format(test_loss, MSE_loss))

        if FLAGS.sanity_print_n > 0:
            _print_sanity_rows(csv_dict, FLAGS.sanity_print_n)

        saver.log_dict_of_dicts_to_csv(f'actual-prediction-{model_id}', csv_dict, csv_dict['header'])
        print(len(data_list), 'out of', len(test_loader))
    else:
        test_loss, loss_dict_test = test(test_loader, 'test', model, 0)
        saver.log_info(('Test loss: {:.3f}'.format(test_loss)))


def test(loader, tvt, model, epoch, plot_test = False, test_losses = [-1], csv_dict = None, data_list = [], is_train_set=False, is_val_set=False):
    model.eval()
    my_softplus = nn.Softplus()
    inference_loss, correct, total, count_data = 0, 0, 0, 1
    points_dict = OrderedDict()
    target_list, loss_dict = set_target_list()
    for target_name in target_list:
        points_dict[target_name] = {
            'true': [],
            'pred': [],
            'physical_true': [],
            'physical_pred': [],
            'kernel': [],
            'sigma_mu': [],
            'sigma+mu': [],
            'sigma': [],
            'error': [],
        }
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(FLAGS.device)
            out_dict, loss, loss_dict_, gae_loss = model(data)
            total_loss_dict = update_total_loss(loss, data, target_list, loss_dict, loss_dict_, out_dict, total, correct)
            if FLAGS.task == 'regression': loss_dict, total = total_loss_dict
            else: pred, correct, total = total_loss_dict

            for target_name in target_list:
                if 'inf' in FLAGS.subtask:
                    saver.info(f'{target_name}')
                if FLAGS.task == 'class': out = pred
                elif FLAGS.encode_log and 'perf' in target_name: out = out_dict['perf']
                else: out = out_dict[target_name]

                for i in range(len(out)):
                    out_value = out[i].item()
                    target_value = _get_y_with_target(data, target_name)[i].item()
                    if FLAGS.encode_log and target_name == 'actual_perf':
                        out_value = 2**(out_value) * (1 / FLAGS.normalizer)
                    if 'inf' in FLAGS.subtask:
                        inference_loss += inference_loss_function(out_value, target_value)
                        count_data += 1
                        update_csv_dict(csv_dict, data, i, target_name, target_value, out_value)

                        if out_value != target_value: # and sigma[i].item() > 0.57:
                            saver.info(f"{target_name} data {i} {_get_y_with_target(data, 'gname')[i]} pramga {_get_y_with_target(data, 'pragmas')[i][0].item()} actual value: {target_value:.2f}, predicted value: {out_value:.2f}") #, sigma: {sigma[i].item()}, log_var: {out_[i, 1].item()}')")

                    points_dict[target_name]['pred'].append((target_value, out_value))
                    points_dict[target_name]['true'].append((target_value, target_value))
                    points_dict[target_name]['error'].append((target_value, abs(target_value - out_value)))

                    physical_attr = (
                        'actual_perf'
                        if target_name in {'perf', 'actual_perf'}
                        else 'actual_area'
                    )
                    physical_true = _get_y_with_target(
                        data, physical_attr
                    )[i].item()
                    points_dict[target_name]['physical_true'].append(
                        physical_true
                    )
                    points_dict[target_name]['physical_pred'].append(
                        _inverse_log2_target(out_value)
                    )
                    points_dict[target_name]['kernel'].append(
                        _kernel_at(data, i)
                    )


    if FLAGS.task != 'class' and FLAGS.plot_pred_points and tvt == 'test' and (
        plot_test or (
            test_losses
            and (total / len(loader.dataset)) < min(test_losses)
        )
    ):
        from utils import plot_points_with_subplot, plot_points_with_subplot_sigma
        saver.log_info(f'@@@ plot_pred_points')
        assert(isinstance(FLAGS.target, list))
        use_sigma = False
        label = f'epoch_{epoch+1}_{tvt}_train' if is_train_set else f'epoch_{epoch+1}_{tvt}_test'
        if is_val_set: label = f'epoch_{epoch+1}_{tvt}_val'
        if 'inf' in FLAGS.subtask:
            plot_points_with_subplot(points_dict, label, saver.plotdir, target_list, use_sigma=use_sigma)

    if FLAGS.task == 'regression':
        df = _report_rmse_etc(points_dict, f'[{tvt}] epoch {epoch}', print_result=True)
        for _, row in df[df['aggregation'] == 'point_micro'].iterrows():
            saver.log_info(
                f"[{tvt}] {row['target']} physical RMSE: "
                f"{row['rmse']:.4f} | MAE: {row['mae']:.4f} | "
                f"MAPE: {row['mape']*100:.2f}% | tau-b: "
                f"{row['tau']:.4f}"
            )
        if tvt in {'best_val', 'test'}:
            metrics_path = join(
                saver.model_logdir, f'{tvt}_physical_metrics.csv'
            )
            df.to_csv(metrics_path, index=False)
            saver.log_info(f'Wrote physical-unit metrics to {metrics_path}')

    if FLAGS.task == 'regression':
        if 'inf' in FLAGS.subtask:
            _report_rmse_etc(points_dict, f'epoch {epoch}:', True)
        example_count = len(loader.dataset)
        return (
            total / example_count,
            {key: v / example_count for key, v in loss_dict.items()},
            gae_loss,
            inference_loss / max(1, count_data) * len(target_list),
        )
    else:
        if 'inf' in FLAGS.subtask: report_class_loss(points_dict)
        return 1 - correct / total, {key: v / len(loader) for key, v in loss_dict.items()}, gae_loss, 0


def report_class_loss(points_dict):
    d = points_dict[FLAGS.target[0]]
    labels = [data for data,_ in d['pred']]
    pred = [data for _,data in d['pred']]
    target_names = ['invalid', 'valid']
    saver.info('classification report')
    saver.log_info(classification_report(labels, pred, target_names=target_names))
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    saver.info(f'Confusion matrix:\n{cm}')

def _report_rmse_etc(points_dict, label, print_result=True):
    """Report physical-unit point and per-kernel macro regression metrics."""
    if print_result:
        saver.log_info(label)

    def metrics(true_values, predicted_values):
        true_values = np.asarray(true_values, dtype=float)
        predicted_values = np.asarray(predicted_values, dtype=float)
        mse = float(mean_squared_error(true_values, predicted_values))
        tau = float(kendalltau(true_values, predicted_values)[0])
        return {
            'mape': float(mean_absolute_percentage_error(
                true_values, predicted_values
            )),
            'rmse': float(np.sqrt(mse)),
            'mse': mse,
            'mae': float(mean_absolute_error(
                true_values, predicted_values
            )),
            'max_err': float(max_error(true_values, predicted_values)),
            'tau': tau,
        }

    rows = []
    for target_name, values in points_dict.items():
        physical_target = (
            'latency_ms'
            if target_name in {'perf', 'actual_perf'}
            else 'area_score'
        )
        true_values = values['physical_true']
        predicted_values = values['physical_pred']
        kernels = values['kernel']
        if not true_values or not (
            len(true_values) == len(predicted_values) == len(kernels)
        ):
            raise RuntimeError(
                f'Incomplete physical metrics for {target_name}.'
            )

        micro = metrics(true_values, predicted_values)
        rows.append({
            'target': physical_target,
            'aggregation': 'point_micro',
            'samples': len(true_values),
            'kernels': len(set(kernels)),
            **micro,
        })

        per_kernel = []
        for kernel in sorted(set(kernels)):
            indices = [
                index for index, value in enumerate(kernels)
                if value == kernel
            ]
            per_kernel.append(metrics(
                [true_values[index] for index in indices],
                [predicted_values[index] for index in indices],
            ))
        macro = {
            name: float(np.nanmean([item[name] for item in per_kernel]))
            for name in ('mape', 'rmse', 'mse', 'mae', 'max_err', 'tau')
        }
        rows.append({
            'target': physical_target,
            'aggregation': 'kernel_macro',
            'samples': len(true_values),
            'kernels': len(per_kernel),
            **macro,
        })

    # Latency and area have different units, so there is deliberately no
    # combined "tot/avg" error row. Report both targets independently.
    df = pd.DataFrame(rows)
    pd.set_option('display.max_columns', None)
    if print_result:
        saver.log_info(df.round(4))
    return df
