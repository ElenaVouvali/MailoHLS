#-----------------------------------------------------------
#                       config.py
#-----------------------------------------------------------

from utils import get_user, get_host, get_root_path
import argparse
import torch
from glob import iglob
from os.path import join

decoder_arch = []

parser = argparse.ArgumentParser()
# TASK = 'class'
TASK = 'regression'
parser.add_argument('--task', default=TASK)

# SUBTASK = 'dse'
# SUBTASK = 'inference'
SUBTASK = 'train'
parser.add_argument('--subtask', default=SUBTASK)
parser.add_argument('--plot_dse', default=False)


#################### visualization ####################
parser.add_argument('--vis_per_kernel', default=True) ## only tsne visualization for now


######################## data ########################

TARGETS = ['perf', 'area'] #, 'quality', 'util-BRAM', 'util-DSP', 'util-LUT', 'util-FF']

ALL_KERNEL = ['machsuite-gemm-blocked', 'machsuite-gemm-ncubed', 'machsuite-md-knn', 'machsuite-sort-radix',
              'machsuite-spmv-ellpack', 'machsuite-stencil2d', 'machsuite-stencil3d', 'machsuite-viterbi',
              'rodinia-backprop-0-baseline-back', 'rodinia-backprop-0-baseline-forward', 'rodinia-backprop-1-tiling-back',
              'rodinia-hotspot-0-baseline', 'rodinia-hotspot-1-tiling', 'rodinia-hotspot-2-pipeline', 'rodinia-hotspot-3-unroll',
              'rodinia-kmeans-0-baseline', 'rodinia-kmeans-1-tiling', 'rodinia-kmeans-2-pipeline', 'rodinia-kmeans-3-unroll',
              'rodinia-knn-0-baseline', 'rodinia-knn-1-tiling', 'rodinia-knn-2-pipeline', 'rodinia-knn-3-unroll',
              'rodinia-knn-4-doublebuffer', 'rodinia-knn-5-coalescing', 'rodinia_cfd_flux_0_baseline_0',
              'rodinia_cfd_step_factor_0_baseline_0', 'rodinia_cfd_step_factor_1_tiling_0', 'rodinia_cfd_step_factor_2_pipeline_0',
              'rodinia_cfd_step_factor_3_unroll_0', 'rodinia_cfd_step_factor_4_doublebuffer_0', 'rodinia_cfd_step_factor_5_coalescing_0',
              'rodinia_dilate_0_baseline_0', 'rodinia_dilate_1_tiling_0', 'rodinia_dilate_2_pipeline_0', 'rodinia_dilate_3_pipeline_0',
              'rodinia_lavaMD_0_baseline', 'rodinia_lavaMD_1_tiling_0', 'rodinia_lavaMD_1_tiling_1', 'rodinia_lavaMD_2_pipeline_0',
              'rodinia_lc_gicov_0_baseline_0', 'rodinia_lc_mgvf_0_baseline_0', 'rodinia_lud_1_tiling_0', 'rodinia_pathfinder_0_baseline_0',
              'rodinia_pathfinder_4_doublebuffer_0', 'rodinia_streamcluster_0_baseline_0', 'rodinia_streamcluster_1_tiling_0',
              'rodinia_streamcluster_2_pipeline_1', 'rodinia_streamcluster_3_doublebuffer_0', 'rodinia_streamcluster_4_coalescing_0',
              'serrano-kalman-filter', 'spcl_example_00', 'spcl_example_01', 'spcl_example_03', 'spcl_example_05']


parser.add_argument(
    '--force_regen',
    action='store_true',
    help='Regenerate cached graph tensors and feature encoders.',
)
parser.add_argument(
    '--allow_incomplete_dataset',
    action='store_true',
    help='Debug/research only: allow missing, failed, stale, or incompatible MLIR graphs.',
)

parser.add_argument(
    '--min_allowed_latency',
    type=float,
    default=0.0,
    help=(
        'Optional latency floor in milliseconds. GNN preprocessing already '
        'removes failed synthesis rows, so production defaults to no extra '
        'floor; use a positive value only for a documented sensitivity test.'
    ),
)
EPSILON = 1e-6
parser.add_argument('--epsilon', default=EPSILON)
NORMALIZER = 1e7
parser.add_argument('--normalizer', default=NORMALIZER)
parser.add_argument('--util_normalizer', default=1)
MAX_NUMBER = 1e10
parser.add_argument('--max_number', default=MAX_NUMBER)

norm = 'log2' # 'speedup-log2' 'const' 'speedup' 'off' 'speedup-const' 'const-log2' 'none' 'speedup-log2'
parser.add_argument('--norm_method', default=norm)
parser.add_argument('--new_speedup', default=True) # new_speedup: same reference point across all,
                                                    # old_speedup: base is the longest latency and different per kernel

parser.add_argument(
    '--include_invalid',
    dest='invalid',
    action='store_true',
    help=(
        'Bypass a positive --min_allowed_latency sensitivity filter. This '
        'does not restore invalid rows removed during CSV preprocessing.'
    ),
)

parser.add_argument('--encode_log', type = bool, default=False)
v_db = 'v21' # 'v20': v20 database, 'v18': v18 database
parser.add_argument('--v_db', default=v_db) # if set to true uses the db of the new version of the tool: 2020.2

test_kernels = None
parser.add_argument(
    '--test_kernels',
    default=test_kernels,
    help='Comma-separated kernels reserved for final evaluation.',
)
parser.add_argument(
    '--val_kernels',
    default=None,
    help='Comma-separated kernels used for model selection.',
)
parser.add_argument(
    '--development_exclude_kernels',
    default=None,
    help=(
        'Comma-separated kernels omitted from development because no '
        'authenticated reference measurement is available. Excluded kernels '
        'cannot also be validation or test kernels.'
    ),
)
target_kernel = None
# target_kernel = 'gemm-blocked'
parser.add_argument('--target_kernel', default=target_kernel)
if target_kernel == None:
    all_kernels = True
else:
    all_kernels = False
parser.add_argument('--all_kernels', type = bool, default=all_kernels)

dataset = 'mlir' # 'harp' 'mlir'
parser.add_argument('--dataset', default=dataset)

benchmark = ['machsuite', 'poly']
parser.add_argument('--benchmarks', default=benchmark)

tag = 'whole-machsuite-poly'
parser.add_argument('--tag', default=tag)


###################### graph type ######################
graph_type = 'original' # original DAC22 graph
graph_type = 'extended-pseudo-block-connected-hierarchy'
parser.add_argument('--graph_type', default=graph_type)

parser.add_argument(
    "--mlir_graph_dir",
    default=None,
    help="Explicit versioned MLIR GEXF dataset directory.",
)

################## model architecture ##################
pragma_as_MLP, type_parallel, type_merge = True, '2l', '2l'
gnn_layer_after_MLP = 1
pragma_MLP_hidden_channels, merge_MLP_hidden_channels = None, None
if 'hierarchy' not in graph_type: ## separate_PT original graph
    gae_T, P_use_all_nodes, separate_pseudo, separate_T, dropout, num_features, edge_dim = True, True, False, True, 0.1, 154, 7
    model_ver = 'original-PT'
else:
    if pragma_as_MLP:
        if gnn_layer_after_MLP == 1: model_ver = 'pragma_as_MLP'

        if type_parallel == '2l': pragma_MLP_hidden_channels = '[in_D // 2]'
        elif type_parallel == '3l': pragma_MLP_hidden_channels = '[in_D // 2, in_D // 4]'

        if type_merge == '2l': merge_MLP_hidden_channels = '[in_D // 2]'
        elif type_merge == '3l': merge_MLP_hidden_channels = '[in_D // 2, in_D // 4]'
        else: raise NotImplementedError()
        gae_T, P_use_all_nodes, separate_pseudo, separate_T, dropout, num_features, edge_dim = False, True, True, False, 0.2, 361, 1807  # dropout = 0.1
    else:
        gae_T, P_use_all_nodes, separate_pseudo, separate_T, dropout, num_features, edge_dim = True, False, False, True, 0.1, 156, 335
        model_ver = 'hierarchy-PT'

################# one-hot encoder ##################
encoder_path = None
pragma_dim_path = None
encode_edge_position = True
use_encoder = False
if use_encoder:
    encoder_path_list = [f for f in iglob(join(get_root_path(), 'save', 'harp', 'all_kernels', '**'), recursive=True) if f.endswith('.klepto') and 'encoders' in f]
    pragma_dim_path_list = [f for f in iglob(join(get_root_path(), 'save', 'harp', 'all_kernels', '**'), recursive=True) if f.endswith('.klepto') and 'pragma_dim' in f]

    assert len(encoder_path_list) == 1, print(encoder_path_list)
    encoder_path = encoder_path_list[0]
    assert len(pragma_dim_path_list) == 1, print(pragma_dim_path_list)
    pragma_dim_path = pragma_dim_path_list[0]

parser.add_argument('--encoder_path', default=encoder_path)
parser.add_argument('--pragma_dim_path', default=pragma_dim_path)


################ model architecture #################
## edge attributes
parser.add_argument('--encode_edge', type=bool, default=True)
parser.add_argument('--encode_edge_position', type=bool, default=encode_edge_position)

num_layers = 4  # 6
parser.add_argument('--num_layers', type=int, default=num_layers)
parser.add_argument('--num_features', type=int, default=None)
parser.add_argument('--edge_dim', type=int, default=None)

multi_target = ['perf', 'area'] #, 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']
if SUBTASK == 'class':
    multi_target = ['perf']

parser.add_argument(
    "--target",
    nargs="+",
    choices=TARGETS,
    default=multi_target,
    help="Targets to predict, for example: --target perf area",
)

parser.add_argument(
    "--decompose_targets",
    action="store_true",
    help=(
        "Deprecated alias for --target_mode kernel_center. Kept only so old "
        "Stage B commands/checkpoints remain reproducible."
    ),
)
parser.add_argument(
    "--target_mode",
    choices=("absolute", "kernel_center", "reference_delta"),
    default="absolute",
    help=(
        "absolute predicts log2 QoR directly; kernel_center reproduces Stage B; "
        "reference_delta predicts log2(QoR)-log2(neutral-reference QoR) and "
        "adds the measured neutral reference back at inference."
    ),
)
parser.add_argument(
    "--baseline_manifest",
    default=None,
    help="CSV produced by generate_neutral_baselines.py for reference_delta mode.",
)
parser.add_argument(
    "--target_device",
    default="xczu7ev-ffvc1156-2-e",
    help="FPGA part that every neutral baseline in the manifest must use.",
)
parser.add_argument(
    "--clock_period_ns",
    type=float,
    default=10.0,
    help="Clock period that every neutral baseline in the manifest must use.",
)
parser.add_argument(
    "--vitis_hls_version",
    default="2021.1",
    help="Vitis HLS version substring required by the neutral-baseline manifest.",
)
parser.add_argument(
    "--center_aux_weight",
    type=float,
    default=0.25,
    help="Auxiliary weight for the static kernel-center prediction.",
)
parser.add_argument(
    "--response_aux_weight",
    type=float,
    default=1.0,
    help="Auxiliary weight for the within-kernel pragma response.",
)
parser.add_argument(
    "--checkpoint_objective",
    choices=("absolute", "qualified_rank"),
    default="absolute",
    help=(
        "Select either the lowest absolute validation-error checkpoint or "
        "the best within-kernel ranking checkpoint that also beats the "
        "constant baseline on every target."
    ),
)
parser.add_argument(
    "--min_rank_tau",
    type=float,
    default=0.0,
    help="Minimum worst-target kernel-macro Kendall tau-b for rank qualification.",
)
parser.add_argument(
    "--max_kernel_zero_baseline_ratio",
    type=float,
    default=1.10,
    help=(
        "Maximum validation loss ratio to the zero/no-learning predictor for "
        "every individual kernel and target in a qualified-rank checkpoint."
    ),
)

parser.add_argument('--MLP_common_lyr', default=0)
gnn_type = 'transformer'
parser.add_argument('--gnn_type', type=str, default=gnn_type)
parser.add_argument('--dropout', type=float, default=dropout)

jkn_mode = 'max'
parser.add_argument('--jkn_mode', type=str, default=jkn_mode)
parser.add_argument('--jkn_enable', type=bool, default=True)
node_attention = True
parser.add_argument('--node_attention', type=bool, default=node_attention)
if node_attention:
    parser.add_argument('--node_attention_MLP', type=bool, default=False)

    separate_P = True
    parser.add_argument('--separate_P', type=bool, default=separate_P)
    separate_icmp = False
    parser.add_argument('--separate_icmp', type=bool, default=separate_icmp)
    separate_T = False
    parser.add_argument('--separate_T', type=bool, default=separate_T)
    separate_pseudo = True
    parser.add_argument('--separate_pseudo', type=bool, default=separate_pseudo)

    if separate_P:
        parser.add_argument('--P_use_all_nodes', type=bool, default=P_use_all_nodes)

## graph auto encoder
parser.add_argument('--gae_T', default = gae_T)
gae_P = False
parser.add_argument('--gae_P', default = gae_P)
if gae_P:
    parser.add_argument('--input_encode', default = False)
    d_type = 'type1'
    parser.add_argument('--decoder_type', default = d_type)

if pragma_as_MLP:
    assert graph_type == 'extended-pseudo-block-connected-hierarchy'
parser.add_argument('--gnn_layer_after_MLP', default=gnn_layer_after_MLP) ## number of message passing layers after MLP (pragma as MLP)
parser.add_argument('--pragma_as_MLP', default=pragma_as_MLP)
pragma_as_MLP_list = ['pipeline', 'unroll', 'array_partition']
parser.add_argument('--pragma_as_MLP_list', default=pragma_as_MLP_list)
pragma_scope = 'block'
parser.add_argument('--pragma_scope', default=pragma_scope)
keep_pragma_attribute = False if pragma_as_MLP else True
parser.add_argument('--keep_pragma_attribute', default=keep_pragma_attribute)
pragma_order = 'parallel_and_merge'
parser.add_argument('--pragma_order', default=pragma_order)
pragma_MLP_hidden_channels = '[in_D // 2]'
parser.add_argument('--pragma_MLP_hidden_channels', default=pragma_MLP_hidden_channels)
merge_MLP_hidden_channels = '[in_D // 2]'
parser.add_argument('--merge_MLP_hidden_channels', default=merge_MLP_hidden_channels)


model_path = None
model_path_list = ['/home/ubuntu/val_model_state_dict.pth']
use_pretrain = False
if use_pretrain:
    #base_path = 'models'
    #keyword =  v_db
    #includes = [keyword, model_ver, 'regression']
    #excludes = ['class']
    #model_base_path = '/home/elvouvali/logs/dse_results_v21_2025-12-16T07-58-00.354322/run1/*'
    #model = [f for f in iglob(model_base_path, recursive=True) if f.endswith('.pth') and 'val' in f]
    #print(model)
    model_path = model_path_list

parser.add_argument('--model_path', default=model_path) ## list of models when used in DSE, if more than 1, ensemble inference must be on

ensemble = 0
ensemble_weights = None
parser.add_argument('--ensemble', type=int, default=ensemble)
parser.add_argument('--ensemble_weights', default=ensemble_weights)
class_model_path = None
if SUBTASK == 'dse':
    keyword =  v_db
    includes = [keyword, model_ver, 'class']
    model = [f for f in iglob(model_base_path, recursive=True) if f.endswith('.pth') and all(k in f for k in includes)]
    assert len(model) == 1
    class_model_path = model[0]
parser.add_argument('--class_model_path', default=class_model_path)


################ transfer learning #################
feature_extract = False
parser.add_argument('--feature_extract', default=feature_extract) # if set to true GNN encoder (or part of it) will be fixed and only MLP will be trained
if feature_extract:
    parser.add_argument('--random_MLP', default=False) # true: initialize MLP randomly
fix_gnn_layer = 1 ## if none, all layers will be fixed
# fix_gnn_layer = 1 ## number of gnn layers to freeze, feature_extract should be set to True
parser.add_argument('--fix_gnn_layer', default=fix_gnn_layer) # if not set to none, feature_extract should be True
FT_extra = False
parser.add_argument('--FT_extra', default=FT_extra) ## fine-tune only on the new data points


################ training details #################
parser.add_argument('--save_model', type = bool, default=True)
resample = False
val_ratio = 0.05    # 0.15
parser.add_argument('--resample', default=resample) ## when resample is turned on, it will divide the dataset in round-robin and train multiple times to have all the points in train/test set
parser.add_argument('--val_ratio', type=float, default=val_ratio) # ratio of database for validation set
parser.add_argument('--activation', default='elu')
parser.add_argument('--D', type=int, default=64)    
scheduler, warmup, weight_decay = 'cosine', 'linear', 1e-4
parser.add_argument('--weight_decay', type=float, default=weight_decay) ## default=0.0001, larger than 1e-4 didn't help original graph P+T
parser.add_argument("--scheduler", default=scheduler)
parser.add_argument("--warmup", default=warmup)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument(
    '--standardize_targets',
    action='store_true',
    help='Standardize each log2 QoR target using training-kernel statistics.',
)
parser.add_argument(
    '--kernel_balanced_loss',
    action='store_true',
    help='Give every training kernel equal total weight in the regression loss.',
)
parser.add_argument(
    '--kernel_uniform_sampling',
    action='store_true',
    help=(
        'Sample training kernels uniformly before sampling their design points. '
        'Use with --kernel_balanced_loss to avoid high-variance inverse-count '
        'training weights.'
    ),
)
parser.add_argument(
    '--samples_per_kernel_per_epoch',
    type=int,
    default=None,
    help=(
        'When kernel-uniform sampling is enabled, cap each epoch at this many '
        'draws per training kernel. This changes exposure, not the validation set.'
    ),
)
parser.add_argument('--rank_aux_weight', type=float, default=0.0)
parser.add_argument('--rank_temperature', type=float, default=1.0)
parser.add_argument('--rank_tie_epsilon', type=float, default=0.05)
parser.add_argument('--resource_aux_weight', type=float, default=0.0)
parser.add_argument(
    '--strict_paired_ablation',
    action='store_true',
    help='Require an exactly matched R0 control for a causal R0/R1 ablation.'
)
parser.add_argument(
    '--paired_control_contract',
    type=str,
    default=None,
    help=(
        'R0 gnn_checkpoint_contract.json used to qualify an R1 resource-head '
        'causal comparison.'
    ),
)
parser.add_argument(
    '--resource_eval_budgets',
    default='0.10;0.25;0.50;0.75;1.00',
    help=(
        'Semicolon-separated utilization-fraction budgets. Each entry is '
        'either one shared fraction or BRAM,DSP,FF,LUT fractions.'
    ),
)
parser.add_argument('--resource_boundary_tolerance', type=float, default=0.02)
parser.add_argument('--kernel_grouped_sampling', action='store_true')
parser.add_argument('--kernels_per_batch', type=int, default=16)
parser.add_argument('--points_per_kernel', type=int, default=4)
parser.add_argument(
    '--warmup_epochs',
    type=int,
    default=3,
    help='Fixed number of warmup epochs; independent of total training length.',
)
parser.add_argument(
    '--early_stopping_patience',
    type=int,
    default=25,
    help='Stop after this many validation epochs without meaningful improvement.',
)
parser.add_argument(
    '--early_stopping_min_delta',
    type=float,
    default=1e-4,
    help='Minimum validation-loss decrease counted as an improvement.',
)
parser.add_argument(
    '--plateau_patience',
    type=int,
    default=4,
    help='Validation epochs without improvement before halving the learning rate.',
)
parser.add_argument(
    '--plateau_factor',
    type=float,
    default=0.5,
    help='Learning-rate multiplier used by the plateau scheduler.',
)
parser.add_argument(
    '--evaluate_test',
    action='store_true',
    help='Explicitly unlock the configured held-out test kernels once.',
)
parser.add_argument(
    '--final_refit',
    action='store_true',
    help=(
        'After model selection, merge train and validation kernels and fit for '
        'a fixed number of epochs. Test kernels remain excluded.'
    ),
)
parser.add_argument(
    '--final_refit_epochs',
    type=int,
    default=None,
    help='Fixed epoch count selected from grouped validation for --final_refit.',
)

parser.add_argument('--random_seed', type=int, default=123)
parser.add_argument(
    '--experiment_name',
    default='all_kernels_GNN_train',
    help='Log-directory name; use a distinct value for every reported seed.',
)
parser.add_argument(
    '--allow_nondeterministic',
    action='store_true',
    help='Debug only: warn instead of failing on a nondeterministic operation.',
)
batch_size = 64
parser.add_argument('--batch_size', type=int, default=batch_size)

parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--eval_num_workers', type=int, default=0)
parser.add_argument('--prefetch_factor', type=int, default=1)
parser.add_argument('--persistent_workers', action='store_true')

loss = 'mse'
parser.add_argument(
    '--loss',
    type=str.lower,
    choices=('mse', 'rmse', 'smooth_l1'),
    default=loss,
)
parser.add_argument(
    '--smooth_l1_beta',
    type=float,
    default=0.5,
    help='Quadratic-region width for --loss smooth_l1 in standardized units.',
)

if model_path == None:
    if TASK == 'regression':
        epoch_num = 200
    else:
        epoch_num = 200
else:
    epoch_num = 400

parser.add_argument('--epoch_num', type=int, default=epoch_num)
parser.add_argument('--sanity_print_n', type=int, default=0)

gpu = 0
device = str('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1
             else 'cpu')
parser.add_argument('--device', default=device)


################ tiny overfit debug ################
parser.add_argument('--tiny_overfit', action='store_true')
parser.add_argument('--tiny_overfit_kernel', type=str, default='machsuite-gemm-blocked')
parser.add_argument('--tiny_overfit_num_samples', type=int, default=64)
parser.add_argument('--tiny_overfit_batch_size', type=int, default=16)
parser.add_argument('--tiny_overfit_epochs', type=int, default=300)
parser.add_argument('--tiny_overfit_workers', type=int, default=0)
parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--load_pretrained', action='store_true')


################# DSE details ##################
explorer = 'exhaustive'
parser.add_argument('--explorer', default=explorer)

model_tag = 'test'
parser.add_argument('--model_tag', default=model_tag)

parser.add_argument('--prune_util', default=True) # only DSP and BRAM
parser.add_argument('--prune_class', default=True)

parser.add_argument('--print_every_iter', type=int, default=100)

plot = True
parser.add_argument('--plot_pred_points', type=bool, default=plot)

"""
Other info.
"""
parser.add_argument('--user', default=get_user())

parser.add_argument('--hostname', default=get_host())


FLAGS = parser.parse_args()

# Preserve old Stage B commands while making the experimental target explicit.
if FLAGS.decompose_targets:
    if FLAGS.target_mode not in ('absolute', 'kernel_center'):
        parser.error('--decompose_targets conflicts with --target_mode reference_delta.')
    FLAGS.target_mode = 'kernel_center'
FLAGS.decompose_targets = FLAGS.target_mode == 'kernel_center'

if FLAGS.center_aux_weight < 0 or FLAGS.response_aux_weight < 0:
    parser.error("Target-decomposition weights must be non-negative.")
if FLAGS.decompose_targets and FLAGS.task != 'regression':
    parser.error('--decompose_targets is available only for regression.')
if FLAGS.target_mode == 'reference_delta':
    if FLAGS.task != 'regression':
        parser.error('--target_mode reference_delta is available only for regression.')
    if FLAGS.norm_method != 'log2':
        parser.error('--target_mode reference_delta requires --norm_method log2.')
    if not FLAGS.baseline_manifest:
        parser.error('--target_mode reference_delta requires --baseline_manifest.')
if FLAGS.smooth_l1_beta <= 0:
    parser.error('--smooth_l1_beta must be positive.')

development_exclusions = {
    item.strip()
    for item in (FLAGS.development_exclude_kernels or '').split(',')
    if item.strip()
}
declared_holdouts = {
    item.strip()
    for value in (FLAGS.val_kernels, FLAGS.test_kernels)
    for item in (value or '').split(',')
    if item.strip()
}
if development_exclusions & declared_holdouts:
    parser.error(
        '--development_exclude_kernels overlaps validation/test kernels: '
        + ', '.join(sorted(development_exclusions & declared_holdouts))
    )

if FLAGS.kernel_uniform_sampling and not FLAGS.kernel_balanced_loss:
    parser.error('--kernel_uniform_sampling requires --kernel_balanced_loss.')
if FLAGS.kernel_uniform_sampling and FLAGS.kernel_grouped_sampling:
    parser.error(
        '--kernel_uniform_sampling and --kernel_grouped_sampling are mutually exclusive.'
    )
if FLAGS.rank_aux_weight < 0:
    parser.error('--rank_aux_weight must be non-negative.')
if FLAGS.rank_temperature <= 0:
    parser.error('--rank_temperature must be positive.')
if FLAGS.rank_tie_epsilon < 0:
    parser.error('--rank_tie_epsilon must be non-negative.')
if FLAGS.resource_aux_weight < 0:
    parser.error('--resource_aux_weight must be non-negative.')

if FLAGS.resource_aux_weight < 0:
    parser.error('--resource_aux_weight must be non-negative.')

if FLAGS.strict_paired_ablation:
    if FLAGS.resource_aux_weight <= 0:
        parser.error(
            '--strict_paired_ablation requires --resource_aux_weight > 0.'
        )
    if not FLAGS.paired_control_contract:
        parser.error(
            '--strict_paired_ablation requires --paired_control_contract.'
        )
    if FLAGS.rank_aux_weight != 0:
        parser.error(
            '--strict_paired_ablation requires --rank_aux_weight 0.'
        )
elif FLAGS.paired_control_contract:
    parser.error(
        '--paired_control_contract is only used with --strict_paired_ablation.'
    )
if FLAGS.resource_boundary_tolerance < 0:
    parser.error('--resource_boundary_tolerance must be non-negative.')
if not -1.0 <= FLAGS.min_rank_tau <= 1.0:
    parser.error('--min_rank_tau must be between -1 and 1.')
if FLAGS.max_kernel_zero_baseline_ratio <= 0:
    parser.error('--max_kernel_zero_baseline_ratio must be positive.')
if FLAGS.kernels_per_batch <= 0 or FLAGS.points_per_kernel <= 0:
    parser.error('--kernels_per_batch and --points_per_kernel must be positive.')
if (
    FLAGS.kernel_grouped_sampling
    and FLAGS.kernels_per_batch * FLAGS.points_per_kernel != FLAGS.batch_size
):
    parser.error(
        '--kernels_per_batch * --points_per_kernel must equal --batch_size.'
    )
if FLAGS.samples_per_kernel_per_epoch is not None:
    if FLAGS.samples_per_kernel_per_epoch <= 0:
        parser.error('--samples_per_kernel_per_epoch must be positive.')
    if not (FLAGS.kernel_uniform_sampling or FLAGS.kernel_grouped_sampling):
        parser.error(
            '--samples_per_kernel_per_epoch requires uniform or grouped kernel sampling.'
        )
if FLAGS.final_refit:
    if FLAGS.tiny_overfit:
        parser.error('--final_refit and --tiny_overfit are mutually exclusive.')
    if FLAGS.final_refit_epochs is None or FLAGS.final_refit_epochs <= 0:
        parser.error('--final_refit requires a positive --final_refit_epochs.')
    if FLAGS.scheduler == 'plateau':
        parser.error(
            '--final_refit has no validation signal; use cosine or no scheduler.'
        )
    if FLAGS.checkpoint_objective != 'absolute':
        parser.error(
            '--final_refit has no validation ranking; use '
            '--checkpoint_objective absolute.'
        )
    FLAGS.epoch_num = FLAGS.final_refit_epochs

if FLAGS.tiny_overfit:
    FLAGS.epoch_num = FLAGS.tiny_overfit_epochs
    FLAGS.batch_size = FLAGS.tiny_overfit_batch_size

    FLAGS.force_regen = False
    FLAGS.target_kernel = FLAGS.tiny_overfit_kernel
    FLAGS.all_kernels = False

    FLAGS.val_ratio = 0.0
    FLAGS.dropout = 0.0
    FLAGS.weight_decay = 0.0
    FLAGS.scheduler = None
    FLAGS.warmup = None

    # Important:
    # keep model_path during inference sanity-check,
    # but disable pretrained loading during training tiny-overfit runs
    if FLAGS.subtask == 'train':
        FLAGS.model_path = None

    FLAGS.save_model = True
    FLAGS.model_tag = f"tiny_overfit_{FLAGS.tiny_overfit_kernel}"
