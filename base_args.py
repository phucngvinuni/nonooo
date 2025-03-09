import argparse

IMGC_NUMCLASS = 10   # CIFAR Data

def get_args():
    parser = argparse.ArgumentParser('T-DeepSC training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_freq', default=15, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--chep', default='', type=str,
                        help='chceckpint path')
    
    # Dataset parameters
    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument('--data_path', default='data/', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='cifar_S32', 
                        choices=['cifar_S32','cifar_S224', 'imagenet'],
                        type=str)
    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size for data')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--channel_type', default='none', choices=['none', 'awgn', 'rayleigh', 'rician'], type=str,
                        help='Type of channel to apply: none (FGSM), awgn, rayleigh, or rician (default: none)')
    parser.add_argument('--snr_db', default=20.0, type=float,
                        help='Signal-to-Noise Ratio in dB for channel noise (default: 20.0)')
            
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.02)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_lr', type=float, default=1e-4, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    parser.add_argument('--model', default='UDeepSC_Model', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)    
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=False)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--if_attack_train', action='store_true' ,help='active train attack')
    parser.add_argument('--if_attack_test', action='store_true', help='active test attack')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')

    # Augmentation parameters 
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    
    parser.add_argument('--save_ckpt', action='store_true')
    parser.set_defaults(save_ckpt=True)
    
    parser.add_argument('--train_type', default='std_train', choices=['std_train','fim_train'],
                        type=str)

    return parser.parse_args()