import torch

import logging
import os
import sys
from time import time
from tqdm import tqdm

from config import parse_args
from utils.average_meter import AverageMeter
from utils.data_loaders import *
from utils.log_helpers import *
from utils.helpers import *

######### Configuration #########
######### Configuration #########
######### Configuration #########
args = parse_args()

# Design Parameters
HIDDEN_UNIT = args.hidden_unit

# Session Parameters
GPU_NUM = args.gpu_num
EMPTY_CACHE = args.empty_cache

# Directory Parameters
DATA_DIR = args.data_dir
DATASET = args.test_dataset
EXP_NAME = args.experiment_name
EXP_DIR = 'experiments/' + EXP_NAME
CKPT_DIR = os.path.join(EXP_DIR, args.ckpt_dir)
LOG_DIR = os.path.join(EXP_DIR, args.log_dir)
WEIGHTS = args.weights
ENC_DIR = args.encode_dir

# Check if directory does not exist
os.system('rm -rf "%s"' % (ENC_DIR))
create_path(ENC_DIR)

# Set up logger
filename = os.path.join(LOG_DIR, 'logs_encode.txt')
logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

for key,value in sorted((args.__dict__).items()):
    print('\t%15s:\t%s' % (key, value))
    logging.info('\t%15s:\t%s' % (key, value))

######### Configuration #########
######### Configuration #########
######### Configuration #########

# Set up Dataset
if 'SIDD' in DATASET:
    test_dataset = SIDD_Dataset(args, 'test')
elif 'mit' in DATASET:
    test_dataset = MIT_Dataset(args, 'test')

Collate_ = DivideCollate()

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    num_workers=2,
    shuffle=False,
    collate_fn=Collate_
)

color_names = ['Y','U','V', 'D']
loc_names = ['d', 'b', 'c']

# Set up networks
networks = setup_networks(color_names, loc_names, logging, HIDDEN_UNIT)

# Set up GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move the network to GPU if possible
if torch.cuda.is_available():
    network2cuda(networks, device, color_names, loc_names)

# Load the pretrained model if exists
if os.path.exists(os.path.join(CKPT_DIR, WEIGHTS)):
    logging.info('Recovering from %s ...' % os.path.join(CKPT_DIR, WEIGHTS))

    for color in color_names:
        path_name = 'network_' + color + '_' + WEIGHTS
        checkpoint = torch.load(os.path.join(CKPT_DIR, path_name))
        for loc in loc_names:
            networks[color][loc]['MSB'].load_state_dict(checkpoint['network_MSB' + color + '_' + loc])
            networks[color][loc]['LSB'].load_state_dict(checkpoint['network_LSB' + color + '_' + loc])
    logging.info('Recover completed.')
else:
    logging.info('No model to load')
    sys.exit(1)

# Inference Current Model
# Metric Holders
bitrates = get_AverageMeter(color_names, loc_names)
bitrates['total'] = {}
bitrates['total']['MSB'] = AverageMeter()
bitrates['total']['LSB'] = AverageMeter()

enc_times = {}
for loc in loc_names:
    enc_times[loc] = AverageMeter()

# Change networks to evaluation mode
network_set(networks, color_names, loc_names, set='eval')

# Read in image names
img_names = os.listdir(os.path.join(DATA_DIR, DATASET, 'test'))
img_names = sorted(img_names)

# FLIF metrics
flif_avg_bpp_msb = 0
flif_avg_bpp_lsb = 0
flif_avg_time = 0

from utils.enc_dec import *

with torch.no_grad():

    for batch_idx, data in enumerate(tqdm(test_dataloader)):

        img_a_msb, img_b_msb, img_c_msb, img_d_msb, img_a_lsb, img_b_lsb, img_c_lsb, img_d_lsb, ori_img, img_name, padding = data
        _, _, H, W = ori_img.shape
        padding = padding[0]
        img_name = img_name[0]

        # Modify image name lena.png -> lena
        img_name = modify_imgname(img_name)
        img_name_wo_space = img_name.replace(" ","")

        # Create directory to save compressed file
        create_path(os.path.join(ENC_DIR, img_name_wo_space))

        # Encode padding
        encode_padding(padding, img_name_wo_space, ENC_DIR)       

        # Encode img_a by flif
        flif_bpp_msb, flif_time_msb = encode_flif(img_a_msb, img_name, img_name_wo_space, H, W, ENC_DIR, isMSB=True)
        flif_bpp_lsb, flif_time_lsb = encode_flif(img_a_lsb, img_name, img_name_wo_space, H, W, ENC_DIR, isMSB=False)

        flif_avg_bpp_msb += flif_bpp_msb
        flif_avg_bpp_lsb += flif_bpp_lsb
        flif_avg_time += (flif_time_msb + flif_time_lsb)

        # Data to cuda
        [img_a_msb, img_b_msb, img_c_msb, img_d_msb] = img2cuda([img_a_msb, img_b_msb, img_c_msb, img_d_msb], device)
        [img_a_lsb, img_b_lsb, img_c_lsb, img_d_lsb] = img2cuda([img_a_lsb, img_b_lsb, img_c_lsb, img_d_lsb], device)
        imgs_msb = abcd_unite(img_a_msb, img_b_msb, img_c_msb, img_d_msb, color_names)
        imgs_lsb = abcd_unite(img_a_lsb, img_b_lsb, img_c_lsb, img_d_lsb, color_names)

        # Inputs / Ref imgs / GTs
        inputs_msb = get_inputs(imgs_msb, color_names, loc_names)
        ref_imgs_msb = get_refs(imgs_msb, color_names)
        gt_imgs_msb = get_gts(imgs_msb, color_names, loc_names)

        inputs_lsb = get_inputs(imgs_lsb, color_names, loc_names)
        ref_imgs_lsb = get_refs(imgs_lsb, color_names)
        gt_imgs_lsb = get_gts(imgs_lsb, color_names, loc_names)        

        for loc in loc_names:

            start_time = time()

            for color in color_names:
                # ==================================== MSB ====================================#
                # Feed to network
                cur_network = networks[color][loc]['MSB']
                cur_inputs = inputs_msb[color][loc]
                cur_gt_img = gt_imgs_msb[color][loc]
                cur_ref_img = ref_imgs_msb[color]                

                # Low Frequency Compressor
                _, q_res_L, error_var_map, error_var_th, mask_L, pmf_softmax_L = cur_network(cur_inputs, cur_gt_img, cur_ref_img, frequency='low', mode='eval')
                mask_H = 1-mask_L

                gt_L = mask_L * cur_gt_img
                input_H = torch.cat([cur_inputs, gt_L], dim=1)

                _, q_res_H, pmf_softmax_H = cur_network(input_H, cur_gt_img, cur_ref_img, frequency='high', mode='eval')

                # Encode by torchac
                bpp_L = encode_torchac(pmf_softmax_L, q_res_L, mask_L, img_name_wo_space, color, loc, H, W, ENC_DIR, EMPTY_CACHE, frequency='low', isMSB=True)
                bpp_H = encode_torchac(pmf_softmax_H, q_res_H, mask_H, img_name_wo_space, color, loc, H, W, ENC_DIR, EMPTY_CACHE, frequency='high', isMSB=True)

                bpp_MSB = bpp_L + bpp_H

                bitrates[color][loc]['MSB'].update(bpp_MSB)

                if EMPTY_CACHE:
                    del q_res_L, pmf_softmax_L, q_res_H, pmf_softmax_H
                    torch.cuda.empty_cache()

                # ==================================== LSB ====================================#
                # Feed to network
                cur_network = networks[color][loc]['LSB']
                cur_inputs = inputs_lsb[color][loc]
                cur_gt_img = gt_imgs_lsb[color][loc]
                cur_ref_img = ref_imgs_lsb[color]                

                # Low Frequency Compressor
                _, q_res_L, error_var_map, error_var_th, mask_L, pmf_softmax_L = cur_network(cur_inputs, cur_gt_img, cur_ref_img, frequency='low', mode='eval')
                mask_H = 1-mask_L

                gt_L = mask_L * cur_gt_img
                input_H = torch.cat([cur_inputs, gt_L], dim=1)

                _, q_res_H, pmf_softmax_H = cur_network(input_H, cur_gt_img, cur_ref_img, frequency='high', mode='eval')

                # Encode by torchac
                bpp_L = encode_torchac(pmf_softmax_L, q_res_L, mask_L, img_name_wo_space, color, loc, H, W, ENC_DIR, EMPTY_CACHE, frequency='low', isMSB=False)
                bpp_H = encode_torchac(pmf_softmax_H, q_res_H, mask_H, img_name_wo_space, color, loc, H, W, ENC_DIR, EMPTY_CACHE, frequency='high', isMSB=False)

                bpp_LSB = bpp_L + bpp_H

                bitrates[color][loc]['LSB'].update(bpp_LSB)

                if EMPTY_CACHE:
                    del q_res_L, pmf_softmax_L, q_res_H, pmf_softmax_H
                    torch.cuda.empty_cache()

            enc_time = time() - start_time
            enc_times[loc].update(enc_time)

        update_total(bitrates, color_names, loc_names)

        # Print Test Img Results
        log_img_info(logging, img_name, bitrates, flif_bpp_msb, flif_bpp_lsb, color_names, loc_names)

    flif_avg_bpp_msb /= len(img_names)
    flif_avg_bpp_lsb /= len(img_names)
    flif_avg_time /= len(img_names)

    log_dataset_info(logging, bitrates, flif_avg_bpp_msb, flif_avg_bpp_lsb, enc_times, flif_avg_time, color_names, loc_names, type='Avg')