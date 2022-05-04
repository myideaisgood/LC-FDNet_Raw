import os
import cv2
import logging
import rawpy
from process_raw import DngFile

from config import parse_args

args = parse_args()

DATA_DIR = args.data_dir
DATASET = args.test_dataset

DO_JPEGXL = True
DO_FLIF = True

# Set up logger
filename = 'logs_conventional.txt'
logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

logging.info("="*100)
logging.info('DATASET : %s' % (DATASET))

imgs = []

for file in os.listdir(os.path.join(DATA_DIR, DATASET, 'test')):
    if file.endswith('.png'):
        imgs.append(os.path.join(DATA_DIR, DATASET, 'test', file))

imgs = sorted(imgs)

if DO_JPEGXL:
    avg_bpp = 0

    for imgname in imgs:

        img = cv2.imread(imgname, cv2.IMREAD_ANYDEPTH)
        h,w = img.shape

        os.system('jpegxl/build/tools/cjxl "%s" output.jxl -q 100' % (imgname))

        filesize = os.stat('output.jxl').st_size

        bpp = 8*filesize/(h*w)
        avg_bpp += bpp

        logging.info("%s : %.4f" % (imgname, bpp))

        os.system('rm %s' % ('output.jxl'))

    avg_bpp /= len(imgs)

    logging.info("JPEGXL Average BPP : %.4f" % (avg_bpp))

    logging.info("="*30)

if DO_FLIF:
    avg_bpp = 0

    for imgname in imgs:

        img = cv2.imread(imgname, cv2.IMREAD_ANYDEPTH)
        h,w = img.shape

        os.system('flif -e -Q100 "%s" output.flif' % (imgname))

        filesize = os.stat('output.flif').st_size

        bpp = 8*filesize/(h*w)
        avg_bpp += bpp

        logging.info("%s : %.4f" % (imgname, bpp))

        os.system('rm %s' % ('output.flif'))

    avg_bpp /= len(imgs)

    logging.info("FLIF Average BPP : %.4f" % (avg_bpp))

    logging.info("="*100)