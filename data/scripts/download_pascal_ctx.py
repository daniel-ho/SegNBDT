"""

Prepare PASCAL Context Dataset

This script was taken from https://github.com/zhanghang1989/PyTorch-Encoding
and modified for SegNBDT purposes.

"""
import os
import shutil
import argparse
import tarfile

from download_utils import download, mkdir

def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize PASCAL Context dataset.')
    parser.add_argument('--download-dir', default='./data/pascal_ctx', help='directory to download data')
    args = parser.parse_args()
    return args

def download_pascal_ctx(download_dir, overwrite=False):
    _AUG_DOWNLOAD_URLS = [
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
         'bf9985e9f2b064752bf6bd654d89f017c76c395a'),
        ('https://codalabuser.blob.core.windows.net/public/trainval_merged.json',
         '169325d9f7e9047537fedca7b04de4dddf10b881'),
        ('https://hangzh.s3.amazonaws.com/encoding/data/pcontext/train.pth',
         '4bfb49e8c1cefe352df876c9b5434e655c9c1d07'),
        ('https://hangzh.s3.amazonaws.com/encoding/data/pcontext/val.pth',
         'ebedc94247ec616c57b9a2df15091784826a7b0c'),
        ]
    for url, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)
        # extract
        if os.path.splitext(filename)[1] == '.tar':
            with tarfile.open(filename) as tar:
                tar.extractall(path=download_dir)
        else:
            shutil.move(filename, os.path.join(download_dir, 'VOCdevkit/VOC2010/'+os.path.basename(filename)))

def install_detail_api(download_dir):
    repo_url = "https://github.com/zhanghang1989/detail-api"
    os.system("git clone {} {}".format(repo_url, download_dir))
    os.system("cd {}/PythonAPI/ && python setup.py install".format(download_dir))
    try:
        import detail
    except Exception:
        print("Installing PASCAL Context API failed, please install it manually %s"%(repo_url))


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.download_dir, exist_ok=True)
    print("Installing Detail-API....")
    install_detail_api(args.download_dir)
    print("Downloading Pascal-Context Dataset....")
    download_pascal_ctx(args.download_dir, overwrite=False)
    print("Download Complete.")
