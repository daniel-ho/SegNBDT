"""

Prepare PASCAL Context Dataset

This script was takene from https://github.com/zhanghang1989/PyTorch-Encoding

"""
import os
import shutil
import argparse
import tarfile

########################
### Helper Functions ###
########################

def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...'%(fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s"%url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None: # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))

    return fname

def mkdir(path):
    """make dir exists okay"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

###########################
# Pascal Context Download #
###########################

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
                tar.extractall(path=path)
        else:
            shutil.move(filename, os.path.join(path, 'VOCdevkit/VOC2010/'+os.path.basename(filename)))

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
    install_detail_api(args.download_dir)    
    download_pascal_ctx(args.download_dir, overwrite=False)
