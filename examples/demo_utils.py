"""
This file contains utility functions for downloading datasets.
The code in this file is partially taken from the torchvision package,
specifically, https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py.
We package it here to avoid users having to install the rest of torchvision.
It is licensed under the following license:

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import os.path
import hashlib
import zipfile
import torch
from typing import Any, Callable, Optional
from tqdm.auto import tqdm
from pathlib import Path


def gen_bar_updater(total) -> Callable[[int, int, int], None]:
    pbar = tqdm(total=total, unit="GiB", unit_scale=True, leave=False, desc="Downloading")

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_url(
    url: str,
    root: str,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    size: Optional[int] = None,
) -> None:
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:  # download the file
        try:
            print("Downloading " + url + " to " + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater(size))
        except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    " Downloading " + url + " to " + fpath
                )
                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater(size))
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")


def extract_archive(
    from_path: str, to_path: Optional[str] = None, remove_finished: bool = True
) -> None:
    if to_path is None:
        to_path = os.path.dirname(from_path)

    with zipfile.ZipFile(from_path, "r") as z:
        z.extractall(to_path)

    if remove_finished:
        os.remove(from_path)


def load_checkpoints(dataset, root) -> list:
    dataset_to_url = {
        "CIFAR10": "https://www.dropbox.com/sh/auu7so63xy77fos/AADkSCwmTRZ_GCYPfsdRYUkwa?dl=1",
        "MSCOCO": "https://www.dropbox.com/scl/fo/xix2zrtge05ppvw71v0ut/h?rlkey=cxqdom2ijmyvpe4jtsfapqn8d&dl=1",
    }
    path = Path(root).joinpath(f"{dataset}_models.zip")
    ckpt_dir = Path(root).joinpath(f"{dataset}_models")
    if not os.path.exists(ckpt_dir):
        download_url(dataset_to_url[dataset], root, f"{dataset}_models.zip")
        extract_archive(from_path=path, to_path=ckpt_dir)
    else:
        print("Models already downloaded and extracted.")
    ckpts = []
    for ckpt_path in sorted(list(ckpt_dir.iterdir())):
        ckpts.append(torch.load(ckpt_path))
    return ckpts


def load_trak_results(dataset, root):
    dataset_to_url = {
        "CIFAR10": "https://www.dropbox.com/sh/k2ndo039iuiik9e/AACRIh_BwF6v4Y2Y20ySmLaQa?dl=1",
        "MSCOCO": "https://www.dropbox.com/sh/p6eujiigdi8xrb1/AABw1idEeZgXI7SlI9XnDVQta?dl=1",
    }

    NAME = f"{dataset}_trak_results.zip"
    path = Path(root).joinpath(NAME)
    ckpt_dir = Path(root).joinpath(f"{dataset}_trak_results")
    if not os.path.exists(ckpt_dir):
        download_url(dataset_to_url[dataset], root, NAME)
        extract_archive(from_path=path, to_path=ckpt_dir)
    else:
        print("Models already downloaded and extracted.")
    pass
