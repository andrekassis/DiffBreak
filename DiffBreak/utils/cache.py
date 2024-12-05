import os
import yaml
import shutil
import pickle
import requests
from tqdm import tqdm, trange
import tarfile
from platformdirs import user_cache_dir
import io
import hashlib
from pathlib import Path
from rich.progress import BarColumn, DownloadColumn, Progress, TaskID, TimeElapsedColumn
from checksumdir import dirhash
from importlib import resources

import DiffBreak.resources.cache as resource_module

from .logs import get_logger

logger = get_logger()


class BufferedWriterWithProgress(io.BufferedWriter):
    def __init__(self, handle: io.BufferedWriter, progress, task_id):
        self.handle = handle
        self.progress = progress
        self.task_id = task_id
        self.total_written = 0
        self.total_read = 0

    def __enter__(self) -> "BufferedWriterWithProgress":
        self.handle.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def closed(self) -> bool:
        return self.handle.closed

    def close(self):
        self.handle.close()

    def fileno(self):
        return self.handle.fileno()

    def flush(self):
        self.handle.flush()

    def isatty(self) -> bool:
        return self.handle.isatty()

    def readable(self) -> bool:
        return self.handle.readable()

    def seekable(self) -> bool:
        return self.handle.seekable()

    def writable(self) -> bool:
        return True

    def read(self, size=-1) -> bytes:
        return self.handle.read(size)

    def read1(self, size=-1) -> bytes:
        return self.handle.read1()

    def readinto(self, b):
        return self.handle.readinto(b)

    def readinto1(self, b):
        return self.handle.readinto1(b)

    def readline(self, size=-1) -> bytes:
        return self.handle.readline(size)

    def readlines(self, hint=-1):
        return self.handle.readlines(hint)

    def write(self, b) -> int:
        n = self.handle.write(b)
        self.total_written += n
        self.progress.advance(self.task_id, n)
        return n

    def writelines(self, lines):
        return self.handle.writelines(lines)

    def seek(self, offset: int, whence: int = 0) -> int:
        pos = self.handle.seek(offset, whence)
        #  self.progress.update(self.task_id, completed=pos)
        return pos

    def tell(self) -> int:
        return self.handle.tell()

    @property
    def raw(self):
        return self.handle.raw

    def detach(self):
        return self.handle.detach()


class Client:
    def __init__(self, chunk_size=1024):
        self.chunk_size = chunk_size

    def get_resource(self, url, temp_file: io.BufferedWriter) -> None:
        response = requests.Session().get(
            url, headers={"user-agent": "Wget/1.16 (linux-gnu)"}, stream=True
        )
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                temp_file.write(chunk)


class ProgressFileObject(io.FileIO):
    def __init__(self, path, *args, **kwargs):
        io.FileIO.__init__(self, path, *args, **kwargs)
        self._total_size = os.path.getsize(path)
        self.started = False
        self.orig_path = path
        self.total_read = 0

        self.progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            DownloadColumn(),
        )

    def read(self, size):
        if not self.started:
            self.progress.start()
            self.task_id = self.progress.add_task(
                f"Extracting [cyan i]{self.orig_path.split('/')[-1]}[/]",
                total=self._total_size,
            )
            self.started = True

        read = io.FileIO.read(self, size)
        self.total_read += len(read)
        self.progress.advance(self.task_id, len(read))

        if self.total_read == self._total_size:
            self.progress.update(
                self.task_id,
                total=self.total_read,
                completed=self.total_read,
            )
            self.progress.stop()
        return read


def calc_checksum(path):
    assert os.path.exists(path)
    if os.path.isdir(path):
        return dirhash(path, "md5")
    hash = hashlib.md5()
    hash.update(Path(path).read_bytes())
    return hash.digest()


def download_file(url, destination, total_size, chunk_size):
    client = Client(chunk_size)
    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        DownloadColumn(),
    )
    progress.start()
    with open(destination, "wb") as cache_file:
        task_id = progress.add_task(
            f"Downloading [cyan i]{destination.split('/')[-1]}[/]", total=total_size
        )
        writer_with_progress = BufferedWriterWithProgress(cache_file, progress, task_id)
        client.get_resource(url, writer_with_progress)
        progress.update(
            task_id,
            total=writer_with_progress.total_written,
            completed=writer_with_progress.total_written,
        )
    progress.stop()


def extract(destination):
    target = "/".join(destination.split("/")[:-1])
    with tarfile.open(destination, "r:gz") as tarf:
        cached = os.path.join(target, tarf.getnames()[0].split("/")[0])
    with tarfile.open(fileobj=ProgressFileObject(destination)) as tar_file:
        tar_file.extractall(target)
    os.unlink(destination)
    return cached


def download_and_cache(f_name, entry, cache_dir=None, chunk_size=32768, strict=False):
    url, total_size = entry["url"], entry["size"]
    url = url + "/download"

    if cache_dir is None:
        cache_dir = os.path.join(user_cache_dir(), "DiffBreak", "hub")
    os.makedirs(cache_dir, exist_ok=True)
    destination = os.path.join(cache_dir, f_name)
    fldr = "/".join(destination.split("/")[:-1])
    if len(fldr) > 0:
        os.makedirs(fldr, exist_ok=True)

    if os.path.exists(os.path.join(cache_dir, ".metadata")):
        with open(os.path.join(cache_dir, ".metadata"), "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = {}

    cached = metadata.get(destination)
    if cached is not None:
        if strict:
            logger.info(f"Verifying cached resource {f_name}.")
            checksum = calc_checksum(cached)
            if checksum == entry["checksum"]:
                return cached
            else:
                logger.info("Failed. Downloading resouce from scratch instead.")
        else:
            return cached

    download_file(url, destination, total_size, chunk_size)
    cached = extract(destination) if tarfile.is_tarfile(destination) else destination

    metadata[destination] = cached
    with open(os.path.join(cache_dir, ".metadata"), "wb") as f:
        pickle.dump(metadata, f)

    if strict:
        logger.info(f"Verifying downloaded resource {f_name}.")
        checksum = calc_checksum(cached)
        assert checksum == entry["checksum"]

    return cached


def resolve_name(resource_name, resource_type):
    if resource_type == "classifier":
        resource_file = "classifiers.yaml"
        prefix = "checkpoints/classifiers"
    elif resource_type == "diffusion_model":
        resource_file = "diffusers.yaml"
        prefix = "checkpoints/diffusion"
    elif resource_type == "dataset":
        resource_file = "datasets.yaml"
        prefix = "datasets"
    elif resource_type == "dataset_index":
        resource_file = "dataset_index.yaml"
        prefix = "datasets/idx"
    else:
        raise NotImplementedError

    with resources.path(resource_module, resource_file) as fspath:
        resource_path = fspath.as_posix()

    with open(resource_path) as f:
        registry = yaml.load(f, Loader=yaml.Loader)

    resource_name_splits = resource_name.split("/")
    entry = registry[resource_name_splits[0]]
    for i in range(1, len(resource_name_splits)):
        entry = entry[resource_name_splits[i]]

    mid = "/".join(resource_name_splits[:-1]) if len(resource_name_splits) > 1 else ""
    return os.path.join(prefix, mid, entry["name"]), entry


def load_from_cache(resource_name, resource_type, cache_dir=None, index_only=False):
    assert resource_type in ["classifier", "dataset", "diffusion_model"]

    strict = True  # not ".tar.gz" in name
    strict_data = True  # not ".tar.gz" in name

    if resource_type == "dataset":
        if not index_only:
            name, entry = resolve_name(resource_name.split("/")[0], resource_type)
            download_and_cache(name, entry, strict=strict_data, cache_dir=cache_dir)
        resource_type = "dataset_index"
        resource_name = resource_name.split("/")
        resource_name[0] = resource_name[0].replace("-", "")
        resource_name = "/".join(resource_name)

    name, entry = resolve_name(resource_name, resource_type)
    return download_and_cache(name, entry, strict=strict, cache_dir=cache_dir)


def load_dataset_from_cache(resource_name, cache_dir=None, index_only=False):
    return load_from_cache(
        resource_name, "dataset", cache_dir=cache_dir, index_only=index_only
    )


def load_dm_from_cache(resource_name, cache_dir=None):
    return load_from_cache(resource_name, "diffusion_model", cache_dir=cache_dir)


def load_classifier_from_cache(resource_name, cache_dir=None):
    return load_from_cache(resource_name, "classifier", cache_dir=cache_dir)
