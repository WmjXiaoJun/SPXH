"""多格式 I/O：RAW / WAV / SigMF / npz bundle."""
from spxh.core.io.api import load_signal, save_signal, describe_file
from spxh.core.io.raw import RAW_DTYPES, infer_datatype, read_raw, write_raw
from spxh.core.io.sigmf import read_sigmf, write_sigmf
from spxh.core.io.wavio import read_wav, write_wav
from spxh.core.io.bundle import read_bundle, write_bundle

__all__ = [
    "RAW_DTYPES",
    "infer_datatype",
    "read_raw",
    "write_raw",
    "read_wav",
    "write_wav",
    "read_sigmf",
    "write_sigmf",
    "read_bundle",
    "write_bundle",
    "load_signal",
    "save_signal",
    "describe_file",
]
