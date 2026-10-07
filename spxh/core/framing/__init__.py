"""帧结构层：Barker 前导、32 位帧头、CRC-16-CCITT."""
from spxh.core.framing.barker import BARKER_13, barker_bits, barker_pm, correlate_bits, find_sync, verify_barker_property
from spxh.core.framing.crc import CRC16_INIT, CRC16_POLY, crc16_ccitt, crc16_ccitt_hex, check_crc16
from spxh.core.framing.format import (
    CRC_BITS,
    FEC_CODES,
    HEADER_BITS,
    HEADER_BYTES,
    MAX_PAYLOAD_BYTES,
    PREAMBLE_BITS,
    build_frame,
    frame_length_bits,
    header_bytes,
    parse_frame,
    parse_header_bytes,
    verify_frame,
)

__all__ = [
    "BARKER_13",
    "barker_bits",
    "barker_pm",
    "correlate_bits",
    "find_sync",
    "verify_barker_property",
    "CRC16_INIT",
    "CRC16_POLY",
    "crc16_ccitt",
    "crc16_ccitt_hex",
    "check_crc16",
    "CRC_BITS",
    "FEC_CODES",
    "HEADER_BITS",
    "HEADER_BYTES",
    "MAX_PAYLOAD_BYTES",
    "PREAMBLE_BITS",
    "build_frame",
    "frame_length_bits",
    "header_bytes",
    "parse_frame",
    "parse_header_bytes",
    "verify_frame",
]
