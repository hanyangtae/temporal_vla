from __future__ import annotations

import base64

import numpy as np

from src.utils.common.feature_blob import (
    decode_feature_blob,
    decode_legacy_feature_array,
    encode_feature_blob,
    encode_legacy_feature_array,
)


def test_feature_blob_round_trips_shape_dtype_and_data():
    hidden = np.arange(2 * 3 * 4, dtype=np.float16).reshape(2, 3, 4)

    blob = encode_feature_blob(hidden)
    decoded = decode_feature_blob(blob)

    assert blob["shape"] == [2, 3, 4]
    assert blob["dtype"] == "float16"
    assert base64.b64decode(blob["data"]) == hidden.tobytes()
    np.testing.assert_array_equal(decoded, hidden)


def test_decode_feature_blob_returns_mutable_copy():
    hidden = np.arange(4, dtype=np.float32)
    decoded = decode_feature_blob(encode_feature_blob(hidden))

    decoded[0] = 99.0

    assert hidden[0] == 0.0
    assert decoded[0] == 99.0


def test_legacy_feature_array_round_trips_np_save_base64():
    hidden = np.arange(2 * 5 * 3, dtype=np.float16).reshape(2, 5, 3)

    encoded = encode_legacy_feature_array(hidden)
    decoded = decode_legacy_feature_array(encoded)

    assert isinstance(encoded, str)
    np.testing.assert_array_equal(decoded, hidden)
