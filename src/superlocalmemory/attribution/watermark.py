# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Steganographic watermarking using zero-width Unicode characters.

Embeds an invisible binary payload (derived from a key string) into
visible text.  The watermark is undetectable to end users but can be
extracted programmatically to prove provenance.

Encoding scheme
---------------
1. Convert the key to binary (each char -> 8-bit representation).
2. Map binary digits to zero-width characters:

   - ``0`` -> U+200B  (zero-width space)
   - ``1`` -> U+200C  (zero-width non-joiner)

3. Frame the bit sequence with markers:

   - Start: U+FEFF  (byte-order mark)
   - End:   U+200D  (zero-width joiner)

4. Insert the entire encoded sequence immediately after the first
   visible character of the host text.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

from typing import Optional

# ---- Zero-width character constants ----
_BIT_ZERO: str = "\u200b"   # Zero-width space
_BIT_ONE: str = "\u200c"    # Zero-width non-joiner
_END_MARKER: str = "\u200d"  # Zero-width joiner
_START_MARKER: str = "\ufeff"  # Byte-order mark

# All zero-width characters used by this module (for stripping).
_ALL_ZW: set[str] = {_BIT_ZERO, _BIT_ONE, _END_MARKER, _START_MARKER}


class QualixarWatermark:
    """Embed and detect invisible watermarks in text.

    Typical usage::

        wm = QualixarWatermark()
        watermarked = wm.embed("Hello world")
        assert wm.detect(watermarked) is True
        assert wm.extract(watermarked) == "qualixar"
        assert wm.strip(watermarked) == "Hello world"

    Args:
        key: The string payload to embed.  On extraction, this exact
            string is recovered if the watermark is intact.
    """

    def __init__(self, key: str = "qualixar") -> None:
        if not key:
            raise ValueError("key must be a non-empty string")
        self._key: str = key
        self._encoded: str = self._encode_key(key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> str:
        """Embed an invisible watermark into *text*.

        The returned string is visually identical to the original when
        rendered — zero-width characters are invisible in all standard
        fonts and terminals.

        Args:
            text: The visible text to watermark.

        Returns:
            A new string containing the embedded watermark.  If *text*
            is empty, returns the empty string unchanged (nowhere to
            insert the payload).
        """
        if not text:
            return text

        # Insert the encoded payload after the first visible character.
        return text[0] + self._encoded + text[1:]

    def detect(self, text: str) -> bool:
        """Check whether *text* contains a valid watermark for this key.

        Args:
            text: The text to inspect.

        Returns:
            ``True`` if the watermark is present and matches the key,
            ``False`` otherwise.
        """
        extracted = self.extract(text)
        return extracted == self._key

    def extract(self, text: str) -> Optional[str]:
        """Extract the watermark payload from *text*.

        Args:
            text: The text to inspect.

        Returns:
            The decoded key string if a valid watermark is found,
            ``None`` if no watermark is present or it is malformed.
        """
        start_idx = text.find(_START_MARKER)
        if start_idx == -1:
            return None

        end_idx = text.find(_END_MARKER, start_idx + 1)
        if end_idx == -1:
            return None

        payload = text[start_idx + 1 : end_idx]
        return self._decode_payload(payload)

    def strip(self, text: str) -> str:
        """Remove all zero-width characters, returning clean text.

        Args:
            text: The potentially watermarked text.

        Returns:
            A copy of *text* with every zero-width character removed.
        """
        return "".join(ch for ch in text if ch not in _ALL_ZW)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_key(key: str) -> str:
        """Convert *key* to a framed zero-width character sequence.

        Each character of *key* is converted to 8 binary digits, each
        digit is mapped to a zero-width char, and the whole thing is
        wrapped with start/end markers.
        """
        bits: list[str] = []
        for char in key:
            byte_val = ord(char)
            for bit_pos in range(7, -1, -1):
                if (byte_val >> bit_pos) & 1:
                    bits.append(_BIT_ONE)
                else:
                    bits.append(_BIT_ZERO)

        return _START_MARKER + "".join(bits) + _END_MARKER

    @staticmethod
    def _decode_payload(payload: str) -> Optional[str]:
        """Decode a zero-width bit sequence back to a string.

        Args:
            payload: The raw zero-width characters between the start
                and end markers (exclusive).

        Returns:
            The decoded string, or ``None`` if the payload length is
            not a multiple of 8 or contains unexpected characters.
        """
        if len(payload) % 8 != 0:
            return None

        chars: list[str] = []
        for i in range(0, len(payload), 8):
            byte_bits = payload[i : i + 8]
            byte_val = 0
            for bit_char in byte_bits:
                byte_val <<= 1
                if bit_char == _BIT_ONE:
                    byte_val |= 1
                elif bit_char == _BIT_ZERO:
                    pass  # bit stays 0
                else:
                    return None  # unexpected character in payload
            chars.append(chr(byte_val))

        return "".join(chars)
