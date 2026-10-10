"""Tiny PDFs written with the standard library, for the document tests."""

from __future__ import annotations


def _stream(text: str) -> bytes:
    body = f"BT /F1 24 Tf 50 100 Td ({text}) Tj ET".encode("latin-1") if text else b""
    return b"<< /Length %d >>\nstream\n" % len(body) + body + b"\nendstream"


def make_pdf(pages: list[str], *, title: str = "", created: str = "", encrypted: bool = False) -> bytes:
    """One page per entry; an empty string is a blank page."""
    objs: list[bytes] = [b"", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        content_no, page_no = len(objs) + 1, len(objs) + 2
        objs.append(_stream(text))
        objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Contents %d 0 R "
                    b"/Resources << /Font << /F1 3 0 R >> >> >>" % content_no)
        kids.append(f"{page_no} 0 R")
    objs[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode()
    trailer_extra = b""
    if title or created:
        info = "<< " + (f"/Title ({title}) " if title else "") + (f"/CreationDate ({created}) " if created else "") + ">>"
        objs.append(info.encode("latin-1"))
        trailer_extra += b" /Info %d 0 R" % len(objs)
    if encrypted:
        pad = b"<" + b"ab" * 32 + b">"
        objs.append(b"<< /Filter /Standard /V 1 /R 2 /O " + pad + b" /U " + pad + b" /P -4 >>")
        trailer_extra += b" /Encrypt %d 0 R /ID [<00112233445566778899aabbccddeeff> <00112233445566778899aabbccddeeff>]" % len(objs)
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R%s >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, trailer_extra, xref)
    return bytes(out)
