#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""">v<🗑回收站-删除一律进回收站（👤2026-10-21：「删除进回收站，别直接删」）。不用 rm / del / os.remove。
用法：python tools/trash.py <路径> [<路径> ...]
在 Python 里：from trash import trash; trash(path)
Windows 的 SHFileOperationW + FOF_ALLOWUNDO（不需要装包）；可以从回收站还原。
"""
import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

FO_DELETE, FOF_ALLOWUNDO, FOF_NOCONFIRMATION, FOF_SILENT, FOF_NOERRORUI = 3, 0x40, 0x10, 0x4, 0x400


class SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
                ("fFlags", ctypes.c_uint16), ("fAnyOperationsAborted", wintypes.BOOL), ("hNameMappings", ctypes.c_void_p),
                ("lpszProgressTitle", wintypes.LPCWSTR)]


def trash(*paths):
    """把文件 / 文件夹移进回收站；不存在的路径报错，不静默跳过。"""
    ps = [str(Path(p).resolve()) for p in paths]
    missing = [p for p in ps if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"不存在：{missing}")
    op = SHFILEOPSTRUCTW(wFunc=FO_DELETE, pFrom="\0".join(ps) + "\0\0",
                         fFlags=FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI)
    r = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if r != 0 or op.fAnyOperationsAborted:
        raise OSError(f"进回收站失败（代码 {r}）：{ps}")
    return ps


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in trash(*sys.argv[1:]):
        print("-> 回收站", p)
