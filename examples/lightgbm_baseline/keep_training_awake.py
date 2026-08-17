from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
SYNCHRONIZE = 0x00100000
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102


def main() -> None:
    parser = argparse.ArgumentParser(description="Prevent automatic sleep while a training PID is alive.")
    parser.add_argument("pid", type=int)
    args = parser.parse_args()

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.SetThreadExecutionState.argtypes = (wintypes.DWORD,)
    kernel32.SetThreadExecutionState.restype = wintypes.DWORD

    handle = kernel32.OpenProcess(SYNCHRONIZE, False, args.pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
            raise ctypes.WinError(ctypes.get_last_error())
        while True:
            result = kernel32.WaitForSingleObject(handle, 60_000)
            if result == WAIT_OBJECT_0:
                break
            if result != WAIT_TIMEOUT:
                raise ctypes.WinError(ctypes.get_last_error())
            if not kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        kernel32.CloseHandle(handle)


if __name__ == "__main__":
    main()
