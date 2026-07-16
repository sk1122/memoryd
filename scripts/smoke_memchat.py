#!/usr/bin/env python3
"""Smoke-test memchat by driving it through a pseudo-terminal.

Boots memchat in a pty, sends a :seed command and a chat message, captures
rendered frames, and quits with Esc. Verifies it doesn't crash and restores
the terminal. Prints the final captured frame.
"""
import os, pty, select, time, sys, errno

CMD = ["cargo", "run", "--bin", "memchat", "--", "--fresh", "--agent", "dev"]
CWD = "/Users/cosmix/code/memoryd"

import re
_ANSI = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[()][0AB]|\r")

def strip_ansi(data: bytes) -> str:
    return _ANSI.sub(b"", data).decode(errors="replace")

def last_frame(data: bytes) -> str:
    """Best-effort readable snapshot: strip ANSI escapes from the tail of the
    capture (the last full-screen redraw lives at the end of the buffer)."""
    text = strip_ansi(data)
    # keep the last ~60 lines
    lines = [ln for ln in text.split("\n")]
    return "\n".join(lines[-60:])

def read_for(fd, duration):
    chunks = []
    end = time.time() + duration
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], max(0.0, end - time.time()))
        if not r:
            continue
        try:
            data = os.read(fd, 65536)
        except OSError as e:
            if e.errno == errno.EIO:
                break
            raise
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks)

def main():
    pid, fd = pty.fork()
    if pid == 0:
        # child
        os.chdir(CWD)
        env = os.environ.copy()
        # no OPENAI_API_KEY -> echo mode (exercises memory pipeline without LLM)
        env.pop("OPENAI_API_KEY", None)
        os.execvpe(CMD[0], CMD, env)
        os._exit(127)
    # parent
    # set a window size so ratatui has a reasonable area
    import struct, fcntl, termios
    winsize = struct.pack("HHHH", 40, 120, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)

    # wait for initial render (allow time for cargo build + ONNX model load)
    boot = read_for(fd, 25.0)
    sys.stdout.write("=== boot ===\n")
    sys.stdout.write(last_frame(boot))
    sys.stdout.write("\n")

    # send :seed  (Enter in raw mode is \r, not \n)
    os.write(fd, b":seed\r")
    out1 = read_for(fd, 6.0)
    sys.stdout.write("\n=== after :seed ===\n")
    sys.stdout.write(last_frame(out1))
    sys.stdout.write("\n")

    # send a chat message
    os.write(fd, b"where does the user work?\r")
    out2 = read_for(fd, 12.0)
    sys.stdout.write("\n=== after chat msg ===\n")
    sys.stdout.write(last_frame(out2))
    sys.stdout.write("\n")

    # quit with Ctrl-C (single unambiguous byte; Esc has crossterm's delay)
    os.write(fd, b"\x03")
    tail = read_for(fd, 2.0)
    sys.stdout.write("\n=== after ctrl-c ===\n")
    sys.stdout.write(last_frame(tail))
    sys.stdout.write("\n")

    try:
        wpid, status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        wpid, status = pid, 0
    if wpid == 0:
        # still running — force kill
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        sys.stdout.write("\n[memchat did not exit on Ctrl-C; killed]\n")
        sys.exit(2)
    code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
    sys.stdout.write(f"\n[memchat exited with code {code}]\n")
    sys.exit(0 if code == 0 else 1)

if __name__ == "__main__":
    main()
