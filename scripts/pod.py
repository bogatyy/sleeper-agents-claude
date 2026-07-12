#!/usr/bin/env python3
"""Drive the RunPod pod over SSH via paramiko.
Usage:
  pod.py run "<command>"        # exec, stream output, exit with remote code
  pod.py put <local> <remote>   # upload file
  pod.py get <remote> <local>   # download file
  pod.py info                    # print connection info
"""
import json, os, socket, sys, time, paramiko
from urllib.parse import urlparse

POD_JSON = os.environ.get("POD_JSON", "/tmp/claude-0/-home-user-sleeper-agents-claude/c38ee35c-f2c4-5eee-9c88-8eb91efec031/scratchpad/pod.json")
KEY = os.path.expanduser("~/.ssh/runpod_zesty")

def info():
    return json.load(open(POD_JSON))

def proxy_socket(host, port, timeout=20):
    """Open a TCP tunnel to host:port through the local HTTP CONNECT proxy (egress is 443-only)."""
    prox = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if not prox:
        s = socket.create_connection((host, port), timeout=timeout); return s
    u = urlparse(prox)
    s = socket.create_connection((u.hostname, u.port or 443), timeout=timeout)
    if u.scheme == "https":
        import ssl
        s = ssl.create_default_context().wrap_socket(s, server_hostname=u.hostname)
    req = f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n"
    s.sendall(req.encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += s.recv(1)
    if b" 200 " not in resp.split(b"\r\n")[0]:
        raise OSError("proxy CONNECT failed: " + resp.split(b"\r\n")[0].decode(errors="replace"))
    s.settimeout(None)
    return s

def connect(retries=8, delay=5):
    inf = info()
    pk = paramiko.Ed25519Key.from_private_key_file(KEY)
    # Prefer RunPod proxy SSH (ssh.runpod.io) if configured; fall back to direct IP.
    px = inf.get("proxy_ssh")
    if px:
        host, port, user = px["host"], px["port"], px["user"]
    else:
        host, port, user = inf["ssh"]["ip"], inf["ssh"]["port"], "root"
    last = None
    for i in range(retries):
        try:
            sock = proxy_socket(host, port)
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(host, port=port, username=user, pkey=pk,
                      sock=sock, timeout=25, banner_timeout=35, auth_timeout=35,
                      look_for_keys=False, allow_agent=False)
            return c
        except Exception as e:
            last = e
            sys.stderr.write(f"[connect retry {i}] {type(e).__name__}: {e}\n")
            time.sleep(delay)
    raise SystemExit(f"could not connect: {last}")

def run(cmd):
    c = connect()
    tr = c.get_transport()
    tr.set_keepalive(30)
    chan = tr.open_session()
    chan.get_pty()
    chan.exec_command(cmd)
    # stream
    buf = b""
    while True:
        if chan.recv_ready():
            data = chan.recv(65536)
            sys.stdout.buffer.write(data); sys.stdout.buffer.flush()
        if chan.exit_status_ready() and not chan.recv_ready():
            break
        time.sleep(0.05)
    # drain
    while chan.recv_ready():
        sys.stdout.buffer.write(chan.recv(65536)); sys.stdout.buffer.flush()
    code = chan.recv_exit_status()
    c.close()
    return code

def put(local, remote):
    c = connect(); sf = c.open_sftp()
    # ensure remote dir
    d = os.path.dirname(remote)
    if d:
        try: sf.stat(d)
        except IOError:
            # mkdir -p
            parts = d.strip("/").split("/"); cur=""
            for p in parts:
                cur += "/"+p
                try: sf.stat(cur)
                except IOError: sf.mkdir(cur)
    sf.put(local, remote)
    print(f"put {local} -> {remote} ({os.path.getsize(local)} bytes)")
    sf.close(); c.close()

def get(remote, local):
    c = connect(); sf = c.open_sftp()
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    sf.get(remote, local)
    print(f"get {remote} -> {local} ({os.path.getsize(local)} bytes)")
    sf.close(); c.close()

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        sys.exit(run(sys.argv[2]))
    elif cmd == "put":
        put(sys.argv[2], sys.argv[3])
    elif cmd == "get":
        get(sys.argv[2], sys.argv[3])
    elif cmd == "info":
        print(json.dumps(info(), indent=2))
    else:
        raise SystemExit("unknown cmd")
