#!/usr/bin/env python3
"""Create a RunPod on-demand GPU pod with SSH, poll until ready. Saves pod info to scratchpad/pod.json."""
import json, os, sys, time, urllib.request, urllib.error

API = "https://api.runpod.io/graphql?api_key=" + os.environ["RUNPOD_API_KEY"]
PUBKEY = open(os.path.expanduser("~/.ssh/runpod_zesty.pub")).read().strip()
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
OUT = os.path.join(os.path.dirname(__file__), "..", "scratchpad_pod.json")
OUT = os.environ.get("POD_JSON", "/tmp/claude-0/-home-user-sleeper-agents-claude/c38ee35c-f2c4-5eee-9c88-8eb91efec031/scratchpad/pod.json")

GPU_FALLBACKS = [
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA A40",
]

def gql(query):
    req = urllib.request.Request(API, data=json.dumps({"query": query}).encode(),
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "curl/8.5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def create():
    env_kv = [("PUBLIC_KEY", PUBKEY), ("HF_TOKEN", os.environ.get("HF_TOKEN","")), ("VENICE_API_KEY", os.environ.get("VENICE_API_KEY",""))]
    env_str = ",".join('{key:"%s",value:"%s"}' % (k, v.replace('"','\\"')) for k, v in env_kv)
    for gpu in GPU_FALLBACKS:
        q = '''mutation { podFindAndDeployOnDemand(input: {
            cloudType: SECURE, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 80,
            minVcpuCount: 8, minMemoryInGb: 40, supportPublicIp: true, startSsh: true,
            gpuTypeId: "%s", name: "zesty-sleeper",
            imageName: "%s", ports: "22/tcp",
            env: [%s]
        }) { id imageName machineId costPerHr } }''' % (gpu, IMAGE, env_str)
        try:
            d = gql(q)
        except urllib.error.HTTPError as e:
            print("HTTPError for", gpu, e.read().decode()[:300]); continue
        if d.get("errors"):
            print("  no capacity on", gpu, "->", d["errors"][0].get("message","")[:120]); continue
        pod = d["data"]["podFindAndDeployOnDemand"]
        if pod:
            print("DEPLOYED on", gpu, "podId=", pod["id"], "costPerHr=", pod.get("costPerHr"))
            pod["gpuType"] = gpu
            return pod
    print("FAILED: no capacity on any GPU type"); sys.exit(1)

def poll(pod_id):
    q = '''query { pod(input:{podId:"%s"}) {
        id name desiredStatus lastStatusChange
        runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }
    } }''' % pod_id
    for i in range(60):  # up to ~10 min
        d = gql(q)
        p = d["data"]["pod"]
        rt = p.get("runtime")
        ports = (rt or {}).get("ports") or []
        ssh = [pp for pp in ports if pp["privatePort"] == 22 and pp.get("isIpPublic")]
        status = p.get("desiredStatus")
        print(f"  [{i}] status={status} runtime={'up' if rt else 'none'} ssh_ports={len(ssh)}")
        if ssh:
            s = ssh[0]
            return {"ip": s["ip"], "port": s["publicPort"]}
        time.sleep(10)
    return None

if __name__ == "__main__":
    pod = create()
    time.sleep(3)
    conn = poll(pod["id"])
    info = {"podId": pod["id"], "gpuType": pod.get("gpuType"), "costPerHr": pod.get("costPerHr"),
            "image": IMAGE, "ssh": conn}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(info, open(OUT, "w"), indent=2)
    print("POD INFO:", json.dumps(info))
    if not conn:
        print("WARNING: SSH not ready yet; re-poll later.")
