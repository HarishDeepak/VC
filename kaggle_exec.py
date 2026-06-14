"""Execute code on a live Kaggle kernel via the Jupyter WebSocket API.

Usage:
    python kaggle_exec.py "print('hello')"
    echo "print('hello')" | python kaggle_exec.py

Update PROXY_HTTP to the current session URL whenever it changes.
"""
import json, sys, uuid
import urllib.request
import websocket

PROXY_HTTP = "https://kkb-production.jupyter-proxy.kaggle.net/k/327105664/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidHlwIjoiSldUIn0..3hSG2yoIP4IEpCtnD2ZtrA.i-ITSjXQ5fcFGAlxt8J8M8ut8dkgEJz6dZYIBhSWi_IBiHX0_gXJoOdekDOEAAyFhknJUqfzMOobZhZTxGLnLMSdiGrGiAB47WfI6fNsa_E7sfYajqRwxSQGXBtuG8U7DT7PJTqpykWaRTdCTJ-TaF80kNyfYF61KU-9VJt8WaoC92_wHfD4R0-8FxZgfHGGtaLb1jCWcx3ytlJernvkxdzDOVX4vt5ERAa-5aXBjYlmFYja4X3OHRNViKWub6Jw.9PWdA4eOrcQ7PdHsztUJWA/proxy"


def _get_kernel_id() -> str:
    url = PROXY_HTTP + "/api/kernels"
    with urllib.request.urlopen(url, timeout=15) as r:
        kernels = json.loads(r.read())
    if not kernels:
        raise RuntimeError("No running kernels found at " + url)
    return kernels[0]["id"]


def run(code: str, timeout: int = 7200) -> str:
    kernel_id = _get_kernel_id()
    ws_url = (PROXY_HTTP
              .replace("https://", "wss://")
              .replace("http://", "ws://")
              + f"/api/kernels/{kernel_id}/channels")

    msg_id = str(uuid.uuid4())
    outputs = []

    def on_message(ws, raw):
        msg = json.loads(raw)
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            return
        mtype = msg.get("msg_type", "")
        content = msg.get("content", {})
        if mtype == "stream":
            outputs.append(content.get("text", ""))
        elif mtype in ("execute_result", "display_data"):
            outputs.append(content.get("data", {}).get("text/plain", ""))
        elif mtype == "error":
            outputs.append("\n".join(content.get("traceback", [])))
        elif mtype == "execute_reply":
            ws.close()

    def on_open(ws):
        ws.send(json.dumps({
            "header": {
                "msg_id": msg_id,
                "msg_type": "execute_request",
                "username": "claude",
                "session": str(uuid.uuid4()),
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
            },
            "channel": "shell",
        }))

    ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_open=on_open)
    ws.run_forever(ping_interval=30, ping_timeout=15)
    return "".join(outputs)


if __name__ == "__main__":
    code = " ".join(sys.argv[1:]) if sys.argv[1:] else sys.stdin.read()
    sys.stdout.buffer.write(run(code).encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
