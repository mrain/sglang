import json
import threading
import time
from array import array
from urllib.request import Request, urlopen

import msgspec
import sglang_server


SERVER_ARGS = {
    "served_model_name": "dummy",
    "model_path": "dummy",
    "model_config": {"context_len": 4096},
    "skip_tokenizer_init": True,
    "tokenizer_worker_num": 1,
    "detokenizer_worker_num": 1,
}


def wait_for_ingress(server, timeout_s=5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        headers, ids_buf, lengths = server.recv_requests(16)
        if headers:
            return headers, ids_buf, lengths
        time.sleep(0.05)
    raise TimeoutError("No request reached Rust ingress before timeout")


def main():
    server = sglang_server.Server(
        bind="127.0.0.1:30123",
        server_args_json=json.dumps(SERVER_ARGS),
    )

    result = {}
    error = {}

    def call_generate():
        try:
            req = Request(
                "http://127.0.0.1:30123/generate",
                data=json.dumps(
                    {
                        "input_ids": [1, 2, 3],
                        "sampling_params": {"max_new_tokens": 2},
                        "stream": False,
                    }
                ).encode(),
                headers={"content-type": "application/json"},
                method="POST",
            )
            result["body"] = urlopen(req, timeout=10).read().decode()
        except Exception as exc:
            error["exception"] = exc

    thread = threading.Thread(target=call_generate)
    thread.start()

    try:
        headers, ids_buf, lengths = wait_for_ingress(server)
        header = msgspec.msgpack.decode(headers[0])
        rid = header[1]

        ids = array("q")
        ids.frombytes(ids_buf)

        print("rid:", rid)
        print("lengths:", lengths)
        print("input_ids:", list(ids))

        chunk = msgspec.msgpack.encode(
            {
                "rid": rid,
                "seq": 0,
                "token_ids": [10, 11],
                "finish_reason": "stop",
                "prompt_tokens": 3,
            }
        )
        if not server.push_chunk(chunk):
            raise RuntimeError("push_chunk returned False")

        thread.join(timeout=5)
        if thread.is_alive():
            raise TimeoutError("HTTP /generate call did not finish")
        if "exception" in error:
            raise error["exception"]

        print(result["body"])
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
