from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    host = "127.0.0.1"
    port = 8000
    server = ThreadingHTTPServer((host, port), SimpleHTTPRequestHandler)
    print(f"Serving {repo_root}")
    print(f"Open http://{host}:{port}/scripts/eda_explorer.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
