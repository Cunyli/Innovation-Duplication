# Port Configuration

Port selection for the static results server is now documented in **[GETTING_STARTED.md](GETTING_STARTED.md#option-4-serve-generated-artifacts)**.

In short:
- `./start_server.sh 8000` launches a simple HTTP server from `results/`.
- Pass another value (e.g. `./start_server.sh 8080`) to avoid conflicts.
- The script validates that the output directory exists before starting.

No additional configuration files are required.
