#!/bin/bash
set -e
exec python3 docker_entrypoint.py "$@"