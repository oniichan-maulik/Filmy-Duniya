#!/usr/bin/env bash
# Run pytest but disable automatic plugin autoloading which can pick up unrelated
# setuptools entrypoints in this environment (e.g., ROS test plugins).
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec .venv/bin/pytest -q "$@"
