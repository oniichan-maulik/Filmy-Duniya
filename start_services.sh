#!/usr/bin/env bash
# Start/stop script for the Movie Recommender backend + Streamlit UI.
#
# Usage:
#   ./start_services.sh start    # start both services (no-op if port in use)
#   ./start_services.sh stop     # stop services started by this script
#   ./start_services.sh status   # show status
#   ./start_services.sh restart  # stop then start

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="$ROOT/.venv/bin/python"
VENV_STREAMLIT="$ROOT/.venv/bin/streamlit"

LOGDIR="$ROOT/logs"
RUNDIR="$ROOT/.run"
mkdir -p "$LOGDIR" "$RUNDIR"

UVICORN_LOG="$LOGDIR/uvicorn.log"
STREAMLIT_LOG="$LOGDIR/streamlit.log"
UVICORN_PIDFILE="$RUNDIR/uvicorn.pid"
STREAMLIT_PIDFILE="$RUNDIR/streamlit.pid"

check_port() {
    local host=$1 port=$2
    # Prefer active TCP connect check; avoids false positives when `ss`
    # cannot access netlink in restricted environments.
    if command -v nc >/dev/null 2>&1; then
        if nc -z "$host" "$port" >/dev/null 2>&1; then
            return 0
        fi
    elif command -v curl >/dev/null 2>&1; then
        if curl -fsS --max-time 1 "http://$host:$port" >/dev/null 2>&1; then
            return 0
        fi
    else
        # Last-resort fallback to ss (suppress permission noise).
        if ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN; then
            return 0
        fi
    fi
    return 1
}

wait_for_port() {
    local host=$1 port=$2 name=$3
    local retries=${4:-20}
    local sleep_s=${5:-0.25}
    for _ in $(seq 1 "$retries"); do
        if check_port "$host" "$port"; then
            return 0
        fi
        sleep "$sleep_s"
    done
    echo "$name failed to start on $host:$port"
    return 1
}

start_uvicorn() {
    if check_port 127.0.0.1 8000; then
        echo "FastAPI already listening on :8000 — skipping start"
        return
    fi
    if [ ! -x "$VENV_PY" ]; then
        echo "Virtualenv python not found at $VENV_PY"
        exit 1
    fi
    echo "Starting FastAPI backend... (logs: $UVICORN_LOG)"
    nohup "$VENV_PY" -m uvicorn src.api:app --host 127.0.0.1 --port 8000 > "$UVICORN_LOG" 2>&1 &
    echo $! > "$UVICORN_PIDFILE"
    echo "FastAPI PID $(cat $UVICORN_PIDFILE)"
    if ! wait_for_port 127.0.0.1 8000 "FastAPI" 120 0.25; then
        echo "Last backend logs:"
        tail -n 60 "$UVICORN_LOG" || true
        exit 1
    fi
}

start_streamlit() {
    if check_port 127.0.0.1 8501; then
        echo "Streamlit already listening on :8501 — skipping start"
        return
    fi
    echo "Starting Streamlit... (logs: $STREAMLIT_LOG)"
    nohup "$VENV_STREAMLIT" run web/app_streamlit.py --server.port 8501 --server.headless true > "$STREAMLIT_LOG" 2>&1 &
    echo $! > "$STREAMLIT_PIDFILE"
    echo "Streamlit PID $(cat $STREAMLIT_PIDFILE)"
    if ! wait_for_port 127.0.0.1 8501 "Streamlit" 80 0.25; then
        echo "Last streamlit logs:"
        tail -n 80 "$STREAMLIT_LOG" || true
        exit 1
    fi
}

stop_pidfile() {
    local pidfile="$1"
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile") || true
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Killing PID $pid"
            kill "$pid" || true
            sleep 0.5
            if kill -0 "$pid" 2>/dev/null; then
                echo "PID $pid still alive, sending SIGKILL"
                kill -9 "$pid" || true
            fi
        fi
        rm -f "$pidfile"
    fi
}

cmd=${1:-start}
case "$cmd" in
    start)
        if [ ! -x "$VENV_PY" ]; then
            echo "Virtualenv python not found at $VENV_PY"
            echo "Create venv with: python3 -m venv .venv && source .venv/bin/activate && python3 -m pip install -r requirements.txt"
            exit 1
        fi
        start_uvicorn
        start_streamlit
        ;;
    stop)
        echo "Stopping services..."
        stop_pidfile "$UVICORN_PIDFILE"
        stop_pidfile "$STREAMLIT_PIDFILE"
        ;;
    status)
        echo "Uvicorn:"
        if check_port 127.0.0.1 8000; then
            echo "  Listening on :8000"
            if [ -f "$UVICORN_PIDFILE" ]; then echo "  PID: $(cat $UVICORN_PIDFILE)"; fi
        else
            echo "  Not listening on :8000"
        fi
        echo "Streamlit:"
        if check_port 127.0.0.1 8501; then
            echo "  Listening on :8501"
            if [ -f "$STREAMLIT_PIDFILE" ]; then echo "  PID: $(cat $STREAMLIT_PIDFILE)"; fi
        else
            echo "  Not listening on :8501"
        fi
        ;;
    restart)
        "$0" stop
        sleep 1
        "$0" start
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 2
        ;;
esac
