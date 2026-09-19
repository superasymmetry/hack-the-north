# Shared environment for everything that talks to the simulator. Sourced, not run.
#
#   VirtualGL     GPU rendering (see docs/setup.md §4)
#   DISPLAY=:1    the X session this machine runs on
#   MINESTUDIO_DIR  /tmp is emptied at every boot (systemd-tmpfiles "D /tmp"), which wipes
#                   the downloaded simulator engine -- keep it somewhere persistent
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# A .env at the repo root, so the tunnel host and token do not have to be exported by hand
# every session. It is loaded before the defaults below, so it can set DISPLAY or
# MINESTUDIO_DIR too; and a variable already in the environment wins over the file, so a
# one-off `AGENT_WS_URL=wss://other ./scripts/local_client.sh` still overrides it.
#
# Parsed rather than sourced. This file holds a token, and `source` on a secrets file
# executes whatever is in it -- a stray backtick in a pasted token is not a thing that
# should be able to run. .env is gitignored; .env.example is the documented template.
_mcagents_load_env() {
    local file="$1" line key value
    [[ -f "$file" ]] || return 0
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%$'\r'}"                              # tolerate CRLF
        line="${line#"${line%%[![:space:]]*}"}"           # drop leading whitespace
        if [[ -z "$line" || "$line" == '#'* ]]; then continue; fi
        line="${line#export }"
        if [[ "$line" != *=* ]]; then continue; fi
        key="${line%%=*}"
        value="${line#*=}"
        key="${key%"${key##*[![:space:]]}"}"
        if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z_0-9]*$ ]]; then continue; fi
        if [[ "$value" == \"*\" || "$value" == \'*\' ]]; then
            value="${value:1:${#value}-2}"                # quoted: keep it exactly
        else
            value="${value%"${value##*[![:space:]]}"}"    # bare: drop trailing whitespace,
        fi                                                # which in a token fails as a 1008
        if [[ -z "${!key:-}" ]]; then export "$key=$value"; fi
    done < "$file"
}
_mcagents_load_env "${MCAGENTS_ENV_FILE:-$REPO_ROOT/.env}"
unset -f _mcagents_load_env

export PATH="${PATH}:/opt/VirtualGL/bin"
export DISPLAY="${DISPLAY:-:1}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-$HOME/.minestudio}"
export MINESTUDIO_GPU_RENDER=1

# conda's activation hooks (openjdk's, at least) reference unset variables, so `set -u` has
# to come off for the duration and go back exactly as it was afterwards.
_mcagents_shell_opts="$(set +o)"
set +u
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$REPO_ROOT/.conda-env"
eval "$_mcagents_shell_opts"
unset _mcagents_shell_opts

cd "$REPO_ROOT"
