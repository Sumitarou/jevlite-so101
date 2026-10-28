#!/usr/bin/env bash
# Install the pre-commit secret check into this clone (hooks are not copied by git clone).
#
#   bash scripts/install_hooks.sh
#
# 1. copies scripts/hooks/pre-commit to .git/hooks/pre-commit
# 2. creates/extends .git/info/secret-patterns (outside version control) with values detected
#    on THIS machine: host name, user name, IP addresses (except loopback, link-local and the
#    Docker default 172.17.0.1), camera serial numbers from /dev/v4l/by-id.
#    The values are written to the file only; they are never printed.
# 3. add anything else by hand (e.g. a personal e-mail address), one "label<TAB>value" per line:
#      printf 'personal-email\t%s\n' 'you@example.com' >> .git/info/secret-patterns
#
# Generic strings such as "192.168." or "/home/" are deliberately not added: they appear in
# docs and tests and would block harmless commits.
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
git_dir=$(git rev-parse --git-common-dir)  # shared by all worktrees
patterns="$git_dir/info/secret-patterns"

install -m 0755 "$repo/scripts/hooks/pre-commit" "$git_dir/hooks/pre-commit"
echo "installed $git_dir/hooks/pre-commit"

mkdir -p "$git_dir/info"
if [[ ! -f "$patterns" ]]; then
    printf '# label<TAB>value, one per line. Not version controlled. Never print this file.\n' \
        >"$patterns"
fi
chmod 600 "$patterns"

added=0
add() {  # add <label> <value>: append unless empty or already present
    local label=$1 value=$2
    [[ -z "$value" ]] && return 0
    if ! cut -f2- "$patterns" | grep -qxF -- "$value"; then
        printf '%s\t%s\n' "$label" "$value" >>"$patterns"
        added=$((added + 1))
    fi
}

add hostname "$(hostname)"
add username "$(id -un)"
for ip in $(hostname -I 2>/dev/null || true); do
    case "$ip" in
        127.* | ::1 | fe80:* | 172.17.0.1) ;;
        *:*) add ipv6 "$ip" ;;
        *) add ipv4 "$ip" ;;
    esac
done
for dev in /dev/v4l/by-id/*-video-index*; do
    [[ -e "$dev" ]] || continue
    serial=$(basename "$dev" | sed -nE 's/^usb-[0-9a-fA-F]{4}_.+_([^_]+)-video-index[0-9]+$/\1/p')
    add camera-serial "$serial"
done

total=$(grep -cv '^#' "$patterns" || true)
echo "$patterns: $added value(s) added, $total in total (values not shown)"
echo "add other values by hand if needed (see the comment at the top of this script)"
