#!/usr/bin/env bash
# Install the systemd user timer that runs bin/daily_run.sh for one schedule directory.
#
#   bin/schedule_run.sh [--dry-run] <schedule-dir>
#
# The schedule is read from <schedule-dir>/env.sh:
#   PYG_SCHEDULE_ONCALENDAR  required. A systemd calendar expression in this host's
#                            time zone, such as "Tue..Sat *-*-* 00:30:00". Without it
#                            nothing is installed, so a directory is only ever
#                            scheduled on purpose.
#   PYG_SCHEDULE_UNIT        optional. The unit's name; default pyg-daily.
#
# The service and the timer are rendered from bin/systemd/pyg-daily.service.example and
# bin/systemd/pyg-daily.timer.example, written to ~/.config/systemd/user/, and the timer
# is enabled and started. --dry-run prints both instead, and changes nothing.
#
# WHY A SYSTEMD TIMER, NOT CRON
# A oneshot service does not start again while its last run is still going, and a run
# takes hours; cron would start a second driver that starves the first. Stopping the
# unit stops the whole run, the Spark driver included. Persistent=true starts a run
# that was missed while the host was down once it is back up.
#
# A user timer fires only while its user has a session, unless linger is on for that
# user. This script does not turn linger on; it says so when it is off. Once, as root:
#   loginctl enable-linger <user>
#
# Nothing in this file names a host or a path.
set -uo pipefail
# Bash 5.2 reads & in a ${var//pattern/replacement} replacement as the matched text.
shopt -u patsub_replacement 2>/dev/null || true

usage() {
  echo "usage: $(basename "$0") [--dry-run] <schedule-dir>" >&2
  exit 2
}

SD=""
DRY_RUN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage
      ;;
    *)
      [[ -z "$SD" ]] || usage
      SD="${1%/}"
      shift
      ;;
  esac
done
[[ -n "$SD" ]] || usage
if [[ ! -f "$SD/env.sh" ]]; then
  echo "no $SD/env.sh -- a schedule directory holds the env.sh its runs are built from" >&2
  exit 2
fi
SD="$(cd "$SD" && pwd)"
TEMPLATES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/systemd"

refuse() { echo "not scheduled: $*" >&2; exit 2; }

# The stand-ins bin/daily_run.sh reads its settings with: env.sh may build paths from
# RUN_ID and the day, and fail on purpose while they are unset.
setting() {
  RUN_ID=00000000T000000Z PYG_RUN_DIR=/nonexistent \
    PYG_DATA_YEAR=0000 PYG_DATA_MONTH=01 PYG_DATA_DAY=01 \
    bash -c '. "$1" >/dev/null 2>&1; printf "%s" "${!2-}"' _ "$SD/env.sh" "$1"
}

CALENDAR="$(setting PYG_SCHEDULE_ONCALENDAR)"
[[ -n "$CALENDAR" ]] \
  || refuse "PYG_SCHEDULE_ONCALENDAR is not set in $SD/env.sh, so this directory is not a schedule"
UNIT="$(setting PYG_SCHEDULE_UNIT)"
UNIT="${UNIT:-pyg-daily}"
[[ "$UNIT" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
  || refuse "PYG_SCHEDULE_UNIT must be a plain unit name, got '$UNIT'"
REPO="$(setting PYG_REPO_ROOT)"
DAILY="$REPO/bin/daily_run.sh"
[[ -n "$REPO" && -x "$DAILY" ]] \
  || refuse "PYG_REPO_ROOT in env.sh has no executable bin/daily_run.sh: '$REPO'"
# A unit file expands $ and reads " and \ in a command line, so a path that holds any
# of them would not arrive as written.
for path in "$DAILY" "$SD"; do
  [[ "$path" != *[\"\\\$]* && "$path" != *$'\n'* ]] \
    || refuse "a unit file cannot carry this path as it is: '$path'"
done

# Checked here, where a typo costs nothing, rather than by a timer that never fires.
if ! checked="$(systemd-analyze calendar "$CALENDAR" 2>&1)"; then
  refuse "systemd does not accept PYG_SCHEDULE_ONCALENDAR='$CALENDAR': ${checked##*$'\n'}"
fi

render() {   # a template -> the unit, on stdout
  local text
  text="$(<"$1")" || return 1
  # A unit file reads % as the start of a specifier, so a path's % is doubled.
  text="${text//@UNIT@/$UNIT}"
  text="${text//@ONCALENDAR@/$CALENDAR}"
  text="${text//@DAILY_RUN@/${DAILY//%/%%}}"
  text="${text//@SCHEDULE_DIR@/${SD//%/%%}}"
  if [[ "$text" =~ @[A-Z_]+@ ]]; then
    echo "$1 has a placeholder this script does not fill: ${BASH_REMATCH[0]}" >&2
    return 1
  fi
  printf '%s\n' "$text"
}
service="$(render "$TEMPLATES/pyg-daily.service.example")" || refuse "cannot render the service"
timer="$(render "$TEMPLATES/pyg-daily.timer.example")" || refuse "cannot render the timer"

DEST="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
if [[ -n "$DRY_RUN" ]]; then
  printf '# %s\n%s\n\n# %s\n%s\n' "$DEST/$UNIT.service" "$service" "$DEST/$UNIT.timer" "$timer"
  exit 0
fi

mkdir -p "$DEST" || refuse "cannot create $DEST"
printf '%s\n' "$service" > "$DEST/$UNIT.service" || refuse "cannot write $DEST/$UNIT.service"
printf '%s\n' "$timer" > "$DEST/$UNIT.timer" || refuse "cannot write $DEST/$UNIT.timer"
systemctl --user daemon-reload || refuse "systemctl --user daemon-reload failed"
systemctl --user enable --now "$UNIT.timer" || refuse "could not enable $UNIT.timer"
echo "installed $DEST/$UNIT.service and $DEST/$UNIT.timer for $SD"
systemctl --user list-timers --no-pager "$UNIT.timer" 2>/dev/null || true

USER_NAME="$(id -un)"
if [[ "$(loginctl show-user "$USER_NAME" --property=Linger --value 2>/dev/null)" != "yes" ]]; then
  echo
  echo "NOTE: linger is off for $USER_NAME, so this timer fires only while $USER_NAME is"
  echo "      logged in. Turn it on once, as root: loginctl enable-linger $USER_NAME"
fi
