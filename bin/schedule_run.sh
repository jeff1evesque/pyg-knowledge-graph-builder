#!/usr/bin/env bash
# Install the systemd user timer that runs bin/daily_run.sh for one schedule directory.
#
#   bin/schedule_run.sh [--dry-run] <schedule-dir>
#
# The schedule is read from <schedule-dir>/env.sh:
#   PYG_SCHEDULE_ONCALENDAR  required. A systemd calendar expression, such as
#                            "Tue..Sat *-*-* 01:00:00 America/New_York"; without a time
#                            zone it is this host's time. Without the setting nothing is
#                            installed, so a directory is only ever scheduled on purpose.
#   PYG_SCHEDULE_UNIT        optional. The unit's name; default pyg-daily.
#
# In order: turn on linger for the user running this, render the service and the timer
# from bin/systemd/pyg-daily.service.example and bin/systemd/pyg-daily.timer.example,
# write them to ~/.config/systemd/user/, and enable and start the timer. --dry-run prints
# the units and what it would do about linger, and changes nothing.
#
# LINGER
# A user's systemd timers run only while that user has a login session, unless linger is
# on for the user. Without it a schedule looks installed and never fires once its user
# logs out. So this turns linger on (loginctl enable-linger), which a user can do for
# themselves on most systems. Where the system wants root for it, nothing is installed,
# and the message names the one command to run first.
#
# WHY A SYSTEMD TIMER, NOT CRON
# A oneshot service does not start again while its last run is still going, and a run
# takes hours; cron would start a second driver that starves the first. Stopping the
# unit stops the whole run, the Spark driver included. Persistent=true starts a run
# that was missed while the host was down once it is back up.
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

USER_NAME="$(id -un)"
linger_on() {
  [[ "$(loginctl show-user "$USER_NAME" --property=Linger --value 2>/dev/null)" == "yes" ]]
}

DEST="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
if [[ -n "$DRY_RUN" ]]; then
  printf '# %s\n%s\n\n# %s\n%s\n' "$DEST/$UNIT.service" "$service" "$DEST/$UNIT.timer" "$timer"
  if linger_on; then
    echo "# linger is already on for $USER_NAME"
  else
    echo "# would turn on linger for $USER_NAME: loginctl enable-linger $USER_NAME"
  fi
  exit 0
fi

# Linger before anything is written, so a system that will not allow it is not left with
# a timer that looks armed and never fires.
if linger_on; then
  echo "linger is already on for $USER_NAME"
else
  loginctl enable-linger "$USER_NAME" || true
  linger_on \
    || refuse "linger could not be turned on for $USER_NAME, and without it the timer fires only while $USER_NAME is logged in. Run once: sudo loginctl enable-linger $USER_NAME, then run this again"
  echo "turned on linger for $USER_NAME"
fi

mkdir -p "$DEST" || refuse "cannot create $DEST"
printf '%s\n' "$service" > "$DEST/$UNIT.service" || refuse "cannot write $DEST/$UNIT.service"
printf '%s\n' "$timer" > "$DEST/$UNIT.timer" || refuse "cannot write $DEST/$UNIT.timer"
systemctl --user daemon-reload || refuse "systemctl --user daemon-reload failed"
systemctl --user enable --now "$UNIT.timer" || refuse "could not enable $UNIT.timer"
echo "installed $DEST/$UNIT.service and $DEST/$UNIT.timer for $SD"
systemctl --user list-timers --no-pager "$UNIT.timer" 2>/dev/null || true
