# Load the active Exness account profile. Sourced by the launchers.
#   demo -> ~/.exness/env        live -> ~/.exness/live.env
# The active profile is the word in ~/.exness/active (switch it from the dashboard's account screen).
_cg_profile=$(cat ~/.exness/active 2>/dev/null || echo demo)
if [ "$_cg_profile" = "live" ]; then _cg_env=~/.exness/live.env; else _cg_profile=demo; _cg_env=~/.exness/env; fi
if [ ! -f "$_cg_env" ]; then echo "missing $_cg_env ($_cg_profile account settings)"; return 1 2>/dev/null || exit 1; fi
source "$_cg_env"
export CG_ACCOUNT_PROFILE="$_cg_profile"
