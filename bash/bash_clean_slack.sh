#!/bin/bash
#
#
# Command: bash ./bash/bash_clean_slack.sh face_parsing
#
#
path_curr=$(pwd)
path_parent=$(dirname ${path_curr})
#
#
vujadeyoonbot_token_usr=xoxp-2693953505635-2687219708646-2691288389205-f9c6d4eb12660eb70fbb9816cd646b7b
vujadeyoonbot_token_bot=xoxb-2693953505635-2691254873461-k5pARFSSoS6y1721mBhvs2Br
#
#
channel=$1
token_usr=${2:-${vujadeyoonbot_token_usr}}
token_bot=${3:-${vujadeyoonbot_token_bot}}
#
#
PYTHONHASHSEED=0 python3 ${path_curr}/_main_clean_slack.py --token_usr ${token_usr} \
                                                           --token_bot ${token_bot} \
                                                           --channel ${channel}
