#!/bin/bash
#
#
# Command: bash ./bash/bash_clean_slack.sh research
#
#
path_curr=$(pwd)
path_parent=$(dirname ${path_curr})
#
#
SLACK_TOKEN_USR="SECRET_SLACK_TOKEN_USR"
SLACK_TOKEN_BOT="SECRET_SLACK_TOKEN_BOT"
#
#
channel=$1
slack_token_usr=${2:-${SLACK_TOKEN_USR}}
slack_token_bot=${3:-${SLACK_TOKEN_BOT}}
#
#
PYTHONHASHSEED=0 python3 ${path_curr}/_main_clean_slack.py --token_usr ${slack_token_usr} \
                                                           --token_bot ${slack_token_bot} \
                                                           --channel ${channel}
