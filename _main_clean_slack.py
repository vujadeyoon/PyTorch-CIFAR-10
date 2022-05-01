import argparse
from vujade import vujade_slack as slack_


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Clean the channel of the slack.')
    parser.add_argument('--token_usr', type=str, required=True, help='Token for the user')
    parser.add_argument('--token_bot', type=str, required=True, help='Token for the bot')
    parser.add_argument('--channel', type=str, required=True, help='The name of the channel for the slack')
    args = parser.parse_args()

    return args


if __name__=='__main__':
    args = get_args()

    slack = slack_.Slack(_token_usr=args.token_usr, _token_bot=args.token_bot, _channel=args.channel)
    slack.clean_channel()
