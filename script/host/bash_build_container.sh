#!/bin/bash
#
#
# Developer: vujadeyoon
# Email: vujadeyoon@gmail.com
#
#
readonly path_curr=$(pwd)
readonly path_parents=$(dirname "${path_curr}")
#
#
readonly repo_tag="${1:-pthclassifcation:latest}"
readonly git_maintainer="${2:-vujadeyoon}"
readonly git_mail="${3:-vujadeyoon@gmail.com}"
readonly is_podman="${4:-true}"
#
#
if [ "${is_podman}" = "true" ]; then
  podman build --build-arg GIT_MAINTAINER=${git_maintainer} --build-arg GIT_EMAIL=${git_mail} -t ${repo_tag} .
else
  sudo docker build --build-arg GIT_MAINTAINER=${git_maintainer} --build-arg GIT_EMAIL=${git_mail} -t ${repo_tag} .
fi
