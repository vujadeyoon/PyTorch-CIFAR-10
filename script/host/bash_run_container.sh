#!/bin/bash
#
#
# Dveloper: Sungjun Yoon
# Email: sungjun.yoon@lge.com
#
#
readonly path_curr=$(pwd)
readonly path_parents=$(dirname "${path_curr}")
#
#
readonly repo_tag="${1:-gid:latest}"
readonly volume_project="${2:-$(pwd):/home/dev/MakeInteriorDesign}"
readonly volume_dataset="${3:-/DATA/Dataset:/DATA/Dataset}"
readonly port="${4:-10001:11001}"
readonly env="${5:-/home/dev}"
readonly is_podman="${6:-true}"
#
#
if [ "${is_podman}" = "true" ]; then
  podman run -it \
             --rm \
             --privileged \
             --security-opt=label=disable \
             --hooks-dir=/usr/share/containers/oci/hooks.d/ \
             --volume /tmp/.X11-unix:/tmp/.X11-unix:ro \
             --env DISPLAY=unix$DISPLAY \
             --ipc=host \
             --net=host \
             --volume ${volume_project} \
             --volume ${volume_dataset} \
             --publish ${port} \
             --env ${env} \
             ${repo_tag} /bin/bash
else
  sudo docker run -it \
                  --rm \
                  --privileged \
                  --runtime nvidia \
                  --volume /tmp/.X11-unix:/tmp/.X11-unix:ro \
                  --env DISPLAY=unix$DISPLAY \
                  --ipc=host \
                  --net=host \
                  --volume ${volume_project} \
                  --volume ${volume_dataset} \
                  --publish ${port} \
                  --env ${env} \
                  ${repo_tag} /bin/bash
fi