#!/bin/bash
#
#
name_dataset=$1
name_experiment=$2
#
#
path_curr=$(pwd)
path_parent=$(dirname ${path_curr})
#
#
if [[ "${name_dataset}" == "" ]]; then
  echo "The name_dataset should be given."
  exit 1
fi
#
#
path_log=${path_curr}/saved/log
path_model=${path_curr}/saved/model
#
#
if [[ "${name_experiment}" == "" ]] || [[ "${name_experiment}" == "*" ]]; then
  path_target_log=${path_log}/${name_dataset}
  path_target_model=${path_model}/${name_dataset}
else
  path_target_log=${path_log}/${name_dataset}/${name_experiment}
  path_target_model=${path_model}/${name_dataset}/${name_experiment}
fi
#
#
if [[ "${name_experiment}" == "" ]]; then
  command="ls"
  echo "Mode: list the experiemnt (i.e. ls)."
  printf "\tCommand: %s %s\n" "${command}" "${path_target_log}"
  printf "\t\t" && ${command} ${path_target_log}
  printf "\tCommand: %s %s\n" "${command}" "${path_target_model}"
  printf "\t\t" && ${command} ${path_target_model}
else
  command="rm -rf"
  echo "Mode: remove the experiment (i.e. rm -rf)."
  printf "\tCommand: %s %s\n" "${command}" "${path_target_log}"
  ${command} ${path_target_log}
  printf "\tCommand: %s %s\n" "${command}" "${path_target_model}"
  ${command} ${path_target_model}
fi
