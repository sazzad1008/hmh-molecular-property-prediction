#!/usr/bin/env bash
set -u

echo "============================================================"
echo "ASA-X RESOURCE AND CONFIGURATION CHECK"
echo "Generated: $(date)"
echo "============================================================"
echo

run_cmd() {
  echo
  echo "------------------------------------------------------------"
  echo "$ $*"
  echo "------------------------------------------------------------"
  "$@" 2>&1 || echo "[command failed or unavailable]"
}

echo "Basic identity"
echo "user: $(whoami 2>/dev/null)"
echo "host: $(hostname 2>/dev/null)"
echo "pwd:  $(pwd 2>/dev/null)"
echo "shell: ${SHELL:-unknown}"
echo

echo "Important environment variables"
echo "HOME=${HOME:-}"
echo "SCRATCH=${SCRATCH:-}"
echo "TMPDIR=${TMPDIR:-}"
echo "PBS_O_WORKDIR=${PBS_O_WORKDIR:-}"
echo "PBS_JOBID=${PBS_JOBID:-}"
echo

run_cmd uname -a
run_cmd df -h
run_cmd quota
run_cmd usage
run_cmd qlimits

echo
echo "Command availability"
for cmd in qsub qstat qdel qselect pbsnodes module python python3 pip pip3 gcc git nvidia-smi; do
  printf "%-12s" "$cmd"
  command -v "$cmd" 2>/dev/null || echo "not found"
done

run_cmd qstat -B
run_cmd qstat -Q
run_cmd qstat -Qf
run_cmd qstat -u "$(whoami)"
run_cmd pbsnodes -a

if type module >/dev/null 2>&1; then
  echo
  echo "------------------------------------------------------------"
  echo "$ module avail"
  echo "------------------------------------------------------------"
  set +u
  module avail 2>&1 || echo "[module avail failed]"
  set -u
  echo
  echo "------------------------------------------------------------"
  echo "$ module list"
  echo "------------------------------------------------------------"
  set +u
  module list 2>&1 || echo "[module list failed]"
  set -u
else
  echo
  echo "module command not available in this shell"
fi

run_cmd python --version
run_cmd python3 --version
run_cmd pip --version
run_cmd pip3 --version
run_cmd gcc --version
run_cmd git --version
run_cmd nvidia-smi
run_cmd lscpu
run_cmd free -h

echo
echo "Recommended next step:"
echo "  Paste this report back to tune PBS queue/resource settings."

