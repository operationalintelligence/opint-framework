#!/bin/bash


## Check code style of the project files using flake8 tool
## :author: Alexey Anisenkov
## :date: February 2019

## modes:
##  1. check modified but not yet commited files (--current), enabled by default
##  2. check files from last commit (--last-commit)
##  3. check all project files (--all)
##

## usage example:
## . check_code.sh
## . check_code.sh --commit
## . check_code.sh --all

#set -e

function check_files() {
    FILES=$1;
    if [[ ! -z "${FILES// }" ]]; then
        CMD="flake8 $FILES";
        echo 'DO EXEC cmd=' $CMD;
        flake8 $FILES;
    fi
}

## default values
check_current=1;
check_commit=0;
check_all=0;

for i in "$@"
do
case $i in
    --current)
    check_current=1
    shift
    ;;
    --last-commit)
    check_commit=1
    shift
    ;;
    --all)
    check_all=1
    shift
    ;;
    *)
          # unknown option
    ;;
esac
done

# check modified files (not commited yet)
if (($check_current)); then
    echo ' .. check modified files since last commit (not committed yet)'
    FILES=$(git ls-files --other --modified --exclude-standard | (grep .py || true))
    check_files "${FILES[@]}";
fi

# check files from last commit
if (($check_commit)); then
    echo ' .. check committed files from last commit'
    FILES=$(git diff-tree --no-commit-id --name-only -m -r HEAD | (grep .py || true))
    check_files "${FILES[@]}";
fi

# check all files from project
if (($check_all)); then
    echo ' .. check all commited files'
    FILES=$(git ls-tree --full-tree -r --name-only HEAD | (grep .py || true))
    check_files "${FILES[@]}";
fi

