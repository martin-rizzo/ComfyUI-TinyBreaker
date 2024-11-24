#!/usr/bin/env bash
# File    : venv-manager.sh
# Brief   : Utilities for managing Python virtual environments
# Author  : Martin Rizzo | <martinrizzo@gmail.com>
# Date    : Sep 17, 2024
# Repo    : https://github.com/martin-rizzo/PixArtToolkit
# License : MIT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                                Safetensors Kit
#     A set of simple commands to create, view, or analyze safetensors files
#
#     Copyright (c) 2024 Martin Rizzo
#
#     Permission is hereby granted, free of charge, to any person obtaining
#     a copy of this software and associated documentation files (the
#     "Software"), to deal in the Software without restriction, including
#     without limitation the rights to use, copy, modify, merge, publish,
#     distribute, sublicense, and/or sell copies of the Software, and to
#     permit persons to whom the Software is furnished to do so, subject to
#     the following conditions:
#
#     The above copyright notice and this permission notice shall be
#     included in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#     CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#     TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE
#     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
#
# FUNCTIONS:
#  - is_venv_active        : Checks if the Python virtual environment is active.
#  - require_venv          : Checks whether a given virtual environment exista y este bien configurado
#  - ensure_venv_is_active : Ensures the specified Python virtual environment is active.
#  - virtual_python        : Runs a command or Python script within the specified virtual environment.
#
#-----------------------------------------------------------------------------

# Stores the path to the last used virtual environment.
# This variable is used by the `virtual_python()` function.
HLP__LAST_VENV=

# ANSI escape codes for colored terminal output
RED='\e[91m'
GREEN='\e[92m'
YELLOW='\e[93m'
#BLUE='\e[94m'
#PURPLE='\e[95m'
CYAN='\e[96m'
DEFAULT_COLOR='\e[0m'


# Display a regular message
message() {
    local format=$1
    if [[ -z "$format" && -z "$2" ]]; then
        echo >&2
        return
    fi
    local prefix="  ${GREEN}>${DEFAULT_COLOR} "
    local suffix=""
    case "$format" in
        check) prefix="  ${GREEN}\xE2\x9C\x94 " ; suffix="${DEFAULT_COLOR}"    ; shift ;;
        wait ) prefix="  ${YELLOW}* "           ; suffix="...${DEFAULT_COLOR}" ; shift ;;
        info ) prefix="  ${CYAN}\xE2\x93\x98  " ; suffix="${DEFAULT_COLOR}"    ; shift ;;
    esac
    echo -e -n "$prefix" >&2
    echo    -n "$@"      >&2
    echo    -e "$suffix" >&2
}

# Display an error message
error() {
    local message=$1
    echo -e "${CYAN}[${RED}ERROR${CYAN}]${RED} $message${DEFAULT_COLOR}" >&2
}

# Displays a fatal error message and exits the script with status code 1
fatal_error() {
    local error_message=$1
    error "$error_message"
    shift

    # print informational messages, if any were provided
    while [[ $# -gt 0 ]]; do
        local info_message=$1
        echo -e " ${CYAN}\xF0\x9F\x9B\x88 $info_message${DEFAULT_COLOR}" >&2
        shift
    done
    exit 1
}

# Checks if the Python virtual environment is active.
#
# Usage:
#   is_venv_active <venv>
#
# Parameters:
#   - venv: the path to the virtual environment to check.
#
# Returns:
#   - 0 if the specified virtual environment is active
#   - 1 if no virtual environment is active
#   - 2 if a different virtual environment is active
#
# Example:
#   is_venv_active "/path/to/my-venv"
#
is_venv_active() {
    local venv=$1
    if [[ -z $VIRTUAL_ENV ]]; then
        return 1 # NO ACTIVE #
    fi
    if [[ "$venv" != *"${VIRTUAL_ENV#\~}" ]]; then
        return 2 # NO ACTIVE (otro venv esta activo) #
    fi
    return 0 # ACTIVE! #
}

# Checks whether a given virtual environment exists and is properly configured.
#
# Usage:
#   require_venv [--quiet] <venv> <python> <requirements>
#
# Parameters:
#   - --quiet     : (optional) if set, suppress output.
#   - venv        : the path to the virtual environment to be checked.
#   - python      : the command to execute python, usually 'python' but a different version could be used
#   - requirements: the path to the 'requirements.txt' file if the venv needs to be initialized with some dependencies
#
# Example:
#   require_venv "/path/to/my-venv"
#   require_venv --quiet "/path/to/my-venv"
#
require_venv() {

    # process all initial parameters that start with '-'
    local quiet=false
    while [[ "$1" =~ ^- ]]; do
        case "$1" in
        --quiet) quiet=true ;;
        -)       shift ; break ;;
        *)       fatal_error "require_venv does not support the parameter '$1'"  ;;
        esac
        shift
    done

    local venv_prompt="venv"
    local venv=$1 python=${2:-python} requirements=$3
    local update=false

    venv_prompt=$(basename "$venv")
    venv_prompt="${venv_prompt%-venv} venv"

    # if the venv does not exist, then create it
    if [[ ! -d $venv ]]; then
        message wait 'creating python virtual environment'
        message "'$python' -m venv '$venv' --prompt '$venv_prompt'"
        "$python" -m venv "$venv" --prompt "$venv_prompt"
        update=true
        message check 'new python virtual environment created:'
        message " - $venv"

    # if the venv already exists but contains a different version of Python,
    # then try to delete it and recreate it with the compatible version
    elif [[ ! -e "$venv/bin/$python" ]]; then
        $quiet || warning "a different version of python was selected ($python)"
        $quiet || message wait "recreating virtual environment"
        rm -Rf "$venv"
        "$python" -m venv "$venv" --prompt "$venv_prompt"
        update=true
        $quiet || message check "virtual environment recreated for $python"

    # if the venv exists and has the correct version of Python, do nothing!
    else
        $quiet || message check 'virtual environment exists'
    fi

    HLP__LAST_VENV=$venv
    if [[ $update == 'true' ]]; then
        message wait 'updating pip version'
        virtual_python "$venv" !pip install --upgrade pip
        message check 'pip version updated'
        [[ $requirements ]] && message wait 'installing requirements.txt'
        [[ $requirements ]] && virtual_python "$VENV_DIR" !pip install -r "$requirements"
        [[ $requirements ]] && message check 'requirements.txt installed'
        echo
        echo
    fi
}

# Ensures the specified Python virtual environment is active.
#
# Usage:
#   ensure_venv_is_active [--quiet] <venv>
#
# Parameters:
#   * --quiet: (optional) if set, suppress output.
#   * venv: the path to the Python virtual environment to be activated.
#
# Example:
#   ensure_venv_is_active "/path/to/my-venv"
#   ensure_venv_is_active --quiet "/path/to/my-venv"
#
ensure_venv_is_active() {

    # process all initial parameters that start with '-'
    local quiet=false
    while [[ "$1" =~ ^- ]]; do
        case "$1" in
        --quiet) quiet=true ;;
        -)       shift ; break ;;
        *)       fatal_error "ensure_venv_is_active does not support the parameter '$1'"  ;;
        esac
        shift
    done

    local venv=$1

    if is_venv_active "$venv"; then
        $quiet || message check "virtual environment already activated"
        return
    fi

    if [[ $? -eq 2 ]]; then
        fatal_error \
            "function ensure_venv_is_active() is unable to switch between virtual environments" \
            "This is an internal error likely caused by a mistake in the code"
    fi

    $quiet || message wait 'activating virtual environment'
    # shellcheck source=/dev/null
    source "$venv/bin/activate"
    $quiet || message check 'virtual environment activated'
}


# Runs a command or Python script within the specified virtual environment.
#
# Usage:
#   virtual_python [<venv>] <command> [args...]
#
# Parameters:
#   - venv (optional):
#      The path to the Python virtual environment to use.
#      If not provided, the function will use the last used virtual environment.
#   - command:
#      - "CONSOLE": Opens an interactive shell in the virtual environment.
#      - Command starting with "!": Runs the specified System command.
#      - Otherwise                : Runs the specified Python script.
#   - args...:
#      Additional arguments to pass to the command or Python script.
#
# Returns:
#   The exit status of the executed command or Python script.
#
# Examples:
#   virtual_python "/path/to/my_venv" CONSOLE
#   virtual_python "/path/to/my_venv" my_script.py arg1 arg2
#   virtual_python "/path/to/my_venv" "!pip install numpy"
#   virtual_python my_script.py  # Uses the last used virtual environment
#
virtual_python() {
    local venv command

    if [[ -d "$1" ]]; then
        # if the first parameter is a directory, it's assumed to be the `venv`
        venv=$1
        command=$2
        shift 2
    else
        # if the first parameter is NOT a directory, the `venv` will be the last used one
        venv=$HLP__LAST_VENV
        command=$1
        shift 1
    fi

    [[ "$venv" ]] || \
        fatal_error "The venv parameter is not previously defined"

    [[ -f "$venv/bin/activate" ]] || \
        fatal_error "The virtual environment '$venv' does not exist." \
                    "Please ensure the virtual environment is correctly installed."

    # handle the CONSOLE command
    if [[ $command == 'CONSOLE' ]]; then
        # shellcheck source=/dev/null
        source "$venv/bin/activate" || return 1
        exec /bin/bash --norc -i
        # the exec command replaces the current shell, so the
        # following line is unreachable, it's included for clarity
        return $?
    fi

    # ensure that the virtual environment is activated and ready,
    # and store it as the last one used for future reference
    ensure_venv_is_active --quiet "$venv"
    HLP__LAST_VENV=$venv

    # if no command was provided, there is nothing to execute
    # (the function was simply used to activate the virtual environment)
    if [[ -z "$command" ]]; then
        return 0

    # if the command starts with "!", execute it as a python script
    elif  [[ "$command" == "!"* ]]; then
        "${command:1}" "$@"

    # otherwise, execute it as a normal linux command
    else
        python "$command" "$@"
    fi
}

