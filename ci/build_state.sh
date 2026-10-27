#!/usr/bin/env bash
# Hands a build on from one CI job to the next: its directories go into compressed parts that the job uploads as an
# artifact, and the next job restores them at the same paths. Packing removes each file as soon as it is in the
# archive and unpacking removes each part as soon as it is extracted, so that neither needs much more disk than the
# build itself. GNU tar keeps the modification times (to the second; truncating every time keeps each "newer than"
# that Ninja relies on) and, run as root, the owners.
#
# usage: ci/build_state.sh pack [--part-size SIZE] --state DIR [--state DIR...] PATH...
#        ci/build_state.sh room MANIFEST DIR...
#        ci/build_state.sh unpack STATE
#        ci/build_state.sh space DIR GB
#   pack    archives each PATH (an absolute directory) into the parts STATE/<n>.tar.zst.<nnn> of SIZE (as split takes
#           it, default 2G) with zstd -3 and writes STATE/manifest; the PATHs are gone afterwards. STATE is the first
#           --state whose file system has the room: on the file system of a PATH, that PATH's largest file (files
#           leave as they are archived), elsewhere the whole PATH. Prints STATE.
#   room    prints the first DIR whose file system has the room to receive the parts that MANIFEST lists and to
#           unpack them; fails if none has
#   unpack  restores the PATHs of STATE/manifest, removing each part once it is extracted, then removes STATE
#   space   prints the free space of DIR's file system and fails unless at least GB gigabytes (10^9 bytes) are free
set -euo pipefail

# free space kept beyond the estimates: compressor and pipe buffers, the manifest, rounding to file system blocks
SLACK=$((1000 * 1000 * 1000))

usage() {
    sed -n '/^# usage:/,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//' >&2
    exit 2
}
die() { echo "error: $*" >&2; exit 1; }
gb() { awk -v b="$1" 'BEGIN { printf "%.1f GB", b / 1e9 }'; }

# the nearest existing directory at or above $1: the file system that $1 is or will be on
existing() {
    local dir=$1
    while [ ! -d "$dir" ]; do dir=$(dirname "$dir"); done
    printf '%s\n' "$dir"
}
device() { stat -c %d "$(existing "$1")"; }
available() { df -B1 --output=avail "$(existing "$1")" | tail -1 | tr -d ' '; }

# Fails, naming each file system and the room it lacks, unless the room in each of $@ (lines "PLACE BYTES"; a PLACE
# stands for its file system, whose needs add up) is free.
check_room() {
    local -A need=() place=()
    local where bytes dev lacks=()
    while read -r where bytes; do
        dev=$(device "$where")
        need[$dev]=$(( ${need[$dev]:-0} + bytes ))
        place[$dev]=$where
    done
    for dev in "${!need[@]}"; do
        bytes=$((need[$dev] + SLACK))
        if [ "$(available "${place[$dev]}")" -lt "$bytes" ]; then
            lacks+=("$(existing "${place[$dev]}") has $(gb "$(available "${place[$dev]}")") free, needs $(gb "$bytes")")
        fi
    done
    [ ${#lacks[@]} -eq 0 ] || { (IFS=';'; echo "${lacks[*]}"); return 1; }
}

# STATE/manifest: the line "part-bytes N", then one line "archive INDEX ARCHIVE_BYTES TREE_BYTES PATH" per PATH
read_manifest() {
    [ -f "$1" ] || die "no manifest $1"
    part_bytes=0 indices=() archive_bytes=() tree_bytes=() paths=()
    local kind index archive tree path
    while read -r kind index archive tree path; do
        case "$kind" in
            part-bytes) part_bytes=$index ;;
            archive) indices+=("$index") archive_bytes+=("$archive") tree_bytes+=("$tree") paths+=("$path") ;;
            *) die "unexpected line in $1: $kind" ;;
        esac
    done < "$1"
    [[ "$part_bytes" -gt 0 && ${#paths[@]} -gt 0 ]] || die "$1 lists no archives"
}

# The room that unpacking the manifest's parts from STATE ($1) needs, as lines for check_room; $2 is yes when the
# parts are on disk already. A tree on the file system of STATE grows as its parts are removed, so it needs only what
# it has beyond its archive, and one part in flight.
unpack_needs() {
    local i parts=$part_bytes
    [ "$2" = yes ] || for i in "${!paths[@]}"; do parts=$((parts + archive_bytes[i])); done
    echo "$1 $parts"
    for i in "${!paths[@]}"; do
        if [ "$(device "${paths[i]}")" = "$(device "$1")" ]; then
            echo "$1 $(( tree_bytes[i] > archive_bytes[i] ? tree_bytes[i] - archive_bytes[i] : 0 ))"
        else
            echo "${paths[i]} ${tree_bytes[i]}"
        fi
    done
}

pack() {
    local part_size=2G states=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --part-size) [ $# -ge 2 ] || usage; part_size=$2; shift 2 ;;
            --state) [ $# -ge 2 ] || usage; states+=("$2"); shift 2 ;;
            -*) usage ;;
            *) break ;;
        esac
    done
    local dirs=("$@")
    [[ ${#states[@]} -gt 0 && ${#dirs[@]} -gt 0 ]] || usage
    local part_bytes i state="" candidate lacks report=()
    part_bytes=$(numfmt --from=iec "$part_size") || die "--part-size takes a size such as 2G, not '$part_size'"
    local -a bytes=() largest=()
    for i in "${!dirs[@]}"; do
        [[ "${dirs[i]}" == /?* && -d "${dirs[i]}" ]] || die "${dirs[i]} is not an absolute path of a directory"
        bytes+=("$(du -sb "${dirs[i]}" | cut -f1)")
        largest+=("$(find "${dirs[i]}" -type f -printf '%s\n' | sort -n | tail -1)")
    done
    for candidate in "${states[@]}"; do
        [[ "$candidate" == /* ]] || die "--state takes an absolute path, not '$candidate'"
        if lacks=$(for i in "${!dirs[@]}"; do
                       if [ "$(device "$candidate")" = "$(device "${dirs[i]}")" ]; then
                           echo "$candidate ${largest[i]:-0}"
                       else
                           echo "$candidate ${bytes[i]}"
                       fi
                   done | check_room); then
            state=$candidate
            break
        fi
        report+=("$lacks")
    done
    [ -n "$state" ] || die "no room for the build state: $(IFS=';'; echo "${report[*]}")"
    [ ! -e "$state" ] || [ -z "$(ls -A "$state")" ] || die "$state is not empty"
    mkdir -p "$state"
    echo "part-bytes $part_bytes" > "$state/manifest"
    for i in "${!dirs[@]}"; do
        echo "packing ${dirs[i]} ($(gb "${bytes[i]}")) into $state" >&2
        tar -C "$(dirname "${dirs[i]}")" --remove-files -cf - "$(basename "${dirs[i]}")" | zstd -q -T0 -3 |
            split -a 3 -d -b "$part_bytes" - "$state/$i.tar.zst."
        echo "archive $i $(cat "$state/$i.tar.zst."* | wc -c) ${bytes[i]} ${dirs[i]}" >> "$state/manifest"
    done
    ls -l "$state" >&2
    printf '%s\n' "$state"
}

room() {
    [ $# -ge 2 ] || usage
    read_manifest "$1"
    shift
    local dir lacks report=()
    for dir in "$@"; do
        [[ "$dir" == /* ]] || die "room takes absolute directories, not '$dir'"
        if lacks=$(unpack_needs "$dir" no | check_room); then
            printf '%s\n' "$dir"
            return 0
        fi
        report+=("$lacks")
    done
    die "no room to receive and unpack the build state: $(IFS=';'; echo "${report[*]}")"
}

unpack() {
    [ $# -eq 1 ] || usage
    local state=$1 i part parts lacks
    read_manifest "$state/manifest"
    lacks=$(unpack_needs "$state" yes | check_room) || die "no room to unpack the build state: $lacks"
    shopt -s nullglob
    for i in "${!paths[@]}"; do
        parts=("$state/${indices[i]}.tar.zst."*)
        [ ${#parts[@]} -gt 0 ] || die "no parts $state/${indices[i]}.tar.zst.*"
        [ ! -e "${paths[i]}" ] || die "${paths[i]} exists already"
        echo "unpacking ${paths[i]} ($(gb "${tree_bytes[i]}")) from ${#parts[@]} parts" >&2
        mkdir -p "$(dirname "${paths[i]}")"
        for part in "${parts[@]}"; do
            cat "$part"
            rm -f "$part"
        done | zstd -q -d | tar -C "$(dirname "${paths[i]}")" -xf -
    done
    rm -rf "$state"
}

space() {
    [[ $# -eq 2 && "$2" =~ ^[0-9]+$ ]] || usage
    local avail
    df -h "$(existing "$1")"
    avail=$(available "$1")
    [ "$avail" -ge $(($2 * 1000 * 1000 * 1000)) ] ||
        die "$(existing "$1") has $(gb "$avail") free, this job needs at least $2 GB"
}

[ $# -ge 1 ] || usage
command=$1
shift
case "$command" in
    pack) pack "$@" ;;
    room) room "$@" ;;
    unpack) unpack "$@" ;;
    space) space "$@" ;;
    *) usage ;;
esac
