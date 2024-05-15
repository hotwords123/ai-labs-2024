#!/bin/bash

num_procs=$(( $(nproc) / 2 ))

function run {
    campaign=$1
    command=$2
    args=("${@:3}")

    out_dir=results/"$campaign"
    mkdir -p "$out_dir"

    echo "$command" >> "$out_dir".txt
    printf "%s\n" "${args[@]}" >> "$out_dir".txt
    echo "" >> "$out_dir".txt

    echo $command
    xargs -P $num_procs -I {} bash -c "echo '{}'; $command > '$out_dir'/'{}'.txt" <<< $(printf "%s\n" "${args[@]}")
}

function battle {
    players=("uct uct" "uct random" "random uct" "uct alphabeta" "alphabeta uct")
    command="python -m pit --seed 42 --players {} --C 1.0 --n_rollout 7 --n_search 64 --deterministic --n_match 1000"
    run "battle" "$command" "${players[@]}"
}

function search_c {
    C=(0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.75 1.0 1.5 2.5 5.0)
    command="python -m pit --seed 42 --players alphabeta uct --C {} --n_rollout 7 --n_search 64 --deterministic --n_match 1000"
    run "search_c" "$command" "${C[@]}"
}

function gobang {
    # [$1, $1 + $2)
    seeds=($(seq $1 $(( $1 + $2 - 1 ))))
    command="python -m pit --seed {} --game gobang --args 5 4 --deterministic ${@:3}"
    run "gobang" "$command" "${seeds[@]}"
}

function uct_config {
    suffix=$([[ $1 -eq 1 ]] && echo "" || echo $1)
    printf -- "--%s$suffix %s " "${@:2}"
}

function gobang_sym {
    gobang $1 $2 --players uct uct $(uct_config 1 $3)
}

function gobang_asym {
    gobang $1 $2 --players uct uct2 $(uct_config 1 $3) $(uct_config 2 $4)
}

function gobang_battle {
    group_a="C 1.0 n_rollout 12 n_search 500"
    group_b="C 1.0 n_rollout 12 n_search 200"
    group_c="C 1.0 n_rollout 12 n_search 900"
    group_d="C 1.0 n_rollout 7 n_search 500"
    group_e="C 1.0 n_rollout 20 n_search 500"
    n_match=20

    gobang_sym 0 $n_match "$group_a"
    gobang_asym 100 $n_match "$group_a" "$group_b"
    gobang_asym 200 $n_match "$group_b" "$group_a"
    gobang_asym 300 $n_match "$group_a" "$group_c"
    gobang_asym 400 $n_match "$group_c" "$group_a"
    gobang_asym 500 $n_match "$group_a" "$group_d"
    gobang_asym 600 $n_match "$group_d" "$group_a"
    gobang_asym 700 $n_match "$group_a" "$group_e"
    gobang_asym 800 $n_match "$group_e" "$group_a"
}

"$@"
