#!/bin/bash

num_procs=4

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

"$@"
