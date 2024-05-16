#!/bin/bash

num_procs=${num_procs:-$(( $(nproc) / 2 ))}
echo Using $num_procs processes

function run {
    campaign=$1
    command=$2
    args=("${@:3}")

    out_dir=results/"$campaign"
    mkdir -p "$out_dir"

    echo "$command" >> "$out_dir".txt
    printf "%s; " "${args[@]}" >> "$out_dir".txt
    printf "\n\n" >> "$out_dir".txt

    echo $command
    xargs -P $num_procs -I {} bash -c "echo + '{}'; $command > '$out_dir'/'{}'.txt; echo - '{}'" <<< $(printf "%s\n" "${args[@]}")
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
    groups=(
        "C 1.0 n_rollout 12 n_search 500"   # A
        "C 1.0 n_rollout 12 n_search 200"   # B
        "C 1.0 n_rollout 12 n_search 1200"  # C
        "C 1.0 n_rollout 7 n_search 500"    # D
        "C 1.0 n_rollout 5 n_search 500"    # E
        "C 1.0 n_rollout 25 n_search 500"   # F
    )
    n_match=$1
    seed=0
    seed_step=100

    for i in $(seq 0 $(( ${#groups[@]} - 1 ))); do
        if [[ i -eq 0 ]]; then
            gobang_sym $seed $n_match "${groups[i]}"
            seed=$(( $seed + $seed_step ))
        else
            gobang_asym $seed $n_match "${groups[0]}" "${groups[i]}"
            seed=$(( $seed + $seed_step ))
            gobang_asym $seed $n_match "${groups[i]}" "${groups[0]}"
            seed=$(( $seed + $seed_step ))
        fi
    done
}

"$@"
