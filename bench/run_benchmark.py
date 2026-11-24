"""
ghost.bench.run_benchmark
----------------------------------------------------------------
Le protocole complet (voir SPEC.md) : pour plusieurs graines
aleatoires, entraine et mesure chaque reference (baseline) sur les
4 variantes, puis agrege les resultats.

Usage :
    python3 run_benchmark.py                 # 5 graines, config par defaut
    python3 run_benchmark.py --seeds 10      # plus de graines (plus fiable, plus long)
    python3 run_benchmark.py --episodes 200  # entrainements plus courts (plus rapide)

Duree indicative : environ 15-30s par (graine x variante x reference)
sur un telephone -- avec la config par defaut (5 graines x 4 variantes
x 2 references directes + 3 fine-tunings + l'oubli), compter plusieurs
minutes. Reduis --seeds ou --episodes pour un premier apercu rapide.
"""

import argparse
import statistics
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from envs import make_variant_a, make_variant_b, make_variant_c, make_variant_d, STATE_SIZE, N_ACTIONS
from dqn_runner import train_dqn, evaluate_success_rate

VARIANTS = {"A": make_variant_a, "B": make_variant_b, "C": make_variant_c, "D": make_variant_d}


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def run_protocol(seeds, n_episodes):
    results = {
        "scratch": {v: [] for v in VARIANTS},
        "finetuned": {v: [] for v in VARIANTS if v != "A"},
        "prioritized": {v: [] for v in VARIANTS},
        "forgetting_A_after_D": [],
    }

    for seed in seeds:
        print(f"\n=== graine {seed} ===")

        # -- references directes (depuis zero) sur les 4 variantes --
        models_scratch = {}
        for name, factory in VARIANTS.items():
            report = train_dqn(factory, STATE_SIZE, N_ACTIONS, n_episodes=n_episodes, seed=seed)
            results["scratch"][name].append(report["episodes_to_threshold"])
            models_scratch[name] = report["final_model"]
            print(f"  [scratch]     {name}: episodes_to_threshold={report['episodes_to_threshold']}")

        # -- fine-tuning : part du modele entraine sur A, applique a B/C/D --
        model_a = models_scratch["A"]
        for name in ["B", "C", "D"]:
            report = train_dqn(VARIANTS[name], STATE_SIZE, N_ACTIONS, n_episodes=n_episodes,
                                seed=seed, init_model=model_a)
            results["finetuned"][name].append(report["episodes_to_threshold"])
            print(f"  [fine-tune A->{name}] episodes_to_threshold={report['episodes_to_threshold']}")

            if name == "D":
                # -- oubli : le modele affine sur D se souvient-il encore de A ? --
                acc_a_after = evaluate_success_rate(report["final_model"], make_variant_a, n_episodes=30, seed=seed + 1000)
                results["forgetting_A_after_D"].append(acc_a_after)
                print(f"  [oubli]       succes sur A apres avoir appris D: {acc_a_after:.2f}")

        # -- replay prioritaire (depuis zero) sur les 4 variantes --
        for name, factory in VARIANTS.items():
            report = train_dqn(factory, STATE_SIZE, N_ACTIONS, n_episodes=n_episodes,
                                seed=seed, use_prioritized=True)
            results["prioritized"][name].append(report["episodes_to_threshold"])
            print(f"  [prioritized] {name}: episodes_to_threshold={report['episodes_to_threshold']}")

    return results


def print_summary(results):
    print("\n" + "=" * 60)
    print("RESUME (episodes pour atteindre 90% de reussite -- plus bas = mieux)")
    print("=" * 60)

    print(f"\n{'Variante':<10}{'Scratch':<18}{'Fine-tune(A->X)':<20}{'Prioritized':<15}")
    for name in VARIANTS:
        m_s, sd_s = mean_std(results["scratch"][name])
        s_s = f"{m_s:.0f} (+/-{sd_s:.0f})" if m_s else "jamais atteint"
        if name != "A":
            m_f, sd_f = mean_std(results["finetuned"][name])
            s_f = f"{m_f:.0f} (+/-{sd_f:.0f})" if m_f else "jamais atteint"
        else:
            s_f = "--"
        m_p, sd_p = mean_std(results["prioritized"][name])
        s_p = f"{m_p:.0f} (+/-{sd_p:.0f})" if m_p else "jamais atteint"
        print(f"{name:<10}{s_s:<18}{s_f:<20}{s_p:<15}")

    m_forget, sd_forget = mean_std(results["forgetting_A_after_D"])
    print(f"\nOubli catastrophique : succes sur A apres avoir appris D = {m_forget:.2f} (+/-{sd_forget:.2f})")
    print("(1.00 = aucun oubli, 0.00 = A completement oublie)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="nombre de graines aleatoires")
    parser.add_argument("--episodes", type=int, default=300, help="episodes d'entrainement par run")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    results = run_protocol(seeds, args.episodes)
    print_summary(results)
