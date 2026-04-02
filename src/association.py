"""
Association & Sequence Mining (Chapter XI).

Implements:
  1. Apriori — mine co-occurrence rules from episode baskets
  2. Sequential pattern mining — temporal escalation within episodes
  3. Association rule visualization (heatmap, network graph)
  4. Rule quality metrics: support, confidence, lift, conviction, leverage
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations

from src.config import (
    OUT_FIG, OUT_RESULT,
    APRIORI_MIN_SUPPORT, APRIORI_MIN_CONFIDENCE, APRIORI_MIN_LIFT,
)
from src.utils import logger, save_figure, save_results


# ═══════════════════════════════════════════════════════════════════════════
# 1. Episode basket construction
# ═══════════════════════════════════════════════════════════════════════════

def build_episode_baskets(df):
    """
    Build transaction baskets from episodes.
    Each episode is a basket of unique event types within that episode.
    """
    logger.info("Building episode baskets...")

    episodes = df[df["EPISODE_ID"].notna()].copy()
    episodes["EPISODE_ID"] = episodes["EPISODE_ID"].astype(int)

    baskets = (episodes.groupby("EPISODE_ID")["EVENT_TYPE"]
               .apply(lambda x: list(set(x)))
               .reset_index())
    baskets.columns = ["EPISODE_ID", "ITEMS"]
    baskets["BASKET_SIZE"] = baskets["ITEMS"].apply(len)

    # Filter to multi-event episodes
    multi = baskets[baskets["BASKET_SIZE"] >= 2]

    logger.info(f"Total episodes: {len(baskets):,}")
    logger.info(f"Multi-event episodes: {len(multi):,} ({len(multi)/len(baskets)*100:.1f}%)")
    logger.info(f"Basket size distribution:\n{baskets['BASKET_SIZE'].describe().to_string()}")

    # Basket size distribution plot
    fig, ax = plt.subplots(figsize=(10, 5))
    size_counts = baskets["BASKET_SIZE"].value_counts().sort_index()
    ax.bar(size_counts.index[:15], size_counts.values[:15], color="#4CAF50", alpha=0.8)
    ax.set_xlabel("Number of Distinct Event Types per Episode")
    ax.set_ylabel("Number of Episodes")
    ax.set_title("Episode Basket Size Distribution", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "25_basket_size_distribution", OUT_FIG)
    plt.close(fig)

    return multi


# ═══════════════════════════════════════════════════════════════════════════
# 2. Apriori implementation
# ═══════════════════════════════════════════════════════════════════════════

def run_apriori(baskets_df):
    """
    Run Apriori algorithm using mlxtend.
    Falls back to manual implementation if mlxtend unavailable.
    """
    logger.info("─" * 40)
    logger.info("Apriori Association Rule Mining")
    logger.info("─" * 40)

    baskets = baskets_df["ITEMS"].tolist()
    n_transactions = len(baskets)

    try:
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import apriori, association_rules

        te = TransactionEncoder()
        te_array = te.fit_transform(baskets)
        basket_df = pd.DataFrame(te_array, columns=te.columns_)

        # Frequent itemsets
        freq = apriori(basket_df, min_support=APRIORI_MIN_SUPPORT, use_colnames=True)
        freq["length"] = freq["itemsets"].apply(len)
        logger.info(f"Frequent itemsets (support ≥ {APRIORI_MIN_SUPPORT}): {len(freq)}")

        if len(freq) == 0:
            logger.warning("No frequent itemsets found. Try lowering min_support.")
            return None, None

        # Association rules
        rules = association_rules(freq, metric="confidence",
                                 min_threshold=APRIORI_MIN_CONFIDENCE)
        rules = rules[rules["lift"] >= APRIORI_MIN_LIFT]
        rules = rules.sort_values("lift", ascending=False)

        # Format for readability
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

        logger.info(f"Association rules (conf ≥ {APRIORI_MIN_CONFIDENCE}, "
                    f"lift ≥ {APRIORI_MIN_LIFT}): {len(rules)}")

        # Save
        rules_save = rules[["antecedents_str", "consequents_str",
                           "support", "confidence", "lift",
                           "leverage", "conviction"]].copy()
        rules_save.columns = ["Antecedent", "Consequent", "Support", "Confidence",
                              "Lift", "Leverage", "Conviction"]
        save_results(rules_save, "association_rules", OUT_RESULT)

        # Top rules
        logger.info("\nTop 15 Association Rules (by lift):")
        for _, row in rules_save.head(15).iterrows():
            logger.info(f"  {{{row['Antecedent']}}} → {{{row['Consequent']}}}  "
                       f"(supp={row['Support']:.3f}, conf={row['Confidence']:.2f}, "
                       f"lift={row['Lift']:.2f})")

        return freq, rules

    except ImportError:
        logger.info("mlxtend not available, using manual Apriori implementation")
        return _manual_apriori(baskets, n_transactions)


def _manual_apriori(baskets, n_transactions):
    """Fallback manual Apriori implementation."""
    min_count = int(APRIORI_MIN_SUPPORT * n_transactions)

    # Count single items
    item_counts = Counter()
    for basket in baskets:
        for item in basket:
            item_counts[item] += 1

    freq_1 = {frozenset([item]): count
              for item, count in item_counts.items()
              if count >= min_count}
    logger.info(f"Frequent 1-itemsets: {len(freq_1)}")

    all_frequent = dict(freq_1)
    current = freq_1

    # Generate higher-order itemsets
    k = 2
    while current and k <= 4:
        candidates = set()
        items = list(current.keys())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                union = items[i] | items[j]
                if len(union) == k:
                    candidates.add(union)

        # Count candidates
        next_freq = {}
        for basket in baskets:
            basket_set = frozenset(basket)
            for cand in candidates:
                if cand.issubset(basket_set):
                    next_freq[cand] = next_freq.get(cand, 0) + 1

        current = {k: v for k, v in next_freq.items() if v >= min_count}
        all_frequent.update(current)
        logger.info(f"Frequent {k}-itemsets: {len(current)}")
        k += 1

    # Generate rules from frequent itemsets
    rules = []
    for itemset, count in all_frequent.items():
        if len(itemset) < 2:
            continue
        support = count / n_transactions
        for item in itemset:
            antecedent = itemset - frozenset([item])
            consequent = frozenset([item])
            ant_count = all_frequent.get(antecedent, 0)
            if ant_count > 0:
                confidence = count / ant_count
                cons_support = all_frequent.get(consequent, 0) / n_transactions
                lift = confidence / cons_support if cons_support > 0 else 0
                if confidence >= APRIORI_MIN_CONFIDENCE and lift >= APRIORI_MIN_LIFT:
                    rules.append({
                        "Antecedent": ", ".join(sorted(antecedent)),
                        "Consequent": ", ".join(sorted(consequent)),
                        "Support": support,
                        "Confidence": confidence,
                        "Lift": lift,
                    })

    rules_df = pd.DataFrame(rules).sort_values("Lift", ascending=False)
    save_results(rules_df, "association_rules", OUT_RESULT)
    logger.info(f"Generated {len(rules_df)} association rules")
    return all_frequent, rules_df


# ═══════════════════════════════════════════════════════════════════════════
# 3. Sequential pattern mining
# ═══════════════════════════════════════════════════════════════════════════

def mine_sequential_patterns(df):
    """
    Mine temporal escalation patterns within episodes.
    Finds common sequences like Hail → Thunderstorm Wind → Tornado.
    """
    logger.info("─" * 40)
    logger.info("Sequential Pattern Mining")
    logger.info("─" * 40)

    episodes = df[df["EPISODE_ID"].notna()].copy()
    episodes["EPISODE_ID"] = episodes["EPISODE_ID"].astype(int)

    # Order events within each episode by time
    episodes = episodes.sort_values(["EPISODE_ID", "BEGIN_DATE_TIME"])

    # Build sequences
    sequences = (episodes.groupby("EPISODE_ID")["EVENT_TYPE"]
                 .apply(list)
                 .reset_index())
    sequences.columns = ["EPISODE_ID", "SEQUENCE"]
    sequences["SEQ_LENGTH"] = sequences["SEQUENCE"].apply(len)
    sequences["N_DISTINCT"] = sequences["SEQUENCE"].apply(lambda x: len(set(x)))
    multi_seq = sequences[sequences["SEQ_LENGTH"] >= 2]

    logger.info(f"Multi-event sequences: {len(multi_seq):,}")

    # ── Sequence length statistics (addresses sparsity concern) ────────────
    logger.info("Sequence length statistics:")
    logger.info(f"  Mean length: {multi_seq['SEQ_LENGTH'].mean():.1f}")
    logger.info(f"  Median length: {multi_seq['SEQ_LENGTH'].median():.0f}")
    logger.info(f"  Max length: {multi_seq['SEQ_LENGTH'].max()}")
    len_dist = multi_seq["SEQ_LENGTH"].value_counts().sort_index()
    for length in sorted(len_dist.index[:8]):
        logger.info(f"  Length {length}: {len_dist[length]:,} episodes "
                    f"({len_dist[length]/len(multi_seq)*100:.1f}%)")
    pct_short = (multi_seq["SEQ_LENGTH"] <= 3).mean() * 100
    logger.info(f"  Episodes with ≤3 events: {pct_short:.1f}% "
                f"(short sequences limit trigram pattern discovery)")

    # Distinct-type distribution
    logger.info("Distinct event types per sequence:")
    for nd in sorted(multi_seq["N_DISTINCT"].value_counts().sort_index().index[:5]):
        cnt = (multi_seq["N_DISTINCT"] == nd).sum()
        logger.info(f"  {nd} distinct types: {cnt:,} ({cnt/len(multi_seq)*100:.1f}%)")

    # Save sequence length distribution
    seq_stats = pd.DataFrame({
        "Metric": ["total_multi_event_episodes", "mean_length", "median_length",
                    "max_length", "pct_length_le_3", "pct_single_type_repeated"],
        "Value": [len(multi_seq), multi_seq["SEQ_LENGTH"].mean(),
                  multi_seq["SEQ_LENGTH"].median(), multi_seq["SEQ_LENGTH"].max(),
                  pct_short, (multi_seq["N_DISTINCT"] == 1).mean() * 100]
    })
    save_results(seq_stats, "sequential_length_stats", OUT_RESULT)

    # ── Bigram (pairwise transition) analysis ──────────────────────────────
    bigram_counts = Counter()
    for seq in multi_seq["SEQUENCE"]:
        for i in range(len(seq) - 1):
            bigram = (seq[i], seq[i + 1])
            if bigram[0] != bigram[1]:  # Skip self-transitions
                bigram_counts[bigram] += 1

    n_total_bigrams = sum(bigram_counts.values())
    bigram_df = pd.DataFrame([
        {"From": bg[0], "To": bg[1], "Count": count,
         "Support": count / len(multi_seq)}
        for bg, count in bigram_counts.most_common(50)
    ])
    save_results(bigram_df, "sequential_bigrams", OUT_RESULT)

    logger.info("Top 15 event transitions (bigrams):")
    for _, row in bigram_df.head(15).iterrows():
        logger.info(f"  {row['From']} → {row['To']}: {row['Count']} "
                    f"(support={row['Support']:.3f})")

    # ── Trigram analysis (strict: no consecutive duplicates) ──────────────
    trigram_counts = Counter()
    trigram_counts_loose = Counter()
    for seq in multi_seq["SEQUENCE"]:
        if len(seq) >= 3:
            for i in range(len(seq) - 2):
                trigram = (seq[i], seq[i + 1], seq[i + 2])
                # Loose: at least 2 distinct types (for comparison)
                if len(set(trigram)) >= 2:
                    trigram_counts_loose[trigram] += 1
                # Strict: no consecutive duplicates (true escalation)
                if trigram[0] != trigram[1] and trigram[1] != trigram[2]:
                    trigram_counts[trigram] += 1

    # Report on sparsity
    n_strict = len(trigram_counts)
    n_loose = len(trigram_counts_loose)
    logger.info(f"\nTrigram patterns found:")
    logger.info(f"  Loose filter (≥2 distinct types): {n_loose} unique trigrams")
    logger.info(f"  Strict filter (no consecutive repeats): {n_strict} unique trigrams")
    logger.info(f"  Filtered out {n_loose - n_strict} trigrams with consecutive "
                f"same-type events ({(n_loose-n_strict)/max(n_loose,1)*100:.1f}%)")

    trigram_df = pd.DataFrame([
        {"Step1": tg[0], "Step2": tg[1], "Step3": tg[2], "Count": count,
         "Support": count / len(multi_seq)}
        for tg, count in trigram_counts.most_common(30)
    ])
    save_results(trigram_df, "sequential_trigrams", OUT_RESULT)

    if len(trigram_df) == 0:
        logger.info("No strict trigram patterns found — sequences are too short "
                    "for meaningful 3-step escalation mining.")
    else:
        logger.info(f"\nTop 10 three-step escalation patterns (strict, no consecutive repeats):")
        for _, row in trigram_df.head(10).iterrows():
            logger.info(f"  {row['Step1']} → {row['Step2']} → {row['Step3']}: "
                        f"{row['Count']} (support={row['Support']:.4f})")

    # ── Transition probability matrix ──────────────────────────────────────
    transition_probs = defaultdict(lambda: defaultdict(float))
    from_counts = Counter()
    for seq in multi_seq["SEQUENCE"]:
        for i in range(len(seq) - 1):
            from_counts[seq[i]] += 1
            transition_probs[seq[i]][seq[i + 1]] += 1

    # Normalize
    for from_type in transition_probs:
        total = from_counts[from_type]
        if total > 0:
            for to_type in transition_probs[from_type]:
                transition_probs[from_type][to_type] /= total

    # Build matrix for top event types
    top_types = [t for t, _ in Counter(
        [e for seq in multi_seq["SEQUENCE"] for e in seq]
    ).most_common(12)]

    trans_matrix = pd.DataFrame(
        [[transition_probs[f].get(t, 0) for t in top_types] for f in top_types],
        index=top_types, columns=top_types,
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(trans_matrix, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, vmin=0, vmax=0.5)
    ax.set_title("Event Type Transition Probability Matrix\n(within storm episodes)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Next Event Type")
    ax.set_ylabel("Current Event Type")
    fig.tight_layout()
    save_figure(fig, "26_transition_matrix", OUT_FIG)
    plt.close(fig)

    return bigram_df, trigram_df, trans_matrix


# ═══════════════════════════════════════════════════════════════════════════
# 4. Visualization
# ═══════════════════════════════════════════════════════════════════════════

def visualize_rules(rules):
    """Visualize association rules: scatter plot and heatmap."""
    if rules is None or len(rules) == 0:
        return

    # Handle both mlxtend DataFrame and manual DataFrame
    if "antecedents_str" in rules.columns:
        rule_data = rules
        conf_col = "confidence"
        lift_col = "lift"
        supp_col = "support"
    else:
        rule_data = rules
        conf_col = "Confidence"
        lift_col = "Lift"
        supp_col = "Support"

    # Support vs. Confidence scatter (sized by lift)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        rule_data[supp_col], rule_data[conf_col],
        c=rule_data[lift_col], cmap="RdYlGn",
        s=rule_data[lift_col] * 50, alpha=0.6, edgecolors="gray",
    )
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Association Rules: Support vs. Confidence (colored by Lift)",
                 fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="Lift")
    fig.tight_layout()
    save_figure(fig, "27_rules_scatter", OUT_FIG)
    plt.close(fig)

    # Co-occurrence heatmap
    cooccurrence = Counter()
    if "antecedents" in rules.columns:
        for _, row in rules.iterrows():
            for a in row["antecedents"]:
                for c in row["consequents"]:
                    cooccurrence[(a, c)] = max(cooccurrence[(a, c)], row["lift"])
    else:
        for _, row in rule_data.iterrows():
            ant = row.get("Antecedent", row.get("antecedents_str", ""))
            con = row.get("Consequent", row.get("consequents_str", ""))
            lift = row.get("Lift", row.get("lift", 0))
            cooccurrence[(ant, con)] = max(cooccurrence.get((ant, con), 0), lift)

    if cooccurrence:
        all_items = sorted(set([k[0] for k in cooccurrence] + [k[1] for k in cooccurrence]))
        if len(all_items) <= 20:
            matrix = pd.DataFrame(0.0, index=all_items, columns=all_items)
            for (a, c), lift in cooccurrence.items():
                if a in matrix.index and c in matrix.columns:
                    matrix.loc[a, c] = lift

            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                        linewidths=0.5)
            ax.set_title("Association Rule Lift Heatmap", fontweight="bold")
            fig.tight_layout()
            save_figure(fig, "28_rules_heatmap", OUT_FIG)
            plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_association_mining(df: pd.DataFrame):
    """Run the full association and sequence mining pipeline."""
    logger.info("=" * 60)
    logger.info("ASSOCIATION & SEQUENCE MINING")
    logger.info("=" * 60)

    baskets = build_episode_baskets(df)
    freq, rules = run_apriori(baskets)
    bigrams, trigrams, trans_matrix = mine_sequential_patterns(df)
    visualize_rules(rules)

    logger.info("Association mining complete.")
    return {
        "frequent_itemsets": freq,
        "rules": rules,
        "bigrams": bigrams,
        "trigrams": trigrams,
        "transition_matrix": trans_matrix,
    }


if __name__ == "__main__":
    from src.config import DATA_PROC
    df = pd.read_parquet(DATA_PROC / "storms_processed.parquet")
    run_association_mining(df)
