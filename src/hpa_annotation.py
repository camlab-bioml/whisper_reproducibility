"""HPA localisation of high-confidence preys (Fig. 6d-e, Supplementary Fig. 21)."""

import os

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch


def run(cfg):

    # ---------- Config ----------
    FDR_THRESHOLD = cfg.get("fdr_threshold", 0.01)
    HPA_TSV = cfg["hpa_path"]   # columns: "Gene name", "Main location"
    bait = cfg["hpa_bait"]
    OUT = cfg["outdir"]
    os.makedirs(OUT, exist_ok=True)

    method_colors = {
        'Whisper':      '#003f5c',
        'SAINTq':       '#ffb347',
        'SAINTexpress': '#ff7f0e',
        'limma':        '#d62728',
    }
    method_order = ['Whisper', 'SAINTq', 'SAINTexpress', 'limma']

    # Gradient tuning
    GAMMA = 1.8  
    FLOOR = 0.15  

    # ---------------- LOAD DATA ----------------
    puppi_df = pd.read_csv(cfg["fdr_output"])
    limma_df = pd.read_csv(cfg["limma"], sep="\t")
    saintq_df = pd.read_csv(cfg["saintq"], sep="\t")
    saintexpress_df = pd.read_csv(cfg["saintexpress"], sep="\t")

    if "Protein" in limma_df.columns and "Prey" not in limma_df.columns:
        limma_df = limma_df.rename(columns={"Protein": "Prey"})


    if "PreyGene" not in saintq_df.columns and "Prey" in saintq_df.columns:
        saintq_df = saintq_df.rename(columns={"Prey": "PreyGene"})

    def keep_first_gene(df, col):
        if col in df.columns:
            df[col] = df[col].astype(str).str.split(';').str[0].str.strip()
        return df

    keep_first_gene(limma_df, "Prey")

    keep_first_gene(saintq_df, "PreyGene")
    keep_first_gene(saintexpress_df, "PreyGene")
    keep_first_gene(puppi_df, "Prey")

    def whisper_scores(prey_list):
        sub = puppi_df[puppi_df["Bait"] == bait][["Prey", "predicted_probability"]].dropna()
        best = sub.groupby("Prey")["predicted_probability"].max().to_dict()
        return np.array([float(best.get(p, 0.0)) for p in prey_list])

    def saint_scores(df, prey_list, prey_col="PreyGene"):
        sub = df[df["Bait"] == bait][[prey_col, "AvgP"]].dropna()
        best = sub.groupby(prey_col)["AvgP"].max().to_dict()
        return np.array([float(best.get(p, 0.0)) for p in prey_list])

    def limma_scores(prey_list):
        sub = limma_df[limma_df["Bait"] == bait][["Prey", "adj.P.Val.one_sided"]].dropna()
        best = sub.groupby("Prey")["adj.P.Val.one_sided"].min().to_dict()
        # higher = better: 1 - adj.P.Val.one_sided
        return np.array([float(np.clip(1.0 - best.get(p, 1.0), 0.0, 1.0)) for p in prey_list])

    def whisper_mask(prey_list):
        sig = puppi_df[(puppi_df["Bait"] == bait) & (puppi_df["FDR"] <= FDR_THRESHOLD)]
        sig_preys = set(sig["Prey"].astype(str))
        return np.array([p in sig_preys for p in prey_list], dtype=bool)

    def saint_mask(df, prey_list, prey_col="PreyGene"):
        sig = df[(df["Bait"] == bait) & (df["BFDR"] <= FDR_THRESHOLD)]
        sig_preys = set(sig[prey_col].astype(str))
        return np.array([p in sig_preys for p in prey_list], dtype=bool)

    def limma_mask(prey_list):
        sig = limma_df[(limma_df["Bait"] == bait) & (limma_df["adj.P.Val.one_sided"] <= FDR_THRESHOLD)]
        sig_preys = set(sig["Prey"].astype(str))
        return np.array([p in sig_preys for p in prey_list], dtype=bool)

    def collect_scores_and_masks(prey_list):
        return (
            {
                'Whisper':      whisper_scores(prey_list),
                'SAINTq':       saint_scores(saintq_df, prey_list, "PreyGene"),
                'SAINTexpress': saint_scores(saintexpress_df, prey_list, "PreyGene"),
                'limma':        limma_scores(prey_list),
            },
            {
                'Whisper':      whisper_mask(prey_list),
                'SAINTq':       saint_mask(saintq_df, prey_list, "PreyGene"),
                'SAINTexpress': saint_mask(saintexpress_df, prey_list, "PreyGene"),
                'limma':        limma_mask(prey_list),
            }
        )

    # ---------- Significant preys per method ----------
    def whisper_sig_preys():
        sub = puppi_df[(puppi_df["Bait"] == bait) & (puppi_df["FDR"] <= FDR_THRESHOLD)].copy()
        if sub.empty: return []
        sub = (sub.sort_values(["Prey","predicted_probability"], ascending=[True,False])
                  .groupby("Prey", as_index=False).first()
                  .sort_values("predicted_probability", ascending=False))
        return sub["Prey"].astype(str).tolist()

    def saint_sig_preys(df):
        sub = df[(df["Bait"] == bait) & (df["BFDR"] <= FDR_THRESHOLD)].copy()
        if sub.empty: return []
        sub = (sub.sort_values(["PreyGene","AvgP"], ascending=[True,False])
                  .groupby("PreyGene", as_index=False).first()
                  .sort_values("AvgP", ascending=False))
        return sub["PreyGene"].astype(str).tolist()

    def limma_sig_preys():
        sub = limma_df[(limma_df["Bait"] == bait) & (limma_df["adj.P.Val.one_sided"] <= FDR_THRESHOLD)].copy()
        if sub.empty: return []
        sub = (sub.sort_values(["Prey","adj.P.Val.one_sided"], ascending=[True,True])
                  .groupby("Prey", as_index=False).first())
        sub["score"] = 1.0 - sub["adj.P.Val.one_sided"]
        sub = sub.sort_values("score", ascending=False)
        return sub["Prey"].astype(str).tolist()

    sig_sets = {
        "Whisper":      whisper_sig_preys(),
        "SAINTq":       saint_sig_preys(saintq_df),
        "SAINTexpress": saint_sig_preys(saintexpress_df),
        "limma":        limma_sig_preys(),
    }

    # ---------- HPA parsing (nuclear sublocs separate, ER separate, others -> Other) ----------
    def parse_hpa_terms(s: str):
        if not isinstance(s, str) or not s.strip():
            return []
        return [t.strip() for t in s.split(";") if t.strip()]

    hpa_df_raw = pd.read_csv(HPA_TSV, sep="\t", usecols=["Gene name", "Main location"])
    hpa_df_raw["Gene name"] = hpa_df_raw["Gene name"].astype(str).str.upper()
    hpa_df_raw["Main location"] = hpa_df_raw["Main location"].astype(str)
    hpa_terms_map = {g: parse_hpa_terms(loc) for g, loc in
                     zip(hpa_df_raw["Gene name"], hpa_df_raw["Main location"])}

    NUCLEAR_SUBLOCS = [
        'Kinetochore','Mitotic chromosome','Nuclear bodies','Nuclear membrane',
        'Nuclear speckles','Nucleoli','Nucleoli fibrillar center','Nucleoli rim','Nucleoplasm'
    ]
    HPA_RING_ORDER = NUCLEAR_SUBLOCS + ['ER', 'Other']
    HPA_RING_COLORS = {
        'Kinetochore':               '#c0368f',
        'Mitotic chromosome':        '#cf3a9b',
        'Nuclear bodies':            '#b03cb5',
        'Nuclear membrane':          '#8f46c3',
        'Nuclear speckles':          '#764fcd',
        'Nucleoli':                  '#5e58d2',
        'Nucleoli fibrillar center': '#4f60d5',
        'Nucleoli rim':              '#3d5fbc',
        'Nucleoplasm':               '#3a5aa8',
        'ER':                        '#E67E22',
        'Other':                     '#9E9E9E',
    }
    def hpa_ring_label(terms):
        if not terms:
            return 'Other'
        tset = set(terms)
        for nuc in NUCLEAR_SUBLOCS:
            if nuc in tset:
                return nuc
        if 'Endoplasmic reticulum' in tset:
            return 'ER'
        return 'Other'

    # ---------- colormap builders ----------
    def make_light_to_color(base_hex: str):
        base = np.array(to_rgb(base_hex))
        light = np.array([1.0, 1.0, 1.0])
        return LinearSegmentedColormap.from_list("light_to_base", [light, base], N=256)
    cmaps = {m: make_light_to_color(method_colors[m]) for m in method_order}

    def map_score_to_color(score, cmap):
        """Absolute 0-1 mapping with floor + gamma (used for ALL methods)."""
        s = float(np.clip(score, 0.0, 1.0))
        s_gamma = FLOOR + (1.0 - FLOOR) * (s ** GAMMA)
        return cmap(s_gamma)

    # ---------- Plot maker ----------
    def make_plot_for_method(main_method: str):
        preys = sig_sets[main_method]
        if len(preys) == 0:
            print(f"[{main_method}] No significant preys at threshold {FDR_THRESHOLD}. Skipping.")
            return

        scores_by_method, sigmask_by_method = collect_scores_and_masks(preys)

        # HPA labels/colors
        hpa_labels = []
        hpa_colors = []
        for p in preys:
            terms = hpa_terms_map.get(p.upper(), [])
            lbl = hpa_ring_label(terms)
            hpa_labels.append(lbl)
            hpa_colors.append(HPA_RING_COLORS[lbl])

        label_rank = {lab: i for i, lab in enumerate(HPA_RING_ORDER)}
        N = len(preys)
        idx = np.arange(N)
        order = sorted(idx, key=lambda i: (label_rank.get(hpa_labels[i], 10**6), i))
        preys       = [preys[i] for i in order]
        hpa_labels  = [hpa_labels[i] for i in order]
        hpa_colors  = [hpa_colors[i] for i in order]
        for m in method_order:
            scores_by_method[m]  = scores_by_method[m][order]
            sigmask_by_method[m] = sigmask_by_method[m][order]
        # --------------------------------------------------------------------------

        # polar layout
        theta  = np.linspace(0, 2*np.pi, N, endpoint=False)
        dtheta = (2*np.pi) / N

        ring_width = 0.18
        gap = 0.03
        r0 = 1.0

        ring_radii = {
            method_order[0]: r0,
            method_order[1]: r0 + (ring_width + gap)*1,
            method_order[2]: r0 + (ring_width + gap)*2,
            method_order[3]: r0 + (ring_width + gap)*3,
        }
        # thin halo height for significance marks
        halo_h = 0.028

        # HPA ring just outside method rings
        r_hpa = r0 + (ring_width + gap)*4 + 0.02

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 12))
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_axis_off()

        # --- method rings (ALL use gradient mapping now) ---
        for method in method_order:
            rbase = ring_radii[method]
            scores = scores_by_method[method]
            cmap = cmaps[method]

            for i in range(N):
                col = map_score_to_color(scores[i], cmap)
                ax.bar(theta[i], height=ring_width, width=dtheta, bottom=rbase,
                       color=col, edgecolor='none', align='edge')

            # significance halo (thin outer band on top of each method ring)
            sigmask = sigmask_by_method[method]
            for i, is_sig in enumerate(sigmask):
                if is_sig:
                    ax.bar(theta[i], height=halo_h, width=dtheta, bottom=rbase + ring_width - halo_h,
                           color='black', edgecolor='none', align='edge', alpha=0.9)

        # --- single HPA ring ---
        for i in range(N):
            ax.bar(theta[i], height=ring_width*0.75, width=dtheta, bottom=r_hpa,
                   color=hpa_colors[i], edgecolor='none', align='edge')

        # --- protein labels with side-aware alignment ---
        for i, prey_name in enumerate(preys):
            angle = theta[i] + dtheta/2
            rlab  = r_hpa + ring_width*0.95
            deg = (np.degrees(angle) + 360) % 360

            if deg <= 180:
                rotation = np.rad2deg(-angle + np.pi/2)
                ha = 'left'
            else:
                rotation = np.rad2deg(-angle + np.pi/2) + 180
                ha = 'right'
            ax.text(angle, rlab, prey_name, fontsize=9, color='black',
                    rotation=rotation, rotation_mode='anchor', ha=ha, va='center')

        # Legends
        method_patches = [Patch(facecolor=method_colors[m], edgecolor='none', label=m) for m in method_order]
        leg1 = ax.legend(handles=method_patches, title="Methods (darker = higher; floor+gamma mapping)",
                         loc='center left', bbox_to_anchor=(1.03, 0.10), frameon=False, fontsize=10, title_fontsize=10)
        ax.add_artist(leg1)

        halo_patch = Patch(facecolor='black', edgecolor='none', label='Significant (halo)')
        ax.legend(handles=[halo_patch], loc='center left', bbox_to_anchor=(1.03, 0.28),
                  frameon=False, fontsize=10)

        present_labels = []
        seen = set()
        for lbl in hpa_labels:
            if lbl not in seen:
                seen.add(lbl)
                present_labels.append(lbl)
        hpa_handles = [Patch(facecolor=HPA_RING_COLORS[k], edgecolor='none', label=k) for k in present_labels]
        # -----------------------------------------------------------------------------------

        ax.legend(handles=hpa_handles, title="HPA ring",
                  loc='center left', bbox_to_anchor=(1.03, 0.52), frameon=False, fontsize=10, title_fontsize=10)

        ax.set_title(
            f"{bait} — Significant set: {main_method} (n={N})\n"
            f"All rings use score gradients with floor+gamma; thin black halo = significant by that method; outer ring = HPA (nuclear sublocs, ER, Other)",
            fontsize=12, pad=20
        )

        out_path = f"{OUT}/{bait}_rings_{main_method}_sig_HPA_allGradients_halos.pdf"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

    # ---------- Generate one plot per method ----------
    for m in method_order:
        make_plot_for_method(m)


    cats = HPA_RING_ORDER  # nuclear sublocs in your order + ['ER','Other']
    cat_colors = [HPA_RING_COLORS[c] for c in cats]

    # Count and percent per method
    counts_by_method = {}
    perc_by_method = {}
    for m in method_order:
        preys = sig_sets[m]
        counts = {c: 0 for c in cats}
        for p in preys:
            terms = hpa_terms_map.get(str(p).upper(), [])
            lbl = hpa_ring_label(terms)
            if lbl not in counts:
                lbl = 'Other'
            counts[lbl] += 1
        total = max(1, sum(counts.values()))
        counts_by_method[m] = counts
        perc_by_method[m] = {c: (counts[c] / total) * 100.0 for c in cats}

    # Remove categories with zero total across all methods
    valid_cats = [c for c in cats if sum(counts_by_method[m][c] for m in method_order) > 0]
    valid_colors = [HPA_RING_COLORS[c] for c in valid_cats]

    # Build stacked bars (%)
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(method_order), dtype=float)

    for c, col in zip(cats, cat_colors):
        vals = np.array([perc_by_method[m][c] for m in method_order], dtype=float)
        bars = plt.bar(method_order, vals, bottom=bottom, color=col, edgecolor='white', linewidth=0.6, label=c)
        bottom += vals


    plt.ylabel('% of significant hits', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.title(f'{bait}: HPA category composition of significant hits', fontsize=12)
    plt.legend(title='HPA category', bbox_to_anchor=(1.02, 1.0), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{bait}_HPA_category_composition_stacked_percentage.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved stacked bar plot of HPA categories per method.")


    # Build stacked bars (%)
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(method_order), dtype=float)

    for c, col in zip(cats, cat_colors):
        vals = np.array([counts_by_method[m][c] for m in method_order], dtype=float)
        bars = plt.bar(method_order, vals, bottom=bottom, color=col, edgecolor='white', linewidth=0.6, label=c)
        bottom += vals

    plt.ylabel('Number of significant hits', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.title(f'{bait}: HPA category composition of significant hits', fontsize=12)
    plt.legend(title='HPA category', bbox_to_anchor=(1.02, 1.0), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{bait}_HPA_category_composition_stacked_numbers.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved stacked bar plot of HPA categories per method.")
