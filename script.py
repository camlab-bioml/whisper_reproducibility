"""Command-line entry point for every step of the workflow.

    python script.py --config config/<dataset>.yaml --step <step>
    python script.py --config config/cross_dataset.yaml --step <cross step>

Dataset steps
-------------
    feature_engineering   intensity matrix -> features.csv
    train_and_fdr         features -> FDR_BaitSpecific_Hierarchical.csv and the
                          feature importances (Sup. 20); peptide and fragment
                          level: also the protein aggregate
    benchmark             AUC bars and PR curves, every method (Fig. 2-3, Sup. 9-10)
    overlap               overlap with the references by depth (Fig. 4, Sup. 11)
    fdr_bins              true positives and precision by FDR bin (Fig. 5, Sup. 14c)
    label_quality         weak labels against the references (Sup. 2)
    auc_range             AUC at depths 300-50 and paired differences (Sup. 3-4)
    per_bait              benchmark bait by bait (Sup. 5)
    saint_intersection    benchmark on the pairs both SAINT tools reported (Sup. 6)
    go_aspects            benchmark against GO:BP, GO:MF and all of GO (Sup. 7)
    gocc_sensitivity      GO:CC term-size cap, IEA and GO release (Sup. 8)
    fdr_bounds_methods    lower and upper bounds on the FDP, every method (Sup. 12)
    fdr_bounds_nulls      the same for WHISPER under three nulls (Sup. 13)
    cross_validation      out-of-fold scores (Sup. 15)
    tuning_sweep          54 random-forest configurations on the weak labels (Sup. 16)
    tuned_model           WHISPER with the selected configuration, benchmarked (Sup. 16)
    posneg_auc            AUC across weak-label counts (Sup. 18)
    label_diagnostics     reference-free metrics across weak-label counts (Sup. 19)
    level_benchmark       peptide / fragment level against SAINTq (Fig. 2d-e)
    go_enrichment         GO:CC enrichment and UpSet of 1% FDR preys (Fig. 6a-c);
                          needs internet
    hpa                   HPA localisation of 1% FDR preys (Fig. 6d-e, Sup. 21)
    supplementary_table   this dataset's Supplementary Table (1-10)
    go_tables             Supplementary Tables 11-15 (dataset 1)
    generate_gocc         regenerate the GO:CC reference
    generate_biogrid      regenerate the BioGRID reference (needs BIOGRID_ACCESS_KEY)
    generate_go_aspects   regenerate the GO aspect references

Cross-dataset steps (config/cross_dataset.yaml)
------------------------------------------------
    fdr_bounds_summary    Sup. 12-13 across datasets
    cv_per_fold           Sup. 15
    tuning_compare        Sup. 16
    transfer              Sup. 17
    variants_similarity   Sup. 1a
    int_spc               Sup. 1b
"""

import argparse
import os

import pandas as pd
import yaml

GAF_FILE = 'src/goa_human.gaf'
OBO_FILE = 'src/go-basic.obo'
TERM_CACHE = 'data/go_term_members_cache.json'
HPA_ALL = 'data/hpa_subcellular_location_all.tsv'
OLD_GO_RELEASE = 'data/go_release_2021-01-01'
TABLES_DIR = 'results/supplementary_tables'
REF_LABELS = {'go_large': 'GO:CC', 'bg_large': 'BioGRID'}


def load_config(path):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg['results_dir'] = cfg['results_dir'].rstrip('/')
    os.makedirs(cfg['results_dir'], exist_ok=True)
    return cfg


def out(cfg, name):
    return os.path.join(cfg['results_dir'], name)


def method_cfg(cfg, outdir=None, scores=None):
    m = cfg.get('methods', {})
    return {
        'name': cfg['dataset'],
        'features': scores or out(cfg, 'FDR_BaitSpecific_Hierarchical.csv'),
        'saintq': m.get('saintq'), 'saintexpress': m.get('saintexpress'),
        'limma': m.get('limma'), 'comppass': m.get('comppass'),
        'sfinx': m.get('sfinx'), 'ppnsaf': m.get('ppnsaf'), 'mist': m.get('mist'),
        'gts': [(src, path) for src, path in cfg['references'].items()],
        'outdir': outdir or cfg['results_dir'],
        'self_gene': cfg.get('self_gene', {}),
        'gaf': cfg.get('map_identifiers', False),
        'gaf_file': GAF_FILE,
        'limma_positive_only': False,
        'all_methods_in_main': cfg.get('all_methods_in_main', False),
        'fdr_bins_map_ids': cfg.get('fdr_bins_map_ids', False),
    }


def bounds_cfg(cfg):
    return {'dataset': cfg['dataset'], 'results_dir': cfg['results_dir'],
            'scores': out(cfg, 'FDR_BaitSpecific_Hierarchical.csv'),
            'features': out(cfg, 'features.csv'), 'methods': cfg.get('methods', {}),
            'references': cfg['references'], 'initial_positives': cfg['initial_positives'],
            'map_identifiers': cfg.get('map_identifiers', False), 'gaf_file': GAF_FILE}


def best_config(cfg):
    t = pd.read_csv(out(cfg, 'hyperparameter_tuning/hp_internal.csv'))
    return t.set_index('config')['cv_logloss'].sort_values().index[0]


def dataset_step(cfg, step):
    level = cfg.get('level', 'protein')

    if step == 'feature_engineering':
        from src.feature_engineering import read_matrix, run_feature_engineering
        feats = run_feature_engineering(
            read_matrix(cfg['input_file'], level), controls=cfg['controls'],
            baits=cfg.get('baits'), level=level,
            control_match=cfg.get('control_match', 'substring'),
            labelling_enzyme=cfg.get('labelling_enzyme', 'birA'),
            bait_self_protein=cfg.get('bait_self_protein', {}),
            drop_unnamed=cfg.get('drop_unnamed', False))
        feats.to_csv(out(cfg, 'features.csv'), index=False)

    elif step == 'train_and_fdr':
        from src.train_and_fdr import run_training_and_fdr
        scored, importance, protein = run_training_and_fdr(
            pd.read_csv(out(cfg, 'features.csv')),
            initial_positives=cfg['initial_positives'],
            initial_negatives=cfg.get('initial_negatives', 200),
            n_decoy_replicates=cfg.get('n_decoy_replicates', 30), level=level)
        scored.to_csv(out(cfg, cfg.get('scores_file', 'FDR_BaitSpecific_Hierarchical.csv')),
                      index=False)
        importance.sort_values('importance', ascending=False).to_csv(
            out(cfg, 'feature_importance_model_based.csv'), index=False)
        import matplotlib.pyplot as plt        # Sup. Fig. 20
        plt.figure(figsize=(6, 4))
        plt.barh(importance['feature'], importance['importance'])
        plt.xlabel('Mean feature importance (bagged RF)')
        plt.title('Feature importance (model-based)')
        plt.tight_layout()
        plt.savefig(out(cfg, 'feature_importance_model_based.pdf'), dpi=300)
        plt.close()
        if protein is not None:
            protein.to_csv(out(cfg, cfg['protein_scores_file']), index=False)

    elif step == 'benchmark':
        from src.benchmark import run_dataset
        run_dataset(method_cfg(cfg))
    elif step == 'overlap':
        from src.overlap import run_dataset
        run_dataset(method_cfg(cfg))
    elif step == 'fdr_bins':
        from src.fdr_bins import run_dataset
        run_dataset(method_cfg(cfg))
    elif step == 'label_quality':
        from src.label_quality import run
        run(out(cfg, 'FDR_BaitSpecific_Hierarchical.csv'), cfg['references'],
            cfg['results_dir'], cfg['initial_positives'], cfg.get('map_identifiers', False),
            GAF_FILE)
    elif step == 'auc_range':
        from src.auc_range import run_dataset
        run_dataset(method_cfg(cfg))
    elif step == 'per_bait':
        from src.per_bait import run_dataset
        run_dataset(method_cfg(cfg))
    elif step == 'saint_intersection':
        from src.saint_intersection import run_dataset
        run_dataset(method_cfg(cfg))
    elif step == 'go_aspects':
        from src.go_aspects import run_dataset
        run_dataset(method_cfg(cfg), cfg['go_aspects_dir'])
    elif step == 'gocc_sensitivity':
        from src.gocc_sensitivity import run
        c = method_cfg(cfg)
        r = cfg['reference_generation']
        c.update(deployed_cap=r['term_size_threshold'], bait_self_prey=r['bait_self_prey'])
        run(c, GAF_FILE, OBO_FILE, TERM_CACHE, OLD_GO_RELEASE)
    elif step == 'fdr_bounds_methods':
        from src.fdr_bounds import run_methods, unreachable_genes
        run_methods(bounds_cfg(cfg), unreachable_genes(GAF_FILE, OBO_FILE, HPA_ALL))
    elif step == 'fdr_bounds_nulls':
        from src.fdr_bounds import run_nulls, unreachable_genes
        run_nulls(bounds_cfg(cfg), unreachable_genes(GAF_FILE, OBO_FILE, HPA_ALL))
    elif step == 'cross_validation':
        from src.cross_validation import train_cv
        train_cv(out(cfg, 'features.csv'), cfg['initial_positives'],
                 out(cfg, 'FDR_BaitSpecific_Hierarchical_CV.csv'))
    elif step == 'tuning_sweep':
        from src.tuning import sweep
        os.makedirs(out(cfg, 'hyperparameter_tuning'), exist_ok=True)
        sweep(out(cfg, 'features.csv'), cfg['initial_positives'],
              out(cfg, 'hyperparameter_tuning/hp_internal.csv'))
    elif step == 'tuned_model':
        from src.benchmark import run_dataset
        from src.train_and_fdr import run_training_and_fdr
        from src.tuning import rf_params_for
        scored, _imp, _p = run_training_and_fdr(
            pd.read_csv(out(cfg, 'features.csv')), cfg['initial_positives'],
            rf_params=rf_params_for(best_config(cfg)))
        tuned = out(cfg, 'FDR_BaitSpecific_Hierarchical_BestConfig.csv')
        scored.to_csv(tuned, index=False)
        run_dataset(method_cfg(cfg, outdir=out(cfg, 'bestconfig'), scores=tuned))
    elif step == 'posneg_auc':
        from src.sensitivity import posneg_auc
        posneg_auc(method_cfg(cfg, scores=out(cfg, 'features.csv')))
    elif step == 'label_diagnostics':
        from src.sensitivity import label_diagnostics
        label_diagnostics(out(cfg, 'features.csv'), cfg['results_dir'])
    elif step == 'level_benchmark':
        from src.level_benchmark import run
        run(level, out(cfg, cfg['scores_file']), cfg['methods']['saintq'], cfg['input_file'],
            list(cfg['references'].items()), cfg['results_dir'])
    elif step == 'go_enrichment':
        from src.go_enrichment import run
        m = cfg['methods']
        run({'fdr_output': out(cfg, 'FDR_BaitSpecific_Hierarchical.csv'), 'limma': m['limma'],
             'saintq': m['saintq'], 'saintexpress': m['saintexpress'],
             'outdir': out(cfg, 'go_enrichment')})
    elif step == 'hpa':
        from src.hpa_annotation import run
        m = cfg['methods']
        run({'fdr_output': out(cfg, 'FDR_BaitSpecific_Hierarchical.csv'), 'limma': m['limma'],
             'saintq': m['saintq'], 'saintexpress': m['saintexpress'],
             'hpa_path': cfg['hpa_path'], 'hpa_bait': cfg['hpa_bait'], 'outdir': out(cfg, 'hpa')})
    elif step == 'supplementary_table':
        from src.supplementary_tables import build
        build(cfg['table_number'], cfg, TABLES_DIR)
    elif step == 'go_tables':
        from src.supplementary_tables import GO_TABLES, build_go
        for num in GO_TABLES:
            build_go(num, out(cfg, 'go_enrichment/enrichment_tables'), TABLES_DIR)
    elif step == 'generate_gocc':
        from src.generate_gocc_ground_truth import generate_gocc_ground_truth
        r = cfg['reference_generation']
        generate_gocc_ground_truth(r['baits'], out(cfg, 'gocc_reference_regenerated.csv'),
                                   GAF_FILE, OBO_FILE, r.get('term_size_threshold', 1000),
                                   r.get('bait_self_prey', {}), TERM_CACHE)
    elif step == 'generate_biogrid':
        from src.generate_biogrid_ground_truth import generate_biogrid_ground_truth
        r = cfg['reference_generation']
        generate_biogrid_ground_truth(r['baits'], out(cfg, 'biogrid_reference_regenerated.csv'),
                                      r.get('bait_self_prey', {}))
    elif step == 'generate_go_aspects':
        from src.go_aspects import generate_references
        r = cfg['reference_generation']
        generate_references(r['baits'], r['term_size_threshold'],
                            out(cfg, 'go_aspects_regenerated'), GAF_FILE, OBO_FILE, TERM_CACHE)
    else:
        raise SystemExit(f'Unknown step: {step}')


def cross_step(cross, step):
    cfgs = {k: load_config(p) for k, p in cross['datasets'].items()}
    out_dir = cross['results_dir']

    def ordered(keys):
        return [cfgs[k] for k in keys]

    if step == 'fdr_bounds_summary':
        from src.fdr_bounds_summary import run
        run([(k, cfgs[k]['dataset'].replace(' (', '\n('), cfgs[k]['results_dir'])
             for k in cross['benchmark_datasets']], os.path.join(out_dir, 'fdr_bounds_summary'))
    elif step == 'cv_per_fold':
        from src.cross_validation import evaluate
        evaluate([(method_cfg(c), out(c, 'FDR_BaitSpecific_Hierarchical_CV.csv'))
                  for c in ordered(cross['benchmark_datasets'])],
                 os.path.join(out_dir, 'hyperparameter_comparison'))
    elif step == 'tuning_compare':
        from src.tuning import compare, select
        hp = os.path.join(out_dir, 'hyperparameter_comparison')
        os.makedirs(hp, exist_ok=True)
        cs = ordered(cross['benchmark_datasets'])
        select({c['dataset']: out(c, 'hyperparameter_tuning/hp_internal.csv') for c in cs},
               os.path.join(hp, 'best_config_per_dataset_internal.csv'))
        compare([(c['dataset'], c['results_dir'], out(c, 'bestconfig')) for c in cs],
                os.path.join(hp, 'best_config_per_dataset_internal.csv'), hp)
    elif step == 'transfer':
        from src.transfer import run
        src_cfg = cfgs[cross['transfer']['source']]
        run(out(src_cfg, 'features.csv'), src_cfg['initial_positives'],
            [method_cfg(c) for c in ordered(cross['transfer']['targets'])],
            os.path.join(out_dir, 'transfer'))
    elif step == 'variants_similarity':
        from src.variants_similarity import run_method
        for method, variants in cross['variants'].items():
            paths = {}
            for name, (key, kind) in variants.items():
                c = cfgs[key]
                if kind == 'scores':
                    paths[name] = out(c, c.get('protein_scores_file', 'FDR_BaitSpecific_Hierarchical.csv'))
                else:
                    paths[name] = c['methods'][kind]
            run_method(method, paths, os.path.join(out_dir, 'variants_similarity'))
    elif step == 'int_spc':
        from src.int_spc import run
        ci, cs = cfgs[cross['int_spc']['intensity']], cfgs[cross['int_spc']['spectral_counts']]
        run(out(ci, 'features.csv'), out(cs, 'features.csv'),
            out(ci, 'FDR_BaitSpecific_Hierarchical.csv'), out(cs, 'FDR_BaitSpecific_Hierarchical.csv'),
            {REF_LABELS[k]: v for k, v in ci['references'].items()},
            os.path.join(out_dir, 'int_plus_spc'), ci['initial_positives'])
    else:
        raise SystemExit(f'Unknown cross-dataset step: {step}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--step', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.get('cross_dataset'):
        cross_step(cfg, args.step)
    else:
        dataset_step(cfg, args.step)


if __name__ == '__main__':
    main()
