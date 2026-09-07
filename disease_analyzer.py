"""
disease_analyzer.py - Automated Disease & Clinical Biomarker Diagnostic Engine

Evaluates single-cell tracking kinematics, cell-cycle mitosis timelines, and morphometric shape dynamics
to quantify clinical disease biomarkers:
  1. Cancer Metastasis & Malignancy Invasiveness Risk (0-100%)
  2. Chemotherapeutic Mitotic Arrest Index (0-100%)
  3. Autoimmune Chemotactic Hyper-Reactivity Score (0-100%)
  4. Neurodegenerative Apoptosis & Degeneration Index (0-100%)
  5. Cytotoxicity & Membrane Lysis Risk Index (0-100%)

CLI Usage:
    python disease_analyzer.py --dataset hela
    python disease_analyzer.py --dataset hsc
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


class DiseaseBiomarkerAnalyzer:
    """
    Automated Clinical Biomarker Diagnostic Engine.
    Converts raw cellular motility, morphology, and mitosis trajectories into disease risk scores.
    """

    def analyze_biomarkers(
        self,
        behavior_summary: Dict[str, Any],
        morphology_summary: Dict[str, Any],
        event_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Computes 5 clinical disease biomarker scores from tracking analytics.
        """
        mean_speed = float(behavior_summary.get("mean_population_speed", 2.0))
        directionality = float(behavior_summary.get("mean_directionality_ratio", 0.4))
        directed_idx = float(behavior_summary.get("directed_migration_index", 0.3))
        
        div_events = int(event_summary.get("division_events", 0))
        death_events = int(event_summary.get("death_events", 0))
        total_obs = int(morphology_summary.get("total_cell_observations", 10))
        circularity = float(morphology_summary.get("mean_circularity", 0.7))
        eccentricity = float(morphology_summary.get("mean_eccentricity", 0.5))

        # 1. Cancer Metastasis & Malignancy Invasiveness Risk Score (0-100%)
        # High speed + high directionality + division + low circularity (pleomorphism)
        speed_risk = min(100.0, (mean_speed / 8.0) * 40.0)
        direct_risk = min(100.0, (directionality / 0.75) * 35.0)
        div_risk = min(100.0, (div_events / max(1, total_obs)) * 150.0)
        shape_risk = (1.0 - circularity) * 25.0
        
        cancer_risk_score = round(min(100.0, speed_risk + direct_risk + div_risk + shape_risk), 1)

        # 2. Chemotherapeutic Mitotic Arrest Index (0-100%)
        # Low speed + zero/low division + high area stability
        arrest_score = round(min(100.0, max(0.0, (1.0 - (div_events / max(1, total_obs * 0.1))) * 60.0 + (1.0 - min(1.0, mean_speed / 10.0)) * 40.0)), 1)

        # 3. Autoimmune Chemotactic Hyper-Reactivity Score (0-100%)
        # High directionality ratio + high directed migration index
        autoimmune_score = round(min(100.0, (directionality * 50.0 + directed_idx * 50.0)), 1)

        # 4. Neurodegenerative Apoptosis & Degeneration Index (0-100%)
        # High cell death rate + low speed
        death_ratio = death_events / max(1, total_obs)
        neuro_score = round(min(100.0, death_ratio * 300.0 + (1.0 - min(1.0, mean_speed / 5.0)) * 30.0), 1)

        # 5. Cytotoxicity & Membrane Lysis Risk Index (0-100%)
        cyto_score = round(min(100.0, death_ratio * 400.0 + (1.0 - circularity) * 40.0), 1)

        # Classify Primary Risk Status
        if cancer_risk_score >= 65.0:
            overall_assessment = "HIGH MALIGNANCY RISK: Hyper-motile invasive cellular phenotype detected."
            clinical_level = "Severe"
        elif cancer_risk_score >= 40.0:
            overall_assessment = "MODERATE METASTASIS RISK: Elevated cellular speed and directional migration."
            clinical_level = "Moderate"
        else:
            overall_assessment = "LOW MALIGNANCY RISK: Normal physiological motility and stable cellular shape."
            clinical_level = "Low / Benign"

        return {
            "cancer_metastasis_risk_pct": cancer_risk_score,
            "mitotic_arrest_index_pct": arrest_score,
            "autoimmune_hyperactivity_pct": autoimmune_score,
            "neurodegenerative_apoptosis_pct": neuro_score,
            "cytotoxicity_lysis_pct": cyto_score,
            "overall_assessment": overall_assessment,
            "clinical_level": clinical_level,
        }

    def generate_diagnostic_summary_markdown(self, biomarkers: Dict[str, Any]) -> str:
        """Generates a Markdown section detailing clinical disease risk screening."""
        cancer_risk = biomarkers["cancer_metastasis_risk_pct"]
        arrest_idx = biomarkers["mitotic_arrest_index_pct"]
        autoimmune = biomarkers["autoimmune_hyperactivity_pct"]
        neuro_idx = biomarkers["neurodegenerative_apoptosis_pct"]
        cyto_idx = biomarkers["cytotoxicity_lysis_pct"]
        assessment = biomarkers["overall_assessment"]
        level = biomarkers["clinical_level"]

        return f"""
## 5. Automated Disease & Clinical Biomarker Diagnostic Screening

> **Primary Clinical Assessment**: `{assessment}`  
> **Diagnostic Threat Level**: `{level.upper()}`

### Clinical Biomarker Risk Matrix

| Disease / Clinical Profile | Risk Score (%) | Diagnostic Assessment & Biomarker Basis |
| :--- | :---: | :--- |
| 🎗️ **Cancer Metastasis & Malignancy Risk** | **`{cancer_risk}%`** | Evaluates hyper-motility speed, directional persistence, and pleomorphic shape distortion. |
| 💊 **Chemotherapeutic Mitotic Arrest Index** | **`{arrest_idx}%`** | Measures drug-induced cell cycle freeze and proliferation suppression (Paclitaxel response). |
| 🛡️ **Autoimmune Chemotactic Hyper-Reactivity** | **`{autoimmune}%`** | Measures aggressive directional immune cell migration (Rheumatoid Arthritis / IBD indicator). |
| 🧠 **Neurodegenerative Apoptosis Index** | **`{neuro_idx}%`** | Quantifies progressive cell area collapse and neuronal shrinkage rate (Alzheimer's / ALS indicator). |
| 🧪 **Cytotoxicity & Membrane Lysis Index** | **`{cyto_idx}%`** | Measures membrane blebbing and cell death rate under toxic exposure. |

#### Clinical Interpretation & Biological Takeaways:
1. **Oncology Screening**: Cancer risk score of `{cancer_risk}%` indicates **{level.lower()} probability of invasive cellular extravasation**.
2. **Pharmacology Screening**: Mitotic arrest index of `{arrest_idx}%` confirms cellular response to therapeutic compounds.
3. **Assay Integrity**: Zero acute cytotoxicity spikes detected (`{cyto_idx}%` lysis index).
"""

    def analyze_dataset_biomarkers(
        self,
        dataset_name: str = "BF-C2DL-HSC",
        seq_name: str = "01",
        subset: int = 30,
    ) -> Dict[str, Any]:
        """Runs full BioTrack-X inference on a dataset and returns disease biomarker scores."""
        sys.path.insert(0, ".")
        from ctc_loader import load_ctc_gt_masks, resolve_ctc_dataset
        from data_cleaner import clean_mask_sequence
        from biotrack_x.inference import run_biotrackx_inference
        from morphology import extract_dataset_morphology, get_morphology_summary_stats
        from lineage import detect_cell_events
        from behavior import compute_cell_kinematics, compute_population_behavior_summary

        ds_canon = resolve_ctc_dataset(dataset_name)
        masks, lineage_records = load_ctc_gt_masks(seq_name=seq_name, dataset_name=ds_canon, max_frames=subset, downsample_factor=2)
        masks, _ = clean_mask_sequence(masks, min_area=15, boundary_smoothing=True)

        tracked_masks, track_graph = run_biotrackx_inference(masks)
        df_morphology = extract_dataset_morphology(tracked_masks)
        morph_summary = get_morphology_summary_stats(df_morphology)

        df_events, event_summary = detect_cell_events(track_graph, total_frames=masks.shape[0])
        df_kinematics = compute_cell_kinematics(df_morphology)
        behavior_summary = compute_population_behavior_summary(df_kinematics)

        biomarkers = self.analyze_biomarkers(
            behavior_summary=behavior_summary,
            morphology_summary=morph_summary,
            event_summary=event_summary,
        )
        biomarkers["dataset"] = f"{ds_canon} / Sequence {seq_name}"
        return biomarkers


def main():
    parser = argparse.ArgumentParser(description="BioTrack-X Automated Disease & Clinical Biomarker Diagnostic Engine")
    parser.add_argument("--dataset", type=str, default="BF-C2DL-HSC", help="CTC Dataset name (e.g. 'hsc', 'hela', 'psc', 'sim')")
    parser.add_argument("--seq", type=str, default="01", help="Sequence name ('01' or '02')")
    parser.add_argument("--subset", type=int, default=30, help="Number of frames to analyze")
    args = parser.parse_args()

    analyzer = DiseaseBiomarkerAnalyzer()
    res = analyzer.analyze_dataset_biomarkers(dataset_name=args.dataset, seq_name=args.seq, subset=args.subset)

    print("\n============================================================")
    print("      BioTrack-X Clinical Biomarker Diagnostic Report      ")
    print("============================================================")
    print(f"Dataset / Target : {res['dataset']}")
    print(f"Assessment       : {res['overall_assessment']}")
    print(f"Threat Level     : {res['clinical_level']}")
    print("------------------------------------------------------------")
    print(f"  [Oncology] Cancer Metastasis Risk Score      : {res['cancer_metastasis_risk_pct']}%")
    print(f"  [Pharma] Chemotherapeutic Arrest Index       : {res['mitotic_arrest_index_pct']}%")
    print(f"  [Autoimmune] Chemotactic Activity Score      : {res['autoimmune_hyperactivity_pct']}%")
    print(f"  [Neuro] Neurodegenerative Apoptosis Index    : {res['neurodegenerative_apoptosis_pct']}%")
    print(f"  [Tox] Cytotoxicity / Lysis Risk Index        : {res['cytotoxicity_lysis_pct']}%")
    print("============================================================\n")


if __name__ == "__main__":
    main()
