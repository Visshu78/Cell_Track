# Project Objective & Research Architecture

## 🎯 Primary Project Objective

> **Develop an intelligent computer vision pipeline that automatically detects, tracks, and reconstructs the lineage of cells in time-lapse microscopy videos while quantifying their spatial, morphological, migratory, and proliferative behavior across diverse imaging conditions.**

---

## 🏗️ System Architecture & Research Modules

```
                    OUR PROJECT
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   CELL PERCEPTION   CELL TRACKING   CELL EVENTS
        │                │                │
   Detection         Identity        Division
   Segmentation      Association     Death
   Morphology        Occlusion       Appearance
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                  LINEAGE GRAPH
                         │
                         ▼
                BEHAVIOR ANALYSIS
                         │
                         ▼
               BIOLOGICAL INSIGHTS
```

---

## 🔬 Four Research Modules Overview

### 1. 👁️ Cell Perception (Detection, Segmentation, & Morphology)
* **Detection**: Identifying individual cell centroids and spatial localization within each video frame $t$.
* **Segmentation**: Delineating fine pixel-level cell boundaries (using 2D/3D segmentation masks).
* **Morphology**: Extracting spatial attributes such as cell area, perimeter, circularity, eccentricity, and orientation.

### 2. 🔗 Cell Tracking (Identity, Association, & Occlusions)
* **Identity Maintenance**: Assigning unique, persistent IDs to individual cells across time frames.
* **Temporal Association**: Linking cell positions between consecutive frames ($t \rightarrow t+1$) using spatial features and Vision Transformers (e.g., Trackastra).
* **Occlusion Handling**: Managing temporary cell overlaps, missing detections, or out-of-focus frames.

### 3. 🧬 Cell Events (Division, Death, & Transitions)
* **Cell Division (Mitosis)**: Detecting parent cell division into daughter cells and constructing branching lineage trees.
* **Cell Death (Apoptosis)**: Identifying cell degeneration, shrinkage, and disappearance.
* **Appearance & Disappearance**: Tracking cells entering or exiting the imaging field of view.

### 4. 📊 Lineage Graph, Behavior Analysis, & Biological Insights
* **Lineage Graph**: Building a directed acyclic graph (DAG) representing cell families across generations.
* **Behavior Analysis**: Quantifying cell motility, migration velocity, directionality, and proliferation rate.
* **Biological Insights**: Generating actionable phenotypic and clinical insights for biomedical research.

---

## 💻 Current Codebase Mapping

| Research Module | Current Project File | Functions / Libraries Used |
| :--- | :--- | :--- |
| **Cell Morphology** | [morphology.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/morphology.py) | Area, circularity, eccentricity per frame |
| **Lineage Events** | [lineage.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/lineage.py) | Mitosis/death detection, DAG family trees |
| **Behavior Analytics** | [behavior.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/behavior.py) | Cell speed, displacement, directionality |
| **Interactive Lineage Tree** | [generate_lineage_visualizer.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/generate_lineage_visualizer.py) | D3.js pedigree trees & Gantt timeline |
| **Novel BioTrack-X Architecture** | [biotrack_x/](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/biotrack_x/) | Unified ST-GT + Erlang prior + TTA uncertainty |
| **Pipeline Integration** | [main.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/main.py) | End-to-end CLI with all flags |

---

## 🧠 BioTrack-X: Novel Architecture

BioTrack-X is a **genuinely novel unified Spatio-Temporal Graph Transformer** for cell tracking, combining innovations from four state-of-the-art papers into a single differentiable PyTorch model.

### Key Novelties (not present in any single existing architecture)

| Feature | TrackFormer | MOTR | Cell-TRACTR | HOCT | Kaiser MHT | **BioTrack-X** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| End-to-end joint track + segment | YES | YES | YES | YES | NO | **YES** |
| Edge-centric division query spawning | NO | NO | NO | YES | YES | **YES** |
| Erlang biological cell-cycle prior | NO | NO | NO | NO | YES | **YES** |
| Aleatoric TTA position uncertainty | NO | NO | NO | NO | YES | **YES** |
| Full-video long-range attention (T>=30) | NO | NO | NO | NO | NO | **YES** |

### Architecture Modules

```
biotrack_x/
  encoder.py        ResNet-18 CNN + 4-shift TTA aleatoric uncertainty
  transformer.py    Spatio-Temporal Graph Transformer (ST-GT)
                    + DivisionQueryHead (edge-centric mitosis)
  erlang_prior.py   Erlang(alpha=2, beta) biological cell-cycle prior
  loss.py           Joint loss: Hungarian + Dice + BCE + Erlang
  model.py          BioTrackX master nn.Module (~1.4M params)
  inference.py      Drop-in adapter replacing Trackastra
```

### Loss Function

```
L_total = lambda1 * L_track   (Hungarian centroid regression)
        + lambda2 * L_seg     (Dice + BCE mask loss)
        + lambda3 * L_div     (division binary cross-entropy)
        + lambda4 * L_bio     (Erlang biological prior)
```

---

## 🎬 Quick Usage

```bash
# 1. Run standard pipeline (Trackastra tracker)
python main.py --export-csv --export-video --web-viewer --cluster

# 2. Run with BioTrack-X novel architecture (replaces Trackastra)
python main.py --biotrackx --no-show

# 3. BioTrack-X on first N frames only (faster for testing)
python main.py --biotrackx --no-show --subset 5

# 4. Run Exploratory Data Analysis (EDA)
python eda.py

# 5. Run Standalone Cell Phenotyping & PCA Clustering
python phenotyping.py

# 6. Launch Interactive Lineage Tree Viewer
# Open lineage_tree_viewer.html in any web browser

# 7. Launch Interactive Time-Lapse Web Video Player
# Open cell_tracker_viewer.html in any web browser
```
