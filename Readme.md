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
| **Data Ingestion** | [config.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/config.py), [data_loader.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/data_loader.py) | `Pooch`, `masks_pred.npz` retrieval |
| **Cell Perception & Model Initialization** | [model_loader.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/model_loader.py) | Trackastra `general_2d` Vision Transformer weights |
| **Cell Tracking & Association** | [tracker.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/tracker.py) | `Trackastra.track()`, candidate graph association |
| **Visualization & Video Player** | [visualize.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/visualize.py), [export_video.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/export_video.py), [generate_web_visualizer.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/generate_web_visualizer.py) | Animated GIF exporter & HTML5 interactive video player |
| **Exploratory Data Analysis (EDA)** | [eda.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/eda.py) | Multi-panel morphology, spatial density, and kinematic analytics |
| **Unsupervised Cell Phenotyping** | [phenotyping.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/phenotyping.py) | PCA projection, K-Means clustering, and cell state discovery |
| **Pipeline Integration** | [main.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/main.py) | End-to-end execution, CSV export, video flags & phenotyping |

---

## 🎬 Quick Usage

```bash
# 1. Run Pipeline with Video, Web Viewer & Phenotyping Clustering
python main.py --export-csv --export-video --web-viewer --cluster

# 2. Run Standalone Cell Phenotyping & PCA Clustering
python phenotyping.py

# 3. Run Exploratory Data Analysis (EDA)
python eda.py

# 4. Launch Interactive Time-Lapse Web Video Player
# Open cell_tracker_viewer.html in any web browser
```
