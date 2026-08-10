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
| **Visualization & Inspection** | [visualize.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/visualize.py) | `Matplotlib` static plots, `Napari` 4D interactive viewer |
| **Pipeline Integration** | [main.py](file:///c:/Users/visha/Desktop/Computer%20Vision/Cell_Track/main.py) | End-to-end execution & per-frame cell statistics |
