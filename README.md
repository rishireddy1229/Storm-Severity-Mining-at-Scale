# Storm Severity Mining at Scale
### COSC 6335 – Data Mining | Spring 2026 | University of Houston

**Damage Prediction, Spatial Hazard Clustering, and Episode-Level Association Discovery on the NOAA Storm Events Database**

---

## Team
| Member | Role |
|--------|--------------|------|
| Sathwik Sangeetham | EDA, preprocessing, density estimation, data storytelling |
| Rishitha Reddy Mekala | Classification/prediction modeling and evaluation |
| Viraat Mindaguditi | Clustering, spatial mining, autoencoder embeddings |
| Anjani Kumar Avadhanam  | Association/sequence mining and outlier detection |

---

## Core Idea

We build an **end-to-end storm-mining pipeline** with three interconnected components:

1. **Damage Severity Prediction** — Classify each storm event into Low / Medium / High / Catastrophic economic damage tiers using Decision Trees, Random Forest, kNN, SVM, and a feedforward neural network.

2. **Spatial Hazard Zone Discovery** — Use DBSCAN on geocoordinates to find density-based clusters of high-damage events across the U.S., revealing natural "hazard zones." Then profile each zone's damage characteristics.

3. **Episode-Level Association & Sequence Mining** — NOAA groups co-occurring storm events into "episodes" (via `episode_id`). We treat each episode as a transaction basket and mine co-occurrence rules (Apriori) and sequential escalation patterns (e.g., Hail → Thunderstorm Wind → Tornado).

**The novelty** (what makes this more than a basic weather classification project): We combine all three into a **"storm hazard profile"** — region-level risk scores derived from spatial clusters + temporal association patterns + learned damage predictors. This gives a multi-faceted, actionable picture of severe weather risk.

---

## Dataset: NOAA Storm Events Database

### Overview
- **Source:** NOAA National Centers for Environmental Information (NCEI)
- **Type:** Official U.S. government publication
- **Time Range:** 1996–2025 (we exclude pre-1996 due to inconsistent event typing)
- **Size:** ~1.5 million event records, ~400 MB of CSVs
- **Coverage:** All 50 U.S. states
- **Attributes:** 51 columns per record
- **Format:** CSV (gzipped), one file per year per table

### Download Links
```
# Three tables per year — details, locations, fatalities
# Pattern: StormEvents_{table}-ftp_v1.0_d{YEAR}_c{CREATED}.csv.gz

Base URL: https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

# Example files:
# StormEvents_details-ftp_v1.0_d2024_c20250120.csv.gz
# StormEvents_locations-ftp_v1.0_d2024_c20250120.csv.gz  
# StormEvents_fatalities-ftp_v1.0_d2024_c20250120.csv.gz
```

**Documentation:**
- Field descriptions: https://www.ncei.noaa.gov/stormevents/
- FAQ & version history: https://www.ncei.noaa.gov/stormevents/faq.jsp
- CSV format spec (PDF): https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/Storm-Data-Bulk-csv-Format.pdf

### Key Fields We Use (from `StormEvents_details`)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `episode_id` | int | Groups co-occurring events into one storm episode | 61280 |
| `event_id` | int | Unique ID per event (primary key) | 383097 |
| `state` | string | U.S. state | TEXAS |
| `event_type` | string | One of 48 standardized categories | Tornado, Hail, Flash Flood |
| `begin_date_time` | datetime | Event start | 01-JAN-2020 14:30:00 |
| `end_date_time` | datetime | Event end | 01-JAN-2020 15:00:00 |
| `begin_lat` / `begin_lon` | float | Start geocoordinates | 29.76, -95.37 |
| `end_lat` / `end_lon` | float | End geocoordinates | 29.80, -95.40 |
| `damage_property` | string | Property damage (needs parsing!) | "25K", "1.5M", "0" |
| `damage_crops` | string | Crop damage (needs parsing!) | "500K", "0" |
| `injuries_direct` | int | Direct injuries | 3 |
| `deaths_direct` | int | Direct deaths | 0 |
| `tor_f_scale` | string | Tornado EF scale (if tornado) | EF1, EF3 |
| `magnitude` | float | Hail size (inches) or wind speed (knots) | 1.75 |
| `flood_cause` | string | Flood cause (if flood) | Heavy Rain |
| `episode_narrative` | string | Text description of episode | "A line of severe..." |
| `event_narrative` | string | Text description of individual event | "Large hail reported..." |

### Critical Preprocessing Notes

**Damage field parsing** — This is the #1 gotcha. Damage fields are STRINGS, not numbers:
```
"25K"   → 25,000
"1.50M" → 1,500,000
"0.00K" → 0
""      → NA (missing)
"0"     → 0
```

```r
# R function to parse damage
parse_damage <- function(x) {
  x <- toupper(trimws(x))
  x[x == "" | is.na(x)] <- "0"
  multiplier <- ifelse(grepl("K$", x), 1e3,
                ifelse(grepl("M$", x), 1e6,
                ifelse(grepl("B$", x), 1e9, 1)))
  as.numeric(gsub("[KMB]$", "", x)) * multiplier
}
```

```python
# Python equivalent
def parse_damage(s):
    if pd.isna(s) or s == '':
        return 0
    s = str(s).upper().strip()
    if s.endswith('K'):
        return float(s[:-1]) * 1e3
    elif s.endswith('M'):
        return float(s[:-1]) * 1e6
    elif s.endswith('B'):
        return float(s[:-1]) * 1e9
    return float(s)
```

**Missing coordinates:** ~8% of records have missing `begin_lat`/`begin_lon`. Strategy: drop for spatial analyses, keep for non-spatial tasks.

**Event duration:** Compute from `begin_date_time` and `end_date_time` (in minutes).

**Season engineering:** Extract month → map to season (Winter: Dec-Feb, Spring: Mar-May, Summer: Jun-Aug, Fall: Sep-Nov).

### The 48 Event Types (post-1996 standardization)
Astronomical Low Tide, Avalanche, Blizzard, Coastal Flood, Cold/Wind Chill, Debris Flow, Dense Fog, Dense Smoke, Drought, Dust Devil, Dust Storm, Excessive Heat, Extreme Cold/Wind Chill, Flash Flood, Flood, Freezing Fog, Frost/Freeze, Funnel Cloud, Hail, Heat, Heavy Rain, Heavy Snow, High Surf, High Wind, Hurricane/Typhoon, Ice Storm, Lake-Effect Snow, Lakeshore Flood, Lightning, Marine Dense Fog, Marine Hail, Marine Heavy Freezing Spray, Marine High Wind, Marine Strong Wind, Marine Thunderstorm Wind, Marine Tropical Depression, Marine Tropical Storm, Rip Current, Seiche, Sleet, Storm Surge/Tide, Strong Wind, Thunderstorm Wind, Tornado, Tropical Depression, Tropical Storm, Tsunami, Waterspout, Wildfire, Winter Storm, Winter Weather.

---

## Implementation Plan (Method by Method)

### 1. Data Download & Loading

```bash
# Download all detail files (1996-2025)
mkdir -p data/raw
for year in $(seq 1996 2025); do
  wget -P data/raw/ "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d${year}*.csv.gz" 
done
gunzip data/raw/*.gz
```

```r
# R: Load and combine all years
library(tidyverse)
files <- list.files("data/raw", pattern = "details.*\\.csv$", full.names = TRUE)
storms <- map_dfr(files, read_csv, show_col_types = FALSE)
cat("Total records:", nrow(storms), "\n")  # Should be ~1.5M
```

```python
# Python: Load and combine all years
import pandas as pd
import glob
files = sorted(glob.glob("data/raw/StormEvents_details*.csv"))
storms = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
print(f"Total records: {len(storms):,}")
```

---

### 2. Preprocessing + EDA (Sathwik) — Ch. 2–3, XIII

**Feature engineering:**
```r
storms <- storms %>%
  mutate(
    damage_property_num = parse_damage(DAMAGE_PROPERTY),
    damage_crops_num    = parse_damage(DAMAGE_CROPS),
    total_damage        = damage_property_num + damage_crops_num,
    duration_min        = as.numeric(difftime(END_DATE_TIME, BEGIN_DATE_TIME, units = "mins")),
    month               = month(BEGIN_DATE_TIME),
    season              = case_when(
      month %in% c(12, 1, 2)  ~ "Winter",
      month %in% c(3, 4, 5)   ~ "Spring",
      month %in% c(6, 7, 8)   ~ "Summer",
      month %in% c(9, 10, 11) ~ "Fall"
    ),
    hour                = hour(BEGIN_DATE_TIME),
    year                = year(BEGIN_DATE_TIME)
  )
```

**Target variable for classification (damage severity tiers):**
```r
# Only on events with damage > 0
damage_events <- storms %>% filter(total_damage > 0)
quantiles <- quantile(damage_events$total_damage, probs = c(0.25, 0.50, 0.75))

storms <- storms %>%
  mutate(damage_class = case_when(
    total_damage == 0                    ~ "None",
    total_damage <= quantiles[1]         ~ "Low",
    total_damage <= quantiles[2]         ~ "Medium",
    total_damage <= quantiles[3]         ~ "High",
    TRUE                                 ~ "Catastrophic"
  ))
```

**EDA deliverables:**
- Yearly event count trend (line plot)
- Top 15 event types by frequency (bar chart)
- Damage distribution by event type (box plots, log scale)
- Seasonal distribution heatmap (event_type × season)
- Geographic heatmap of event density (leaflet)
- Correlation matrix of numeric features
- Missing value analysis

---

### 3. Density Estimation (Sathwik) — Ch. V

```r
library(ggplot2)

# KDE on damage distributions per event type
top_types <- storms %>%
  filter(total_damage > 0) %>%
  count(EVENT_TYPE, sort = TRUE) %>%
  head(10) %>%
  pull(EVENT_TYPE)

storms %>%
  filter(EVENT_TYPE %in% top_types, total_damage > 0) %>%
  ggplot(aes(x = log10(total_damage), fill = EVENT_TYPE)) +
  geom_density(alpha = 0.4) +
  labs(title = "KDE of Log-Damage by Event Type", x = "Log10(Total Damage $)")

# Parametric MLE fit (e.g., lognormal) vs. non-parametric KDE
library(MASS)
tornado_damage <- storms %>%
  filter(EVENT_TYPE == "Tornado", total_damage > 0) %>%
  pull(total_damage)
fit <- fitdistr(tornado_damage, "lognormal")
# Compare fit vs. empirical KDE
```

---

### 4. Classification (Rishitha) — Ch. IX–X

```r
library(caret)

# Prepare features
clf_data <- storms %>%
  filter(damage_class != "None") %>%
  select(EVENT_TYPE, duration_min, season, hour, BEGIN_LAT, BEGIN_LON,
         INJURIES_DIRECT, DEATHS_DIRECT, damage_class) %>%
  drop_na() %>%
  mutate(across(where(is.character), as.factor))

# SMOTE for class imbalance
library(smotefamily)

# 10-fold stratified CV setup
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE,
                     summaryFunction = multiClassSummary)

# Decision Tree
dt_model <- train(damage_class ~ ., data = clf_data, method = "rpart",
                  trControl = ctrl, metric = "F1")

# Random Forest
rf_model <- train(damage_class ~ ., data = clf_data, method = "rf",
                  trControl = ctrl, metric = "F1", ntree = 200)

# kNN
knn_model <- train(damage_class ~ ., data = clf_data, method = "knn",
                   trControl = ctrl, metric = "F1",
                   tuneGrid = data.frame(k = c(3, 5, 7, 11, 15)))

# SVM
svm_model <- train(damage_class ~ ., data = clf_data, method = "svmRadial",
                   trControl = ctrl, metric = "F1")
```

```python
# Neural Network (Keras) — feedforward
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(n_features,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(4, activation='softmax')  # 4 classes
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)
```

**Evaluation:**
- Accuracy, Macro-F1, ROC-AUC (one-vs-rest), confusion matrix
- Feature importance: Random Forest importance + permutation importance
- Compare all 5 models in a summary table

---

### 5. Clustering + Spatial Mining (Viraat) — Ch. VIII, XII, XIV

```r
library(dbscan)

# DBSCAN on high-damage event coordinates
high_damage <- storms %>%
  filter(total_damage >= quantiles[3]) %>%  # top 25%
  filter(!is.na(BEGIN_LAT), !is.na(BEGIN_LON)) %>%
  select(BEGIN_LAT, BEGIN_LON)

# eps = ~50km in degrees ≈ 0.45, minPts = 50 (tune these!)
db <- dbscan(high_damage, eps = 0.45, minPts = 50)
high_damage$cluster <- db$cluster
cat("Clusters found:", max(db$cluster), "\n")
cat("Noise points:", sum(db$cluster == 0), "\n")

# Visualize clusters on Leaflet map
library(leaflet)
pal <- colorFactor("Set1", domain = unique(high_damage$cluster))
leaflet(high_damage %>% filter(cluster > 0)) %>%
  addTiles() %>%
  addCircleMarkers(~BEGIN_LON, ~BEGIN_LAT, color = ~pal(cluster), radius = 2)
```

```r
# K-Means on event profiles
profile_data <- storms %>%
  filter(!is.na(total_damage), total_damage > 0) %>%
  select(duration_min, total_damage, INJURIES_DIRECT, DEATHS_DIRECT) %>%
  drop_na() %>%
  scale()

# Elbow method
wss <- sapply(2:15, function(k) kmeans(profile_data, k, nstart = 25)$tot.withinss)
plot(2:15, wss, type = "b", xlab = "k", ylab = "Total WSS")

km <- kmeans(profile_data, centers = 5, nstart = 25)

# Silhouette validation
library(cluster)
sil <- silhouette(km$cluster, dist(profile_data))
mean(sil[, 3])  # average silhouette width
```

```r
# Hierarchical Clustering on event-type profiles
type_profiles <- storms %>%
  group_by(EVENT_TYPE) %>%
  summarise(
    avg_damage    = mean(total_damage, na.rm = TRUE),
    avg_duration  = mean(duration_min, na.rm = TRUE),
    avg_injuries  = mean(INJURIES_DIRECT, na.rm = TRUE),
    avg_deaths    = mean(DEATHS_DIRECT, na.rm = TRUE),
    event_count   = n()
  ) %>%
  column_to_rownames("EVENT_TYPE")

hc <- hclust(dist(scale(type_profiles)), method = "ward.D2")
plot(hc, cex = 0.7, main = "Event Type Dendrogram")

# Spatial autocorrelation (Moran's I)
library(spdep)
# Compute on cluster-level average damage
```

---

### 6. Association + Sequence Mining (Anjani) — Ch. XI

```r
library(arules)

# Build transaction baskets from episodes
episode_baskets <- storms %>%
  filter(!is.na(EPISODE_ID)) %>%
  group_by(EPISODE_ID) %>%
  summarise(event_types = list(unique(EVENT_TYPE))) %>%
  filter(lengths(event_types) >= 2)  # only multi-event episodes

# Convert to transactions
trans <- as(episode_baskets$event_types, "transactions")
summary(trans)

# Apriori
rules <- apriori(trans, parameter = list(supp = 0.01, conf = 0.3, minlen = 2))
rules <- sort(rules, by = "lift", decreasing = TRUE)
inspect(head(rules, 20))

# Visualize
library(arulesViz)
plot(rules, method = "graph", limit = 20)
```

```r
# Sequential pattern mining on temporally ordered events within episodes
library(arulesSequences)

# Build sequences: order events within each episode by begin_date_time
episode_sequences <- storms %>%
  filter(!is.na(EPISODE_ID)) %>%
  arrange(EPISODE_ID, BEGIN_DATE_TIME) %>%
  group_by(EPISODE_ID) %>%
  mutate(seq_order = row_number()) %>%
  ungroup()

# Write to basket format for cSPADE
# sequenceID, eventID, SIZE, items
write_basket <- episode_sequences %>%
  select(EPISODE_ID, seq_order, EVENT_TYPE) %>%
  mutate(SIZE = 1)

write.table(write_basket, "data/sequences.txt", 
            row.names = FALSE, col.names = FALSE, quote = FALSE)

# Run cSPADE
seq_data <- read_baskets("data/sequences.txt", info = c("sequenceID", "eventID", "SIZE"))
freq_seq <- cspade(seq_data, parameter = list(support = 0.01))
inspect(head(sort(freq_seq, by = "support"), 20))
```

**Expected findings:** Rules like `{Hail, Thunderstorm Wind} → {Tornado}` with high lift, sequential patterns showing escalation chains.

---

### 7. Outlier Detection (Anjani) — Ch. VII

```r
# Z-score based outlier detection (per event type)
outliers_zscore <- storms %>%
  filter(total_damage > 0) %>%
  group_by(EVENT_TYPE) %>%
  mutate(z_damage = (total_damage - mean(total_damage)) / sd(total_damage)) %>%
  filter(abs(z_damage) > 3) %>%
  ungroup()

cat("Z-score outliers:", nrow(outliers_zscore), "\n")
```

```python
# Isolation Forest
from sklearn.ensemble import IsolationForest

features = storms_clean[['total_damage', 'duration_min', 'INJURIES_DIRECT', 
                          'DEATHS_DIRECT', 'BEGIN_LAT', 'BEGIN_LON']].dropna()

iso = IsolationForest(contamination=0.05, random_state=42)
features['anomaly'] = iso.fit_predict(features)
outliers_iso = features[features['anomaly'] == -1]
print(f"Isolation Forest outliers: {len(outliers_iso)}")
```

```python
# Local Outlier Factor (LOF)
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
features['lof_label'] = lof.fit_predict(features.drop(columns=['anomaly']))
outliers_lof = features[features['lof_label'] == -1]
```

**Deliverable:** Characterize top outliers — what makes them anomalous? Map them. Use in storytelling.

---

### 8. Autoencoders (Viraat) — Ch. VI, X

```python
from tensorflow import keras
import numpy as np

# Shallow autoencoder on engineered features
n_features = X_scaled.shape[1]  # e.g., 8-10 features

encoder = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(n_features,)),
    keras.layers.Dense(6, activation='relu'),  # bottleneck
])

decoder = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(6,)),
    keras.layers.Dense(n_features, activation='linear'),
])

autoencoder = keras.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=256, validation_split=0.2)

# Reconstruction error as anomaly score
reconstructed = autoencoder.predict(X_scaled)
recon_error = np.mean((X_scaled - reconstructed) ** 2, axis=1)
threshold = np.percentile(recon_error, 95)

# Embeddings for improved clustering
embeddings = encoder.predict(X_scaled)
# Feed embeddings into K-Means and compare silhouette vs. raw features
```

---

### 9. Data Storytelling (Sathwik) — Ch. XV

**Deliverables:**
- **Interactive Leaflet map:** Hazard zone clusters color-coded by severity, with popup details
- **Sankey diagram:** Episode escalation flows (event type sequences) using `networkD3` in R
- **Dashboard-style summary:** Key metrics, top association rules, model comparison table
- **Narrative:** "Here's what the data tells us about U.S. severe weather risk" — connecting predictions → clusters → rules

```r
# Example: Sankey diagram of episode flows
library(networkD3)

# Build flows from sequential patterns
flows <- episode_sequences %>%
  group_by(EPISODE_ID) %>%
  mutate(next_event = lead(EVENT_TYPE)) %>%
  filter(!is.na(next_event)) %>%
  count(EVENT_TYPE, next_event, sort = TRUE) %>%
  head(20)

sankeyNetwork(Links = flows_df, Nodes = nodes_df,
              Source = "source", Target = "target", Value = "n")
```

---

## Project Structure

```
storm-severity-mining/
├── data/
│   ├── raw/                    # Downloaded CSVs (gitignored)
│   └── processed/              # Cleaned RDS/parquet files
├── R/
│   ├── 01_download.R           # Data download script
│   ├── 02_preprocess.R         # Cleaning + feature engineering
│   ├── 03_eda.R                # EDA + density estimation
│   ├── 04_classification.R     # All 4 R classifiers
│   ├── 05_clustering.R         # DBSCAN + K-Means + Hierarchical
│   ├── 06_association.R        # Apriori + cSPADE
│   ├── 07_outliers.R           # Z-score + LOF
│   └── 08_storytelling.R       # Visualizations + Leaflet maps
├── python/
│   ├── neural_network.py       # Keras feedforward NN classifier
│   ├── autoencoder.py          # Autoencoder embeddings + anomaly
│   ├── isolation_forest.py     # Isolation Forest + LOF
│   └── utils.py                # Damage parsing, shared functions
├── output/
│   ├── figures/                # All plots (PNG/PDF)
│   ├── models/                 # Saved model objects
│   └── results/                # CSV tables of results
├── presentation/               # Slides for 2.5-min and 12-min talks
├── report/                     # Final report (Word/PDF)
└── README.md                   # This file
```

---

## Dependencies

### R Packages
```r
install.packages(c(
  "tidyverse", "caret", "dbscan", "cluster", "arules", "arulesViz",
  "arulesSequences", "leaflet", "sf", "spdep", "MASS", "smotefamily",
  "networkD3", "corrplot", "ggridges", "viridis", "e1071", "rpart",
  "randomForest", "class", "kernlab", "pROC", "dendextend"
))
```

### Python Packages
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

---

## Timeline

| Week | Dates | Tasks | Owner |
|------|-------|-------|-------|
| 1 | Feb 24 – Mar 2 | Data download, preprocessing, EDA, density estimation | Sathwik |
| 2 | Mar 3 – Mar 9 | Classification pipeline (DT, RF, kNN, SVM, NN) | Rishitha |
| 2 | Mar 3 | **2.5-min intro presentation** | All |
| 3 | Mar 10 – Mar 16 | Clustering + spatial mining + autoencoder embeddings | Viraat |
| 4 | Mar 17 – Mar 23 | Association/sequence mining + outlier detection | Anjani |
| 5 | Mar 24 – Mar 29 | Integration, storytelling, final report | All |
| — | Apr 2 or Apr 7 | **12-min final presentation** | All |

---

## References

1. NOAA NCEI. Storm Events Database – Bulk Data Download: https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
2. NOAA NCEI. Storm Events Database – Documentation: https://www.ncei.noaa.gov/stormevents/
3. NOAA NCEI. Storm Data FAQ: https://www.ncei.noaa.gov/stormevents/faq.jsp
4. Tan, Steinbach, Karpatne, Kumar. *Introduction to Data Mining* (2nd Ed.), Pearson.
