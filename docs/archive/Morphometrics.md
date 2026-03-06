# Building Morphological Indicators

[cite_start]This study formally defines 25 morphological indicators used for urban analysis[cite: 2].

## 1. Key Definitions and Symbols
[cite_start]Let the following symbols denote the fundamental properties of the building set[cite: 3]:
* [cite_start]**$P$**: A building polygon[cite: 3].
* [cite_start]**$A$**: Area of the polygon[cite: 3].
* [cite_start]**$\partial P$**: The boundary of the polygon[cite: 11].
* [cite_start]**$L_{major}, L_{minor}$**: Lengths of the major and minor axes[cite: 3].
* [cite_start]**$N$**: Total number of buildings[cite: 3].

---

## 2. Geometric Properties
[cite_start]These properties define the basic spatial dimensions of a building[cite: 5].

| Feature | Mathematical Expression | Definition |
| :--- | :--- | :--- |
| **Building Area** | $A=area(P)$ | [cite_start]The area of polygon $P$[cite: 7, 8]. |
| **Building Perimeter** | $P=length(\partial P)$ | [cite_start]The length of the boundary of polygon $P$[cite: 10, 11]. |
| **Longest Axis Length** | $L_{max}=max(L_{major},L_{minor})$ | [cite_start]The maximum of the major and minor axis lengths[cite: 13, 14]. |

---

## 3. Compactness and Shape Metrics
[cite_start]These indicators evaluate the complexity and efficiency of a building's shape[cite: 15].



* [cite_start]**Shape Index ($SI$):** Measures compactness based on the ratio of perimeter to the square root of area[cite: 17, 18].
  $$SI=\frac{P}{\sqrt{A}}$$
* [cite_start]**Compactness Weighted Axis ($CWA$):** Multiplies elongation by standard compactness[cite: 19, 21].
  $$CWA=\frac{L_{major}}{L_{minor}}\cdot\frac{4\pi A}{P^{2}}$$
* [cite_start]**Convexity ($C$):** The ratio of the building area to its convex hull area ($A_{convex}$)[cite: 23, 24].
  $$C=\frac{A}{A_{convex}}$$
* [cite_start]**Facade Ratio ($FR$):** The ratio of total façade length ($L_{facade}$) to the total perimeter ($P$)[cite: 25, 27].
  $$FR=\frac{L_{facade}}{P}$$
* [cite_start]**Shared Walls ($SW$):** The sum of the boundary lengths ($l_{shared}$) shared with adjacent buildings[cite: 36, 37, 39].
* [cite_start]**Perimeter Wall ($PW$):** The external perimeter not shared with any neighbors[cite: 40, 41, 43].
  $$PW=P-SW$$
* [cite_start]**Number of Corners ($NC$):** A count of the total number of polygon vertices[cite: 44, 45, 47].
* [cite_start]**Equivalent Rectangular Index ($ERI$):** The ratio of the building area to its minimum bounding rectangle ($A_{rect}$)[cite: 48, 49, 50].
* [cite_start]**Rectangularity ($R$):** The building area relative to the rectangle defined by principal axes[cite: 51, 52, 53].
  $$R=\frac{A}{L_{major}\times L_{minor}}$$
* [cite_start]**Squareness ($SQ$):** Evaluates how close interior angles ($\theta_{i}$) are to 90 degrees[cite: 54, 57].
  $$SQ=1-\frac{1}{n}\sum_{i=1}^{n}\frac{|\theta_{i}-90^{\circ}|}{90^{\circ}}$$
* [cite_start]**Square Compactness ($SC$):** The building area normalized by the square of the longest axis[cite: 55, 57].
  $$SC=\frac{A}{(max(L_{major},L_{minor}))^{2}}$$
* [cite_start]**Elongation ($E$):** The ratio between the major and minor axis lengths[cite: 56, 57].
  $$E=\frac{L_{major}}{L_{minor}}$$
* [cite_start]**Compactness-weighted Axis of Each Tessellation ($CWT$):** Uses the tessellation axis ($L_{axis}$) and cell area ($A_{tess}$)[cite: 58, 59, 60].
  $$CWT=\frac{L_{axis}}{\sqrt{A_{tess}}}$$

---

## 4. Spatial Distribution Metrics
[cite_start]Metrics used to describe the arrangement and density of buildings within a site[cite: 68].

* [cite_start]**Mean Distance Between Buildings ($d_{mean}$):** The average centroid distance ($d(i,j)$) between all building pairs[cite: 69, 70, 71].
  $$d_{mean}=\frac{1}{N^{2}}\sum_{i}\sum_{j}d(i,j)$$
* [cite_start]**Mean Interbuilding Distance ($d_{NN}$):** The average distance to each building's nearest neighbor[cite: 72, 75].
  $$d_{NN}=\frac{1}{N}\sum_{i}min_{j}d(i,j)$$
* [cite_start]**Building Adjacency ($ADJ$):** Defined by the number of other polygons that intersect with polygon $i$[cite: 76, 77, 79].
* [cite_start]**Covered Area Ratio ($CAR$):** The ratio of the sum of all building areas ($A_{i}$) to the total site area ($A_{site}$)[cite: 80, 81, 83].
  $$CAR=\frac{\sum_{i}A_{i}}{A_{site}}$$

---

## 5. Tessellation-related Metrics
[cite_start]Derived from the Voronoi or Thiessen polygon ($V_{i}$) generated for each building[cite: 84, 88].



* [cite_start]**Tessellation Areas ($A_{v}$):** The area of the Voronoi polygon ($V_{i}$) associated with building $i$[cite: 85, 86, 88].
* [cite_start]**Cell Alignment ($CA$):** Measures the consistency of cell orientation ($\theta_{i}$) compared to the mean orientation ($\overline{\theta}$)[cite: 89, 90, 92].
  $$CA=\frac{1}{N}\sum_{i}cos(\theta_{i}-\overline{\theta})$$
* [cite_start]**Tessellation Number of Neighbors ($TN$):** Count of buildings that share an edge in the tessellation[cite: 93, 94, 96].

---

## 6. Topological Features
[cite_start]Advanced metrics describing complexity and weighted spatial relationships[cite: 97].

* [cite_start]**Fractal Dimension ($FD$):** Relates perimeter to area to measure boundary complexity[cite: 98, 99].
  $$FD=\frac{2~ln(P)}{ln(A)}$$
* [cite_start]**Average Weighted Distance ($d_{w}$):** Centroid distances ($d(i,j)$) adjusted by a weight ($w_{ij}$) based on adjacency or size[cite: 102, 103, 104].
  $$d_{w}=\frac{1}{N}\sum_{i}\sum_{j}w_{ij}d(i,j)$$