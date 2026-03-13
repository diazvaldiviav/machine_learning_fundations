# Assignment 3 - Dimensionality Reduction on MNIST

**Course:** CAI 2100C - Machine Learning Foundations
**Students:** Victor, Dudley, Natassia

## Video Presentation

[Watch the video explanation on Google Drive](https://drive.google.com/file/d/1OWns3f3h-D1rRrvUPWFQNrMWYrLyGedZ/view?usp=sharing)

## Assignment Requirements

The professor provided a Jupyter notebook (`[Notebook] Assignment 3.ipynb`) that demonstrated PCA and t-SNE on the MNIST dataset using **reduced subsets** (15,000 points for PCA, 1,000 for t-SNE). The assignment asked us to:

1. **Run the same analysis using all 42,000 data points** with various values of perplexity and iterations.
2. Ensure the Python code is **complete and working**.
3. Produce **correct results** matching the expected output.
4. Deliver a **5-10 minute presentation** explaining how everything works.

Reference blog for expected output: [Visualizing MNIST - colah's blog](http://colah.github.io/posts/2014-10-Visualizing-MNIST/)

## Solution Notebook: Cell-by-Cell Breakdown

Our solution is in `Assigment_3_answer.ipynb`. Below is a detailed explanation of each cell.

### Cell 1 - Import Required Modules

Imports `numpy`, `pandas`, `matplotlib`, and suppresses warnings. These are the core libraries needed for numerical operations, data handling, and plotting.

### Cell 2 - Load MNIST Data

Loads the full MNIST dataset from `mnist_train.csv` (42,000 images, 784 pixels each). Separates the digit labels from the pixel data into two variables: `l` (labels) and `d` (pixel features).

### Cell 3 - Use All 42K Data Points (Exercise 1)

**This is where our solution differs from the original notebook.** Instead of taking only 15,000 points (`l.head(15000)`), we use the entire dataset:
```python
labels = l    # All 42,000 labels
data = d      # All 42,000 images
```
Output shape: `(42000, 784)`

### Cell 4 - Data Standardization

Applies `StandardScaler` to transform all features to have mean=0 and standard deviation=1. This is critical because PCA is sensitive to feature scale — without standardization, pixels with larger values would dominate the principal components.

### Cell 5 - Compute Covariance Matrix

Computes the 784x784 covariance matrix using matrix multiplication (`A^T * A`). This matrix captures how each pair of pixel features varies together, which is the foundation for finding principal components.

### Cell 6 - Eigen Decomposition

Extracts the top 2 eigenvalues and eigenvectors using SciPy's `eigh()` function. These represent the two directions of maximum variance in the data. The eigenvectors are transposed from shape (784, 2) to (2, 784) for easier matrix multiplication.

> **Note:** We used `subset_by_index=[782,783]` instead of the deprecated `eigvals=(782,783)` parameter, which was removed in recent SciPy versions.

### Cell 7 - Project Data to 2D

Multiplies the eigenvectors (2, 784) by the transposed data (784, 42000) to project all 42,000 images from 784 dimensions down to just 2 dimensions. Result shape: `(2, 42000)`.

### Cell 8 - Build DataFrame for Plotting

Combines the 2D projected coordinates with the original labels into a pandas DataFrame with columns: `1st_principal`, `2nd_principal`, and `label`.

### Cell 9 - Plot Manual PCA (42K Points)

Creates a scatter plot using seaborn's `FacetGrid`, color-coded by digit label (0-9). This visualization shows how well PCA separates the different digits in 2D space using our manual implementation.

### Cell 10 - PCA Using Scikit-Learn

Repeats the PCA analysis using `sklearn.decomposition.PCA` with `n_components=2`. This is a simpler approach — Scikit-Learn handles the covariance matrix, eigenvalue extraction, and projection internally. Produces virtually identical results to the manual approach, validating our implementation.

### Cell 11 - Plot Scikit-Learn PCA (42K Points)

Plots the Scikit-Learn PCA results. The visualization is nearly identical to the manual PCA plot, confirming both methods are equivalent.

### Cell 12 - PCA Cumulative Variance Analysis

Runs PCA with all 784 components to analyze how much variance each component explains. Plots the cumulative explained variance curve. Key finding: **~200 components capture 90% of the total variance**, meaning we can reduce from 784 to ~200 dimensions and still retain most of the information.

### Cell 13 - t-SNE with Default Parameters (42K Points, perplexity=30)

**This is Exercise 2 — running t-SNE on all 42K points.** The original notebook only used 1,000 points. We run t-SNE with default parameters (`perplexity=30`, `max_iter=1000`) on all 42,000 data points. The result shows well-separated digit clusters, significantly better than PCA for visualization.

### Cell 14 - t-SNE with perplexity=50 (42K Points)

Runs t-SNE with `perplexity=50`. Higher perplexity means the algorithm considers more neighbors, capturing more global structure. With 42K points, the result is similar to perplexity=30 because there is enough data for robust cluster formation.

### Cell 15 - t-SNE with perplexity=50, max_iter=5000 (42K Points)

Increases iterations from 1,000 to 5,000 while keeping `perplexity=50`. More iterations allow the optimization to converge better, potentially producing cleaner and more defined clusters.

### Cell 16 - t-SNE with perplexity=2 (42K Points)

Runs t-SNE with a very low `perplexity=2`. This means each point only considers ~2 nearest neighbors, producing many small, fragmented clusters instead of well-defined groups. This demonstrates the importance of choosing an appropriate perplexity value.

## Key Findings

| Technique | Pros | Cons |
|---|---|---|
| **PCA** | Fast, deterministic, good for dimensionality reduction | Linear — significant overlap between digit clusters in 2D |
| **t-SNE** | Excellent cluster separation in 2D visualization | Slow (especially with 42K points), non-deterministic, sensitive to hyperparameters |

- **~200 PCA components** capture 90% of variance (out of 784 total dimensions)
- **t-SNE perplexity=30-50** produces the best cluster separation for the full dataset
- **t-SNE perplexity=2** produces fragmented, unusable clusters
- **More iterations** (5000 vs 1000) improve convergence and cluster quality

## How to Run

1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn scipy seaborn
   ```

2. The MNIST dataset (`mnist_train.csv`) is already included in this repository.

3. Open and run `Assigment_3_answer.ipynb` in Jupyter Notebook.

## Files

| File | Description |
|---|---|
| `[Notebook] Assignment 3.ipynb` | Original notebook provided by the professor (15K/1K points) |
| `Assigment_3_answer.ipynb` | Our solution using all 42K data points |
| `Assignment 3.pptx` | Assignment instructions from the professor |
| `Assignment-3-Dimensionality-Reduction-on-MNIST.pptx` | Our presentation slides |
| `mnist_train.csv` | MNIST dataset (42,000 handwritten digit images) |
