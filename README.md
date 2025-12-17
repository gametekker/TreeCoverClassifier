# Project Title: Predicting Tree Cover Type using Cartographic Data from the Roosevelt National Forest

## Introduction/Background

- This project aims to predict forest cover types using only cartographic variables, offering a low-cost alternative to remote sensing. The study will use a dataset from the Roosevelt National Forest in northern Colorado to build a predictive model.

- Dataset Description:
    - Features:
        - 54 features total, including 10 continuous variables (e.g., elevation, slope) and 44 binary variables for wilderness areas and soil types.
    - Data Format:
        - The data is in its raw, unscaled form
        - 7 different cover types
- Dataset Link: https://archive.ics.uci.edu/dataset/31/covertype

## Problem Definition

**Problem:** In our project, we aim to accurately classify cover types in the Roosevelt National Forest dataset using features such as elevation, slope, and soil type, despite heavy class imbalance.

**Motivation:** As outdoors lovers, we hope to leverage machine learning algorithms for practical conservation efforts, and build a model that can accurately identify forest cover types from cartographic data. We hope this will be a useful tool for monitoring ecosystem health, managing wildlife habitats and protecting the natural landscapes we love.

## Literature Review

When selecting forest type classification, we were motivated by data we expected as more consistent from nature rather than human-surveyed or generated. Classifying cover types with cartographic data has become a benchmark for supervised learning, blending ecological applications with technical ML evaluation.

Prior work by Chen, Liaw, and Breiman shows how Random Forests can address imbalance through balanced sampling and cost-sensitive approaches, specifically Balanced Random Forest and Weighted Random Forest [1]. This foundational paper shows why such methods will be crucial for our dataset, since Lodgepole Pine and Spruce/Fir (classes 1 and 2) make up almost 85% of the UCI dataset. Mellor et al. extend this, showing imbalance doesn't just lower accuracy but also weakens ensemble strength if not addressed [2]. Based on these findings, we will use resampling (e.g., SMOTE) and evaluate beyond accuracy.

Survey work by Chen et al. outlines three categories of approaches: data-level (under/over-sampling), algorithm-level (cost-sensitive learning), and hybrids. This is directly relevant: we have already applied SMOTE at data-level, and will apply cost weighting at algorithm-level and ensembles as hybrid as we progress our analysis. In their exploration of training data in random forest, they applied decision trees, Random Forests, and KNN to cover type classification. Using this hybrid ensemble, they reported over 97% accuracy [3]. Together, these studies demonstrate techniques we believe are well suited to this problem and aligned with the requirements and challenges of the dataset.

## Methods

We implemented a fully supervised machine learning pipeline testing three different algorithms. Out of the methods we outlined in our proposal, we tested Random Forest, Logistic Regression (as a baseline linear classifier), and a Multilayer Perceptron Neural Network. Random Forest, beyond just a name that fits well with our project, is uniquely well positioned to handle tabular ecological data. It is also particularly useful for handling a mix of binary and numeric features as we see in our dataset. In our preprocessing, we specifically tried to address the extreme class imbalance seen in the data by stratifying our different classes equally.

### Data Preprocessing

- **Actions**
    - Data was split into 70% training / 30% testing.
    - We stratified this data by cover type to maintain class representation that is proportional to the entire dataset for both the training and testing sets.
    - We generated two different, new datasets:
        - `preprocessed_test.csv.gz`: This dataset has the continuous variables standardized using zero-mean and unit variance, but no other preprocessing
        - `preprocessed_train.csv.gz`: This dataset has both the continuous variables standardized and SMOTE applied to address the class imbalance

- **Class Imbalance**
    - Our dataset has an extreme class imbalance before preprocessing.
    - The largest class (2 – Lodgepole Pine) has more than 10x as many features as the smallest class (4 – Cottonwood/Willow).

    | Class | Instances |
    |-------|-----------|
    | 1 – Spruce/Fir | 148,288 |
    | 2 – Lodgepole Pine | 198,310 |
    | 3 – Ponderosa Pine | 25,028 |
    | 4 – Cottonwood/Willow | 1,923 |
    | 5 – Aspen | 6,645 |
    | 6 – Douglas Fir | 12,156 |
    | 7 – Krummholz | 14,357 |

    - We applied SMOTE (Synthetic Minority Oversampling Technique), which generates new, synthetic datapoints from the minority classes to help balance our representation of the different classes.
    - After applying SMOTE, all classes had equal representation with 198,310 datapoints (the same as the largest class before processing)

- **Feature Standardization**
    - Since most Machine Learning algorithms are influenced by feature magnitude, we also wanted to make sure that all the data was represented on the same scale.
    - Units vary widely in the dataset: between miles, meters, etc. but also entirely different unit systems and contextual meanings (distance, height, degrees, etc).
    - We used Standard Scaler to standardize all 10 non-binary variables with a zero-mean and the same unit variance of 1.
    - This ensures that all the numeric variables contribute equally to the model.

### Machine Learning Algorithms

#### Random Forest

- **Choosing Random Forest**
    - We selected Random Forest as the supervised model to test first.
    - We chose Random Forest since:
        - The literature states how it particularly handles nonlinear interactions well, which we expect to be important given the large number of variables and their randomness in a true natural environment.
        - Random Forests is resilient to noise.
        - We wanted to measure which features were most important in predicting cover type, which this algorithm is well-suited for.

- **Implementation**
    - `n_estimators = 100`
        - We chose 100 trees as the sample size to train the model on. We choose this middle-of-the-road number to balance diversity in the features and runtime.
    - `random_state = 42`
        - For reproducibility
    - All other parameters in Random Forests were left at their default levels.

#### Logistic Regression

- **Choosing Logistic Regression**
    - We added Logistic Regression as a baseline linear classifier
    - We wanted to build on the idea from our proposal of testing simple, interpretable models as an element of the analysis
    - Since Linear Regression predicts continuous outputs while Logistic Regression handles multi-class classification directly, we determined it was the more appropriate linear method for our 7-class problem.
    - We used Logistic Regression in a number of ways:
        - To test whether the cover types are linearly separable within the 54-dimensions found in the cartographic feature space.
        - To provide a baseline comparison for more complex models like Random Forests and Neural Networks.
        - To extend our "simple" model beyond the originally proposed Linear Regression, but in a way that is mathematically compatible with classification.
    - It also acts as a reality check: if Logistic Regression performs poorly, it confirms that non-linear structure dominates ecological feature interactions.

- **Implementation**
    - Used `sklearn.linear_model.LogisticRegression` with:
        - `multi_class="multinomial"`
        - `solver="lbfgs"`
        - `max_iter=2000` for convergence on the large feature set
    - Standardized inputs were directly fed into the model since Logistic Regression is scale-sensitive
    - Training used the SMOTE-balanced training data
    - Evaluation used the original, untouched test set as in other versions

#### Multilayer Perceptron (MLP) Neural Network

- **Choosing an MLP NN**
    - From the proposal, Neural Networks were planned to explore deeper non-linear relationships beyond our other models.
    - We expanded beyond the sklearn `MLPClassifier` to also include a **custom PyTorch-built MLP**, giving finer control over the architecture and training loop.
    - Earlier experiments with a simpler neural network plateaued at **~85% accuracy**, while the tuned PyTorch model achieved **83.13% accuracy on the (standardized, untouched) test set** with a **weighted F1 of 0.8334** and **macro F1 of 0.7989**.
    - PyTorch’s additional functionality helped us:
        - Use automatic differentiation, GPU acceleration, and flexible tensor operations.
        - Tune learning rate, hidden layer size, and number of epochs more precisely.
    - The network automatically learned feature interactions between terrain, hydrology, sunlight, and distance features without manual feature engineering—useful in an ecological system where relationships are complex and potentially hierarchical.
    - This MLP provided a test of whether **increasing representational power** could improve performance beyond the already strong Random Forest baseline.
    - **Important note:** this is an MLP applied to **tabular data**, a setting where tree-based models (e.g., Random Forests, Gradient Boosted Trees) are often better suited. This likely explains why the MLP did not clearly outperform the Random Forest, despite comparable accuracy.

- **Implementation**
    - Built a custom PyTorch MLP with:
        - Input layer for all **54 standardized features**
        - **Two hidden layers** with ReLU activation
        - Output layer with **7 logits** (one per cover type)
    - **Loss function:** `CrossEntropyLoss`
    - **Optimizer:** `Adam`
    - **Training data:** standardized features with **SMOTE-balanced** classes
    - **Evaluation:** performed on the untouched standardized test set
    - **Tuned components:**
        - Learning rate  
        - Hidden layer sizes  
        - Number of epochs

## Results and Discussion

After implementing our preprocessing (feature scaling and SMOTE for class imbalance) and training our classification models, we used the unmodified test set to test our accuracy and assess the results.

### Random Forest Results

- **Accuracy: 0.9558, Precision: 0.9558, Recall: 0.9558**
- The model achieved an overall accuracy of 0.9558, meaning it correctly classified roughly 95.6% of all test samples.
- This result significantly exceeds our original project goal of ~80% accuracy, confirming using SMOTE and scaling along with the Random Forest architecture was an effective strategy.
- **Macro Precision = 0.9253, Macro Recall = 0.9432**
    - The Macro values (which treat every class equally regardless of size) show that using SMOTE to balance the classes does improve metrics, but is not necessarily critical to the success of the Random Forests algorithm in classifying this dataset.

### Logistic Regression Results

- **Accuracy: 0.6191, Precision: 0.7064, Recall: 0.6191, F1-Score: 0.6440**
- The model achieved an accuracy of 0.6191, meaning it correctly classified roughly 62% of all test samples.
    - This falls well below our Random Forest performance and confirms that a purely linear model cannot capture the ecological complexity present in the data.
- **Macro Precision = 0.48, Macro Recall = 0.70**
- These macro values, which treat each class equally regardless of size, indicate large variation in class performance. High recall with low precision shows the model is frequently overpredicting minority classes.
    - Reasonable performance on large classes (1 and 2), but not competitive with Random Forest
    - Extremely low precision on minority classes, such as Class 5 (precision 0.12)
    - High recall but low precision for several classes (4, 5, 7), meaning many incorrect labels
    - Indicates the model cannot form clean boundaries in the 54-dimensional feature space
- **Confusion Matrix Results**
    - Major misclassifications between Spruce/Fir (1) and Lodgepole Pine (2)
    - Severe overprediction of minority classes (especially 4, 5, and 7)
    - Demonstrates that linear separability assumptions do not hold for this ecological dataset

### MLP Neural Network (PyTorch) Results

- **Final test performance:**
        - Accuracy: **0.8313**
        - Precision (weighted): **0.8446**
        - Recall (weighted): **0.8313**
        - F1 Score (weighted): **0.8334**
        - Macro Precision: **0.7523**
        - Macro Recall: **0.8772**
        - Macro F1: **0.7989**
    - **Per-class performance (test set):**

        | Class | Precision | Recall | F1-score | Support |
        |-------|-----------|--------|----------|---------|
        | 1     | 0.79      | 0.88   | 0.83     | 63,552  |
        | 2     | 0.90      | 0.78   | 0.84     | 84,991  |
        | 3     | 0.87      | 0.85   | 0.86     | 10,726  |
        | 4     | 0.68      | 0.93   | 0.78     | 824     |
        | 5     | 0.48      | 0.91   | 0.63     | 2,848   |
        | 6     | 0.65      | 0.90   | 0.76     | 5,210   |
        | 7     | 0.88      | 0.90   | 0.89     | 6,153   |

    - The model performs strongly on the majority classes (1, 2, 3, 7) and, with SMOTE, attains high recall on minority classes (4, 5, 6), although precision is lower for some of these rare classes (especially class 5).

- **Training Dynamics (Loss Curve)**

    The training loss over epochs shows an overall downward trend with some late-epoch noise, suggesting that while the model is learning, there may be some instability due to the learning rate, SMOTE balancing, or limited suitability of MLPs for tabular data:

    ![Training Loss Over Epochs](mlp_loss_curve.png)

## Visualizations (Our Best Model: Random Forest)

### Confusion Matrix

![Confusion Matrix](assets/confusion-matrix.png)

- The confusion matrix showed a strong diagonal trend, confirming that the model did a great job at predicting the actual cover type of each datapoint based on the features.
- The most frequent class confusions were between 1 and 2: this makes sense, as both trees have similar characteristics and tend to occupy the same high-elevation zones within the Roosevelt National Forest sample from which this was taken.
- Krummholz trees had almost perfect class identification by the model, likely due to their unique ecological conditions and characteristic.

### Feature Importance Chart

![Feature Importance](assets/feature-importance.png)

- Random Forest has a built in attribute *feature importance*, which uses MDI (mean decrease in impurity) to measure which features influence classification the most
- This measures how much each feature reduces classification error when it is included and then uses that information to rank importance
- Elevation was far and away the most important feature, showing the importance of elevation in mountain environments in determining what kind of forest cover grows there.
- Overall, the combination of the physical terrain (Elevation and Slope), distance to nearby water, and sunlight-related variables (Hillshade, etc.) were the most important in predicting tree cover.
- Soil type and the wilderness area location were also relevant factors in the top 15 out of the 50+ features.
- This shows that the model is deriving real ecological insights and providing information about what factors in nature actually have an impact on tree cover!

## Model Comparisons

### Overall Performance Comparison

| Model | Accuracy | Key Observations |
|-------|----------|------------------|
| **Random Forest** | 95.6% | Best overall accuracy and most stable performance |
| **MLP (PyTorch)** | 74.5% | Strong but below RF; requires tuning; harder to interpret |
| **Logistic Regression** | 61.9% | Far below both; confirms the problem is non-linear |

### Overall Interpretability

- **Random Forest**
    - Clear feature importance
    - Ecologically meaningful insights
- **Logistic Regression**
    - Coefficients interpretable, but model fit is poor
    - This makes interpretation unreliable
- **MLP**
    - Least interpretable (weights are very non-transparent)

### Complexity & Training Cost

- **Random Forest**
    - Moderate compute cost
    - Minimal tuning required
- **Logistic Regression**
    - Fastest to train
    - Poor real-world performance
- **MLP**
    - Highest training cost
    - Sensitive to hyperparameters

### Overall Takeaways

- **Random Forest remains the best-performing and most practical model:**
    - High accuracy
    - Low tuning cost
    - Ecological interpretability
- The MLP is competitive in terms of usage but provides no meaningful performance gain despite a high level of added complexity and compute cost.
- Logistic Regression reinforces the conclusion that non-linear approaches are essential for cartographic forest cover classification due to its poor performance.

## Next Steps

### Model Optimization and Tuning
- Tune Random Forest hyperparameters (`max_depth`, `max_features`, `n_estimators`)
- Add cost-sensitive RF (e.g., `class_weight="balanced_subsample"`)
- Improve PyTorch MLP using dropout and learning rate scheduling
- Compare PyTorch MLP to sklearn's easier to tune `MLPClassifier`
- Explore early stopping and longer training epochs for stability
- Evaluate Logistic Regression with regularization strength adjustments
- Re-run all models with fixed random seeds for full reproducibility

### Additional Models & Preprocessing Variants
- Test Gradient Boosting methods (XGBoost, LightGBM, CatBoost)
- Compare SMOTE vs class weighting vs undersampling
- Try PCA or feature reduction to help neural networks generalize
- Add a Balanced Random Forest or Cost-Sensitive Random Forest
- Evaluate performance without SMOTE to measure its true impact
- Add permutation importance for the MLP to compare interpretability
- Use polynomial feature interactions with Logistic Regression as a linear–nonlinear hybrid baseline

### Clustering and Potential Unsupervised Methods
- Apply K-Means on the standardized features
- Run Gaussian Mixture Models to test cluster separability
- Visualize clusters in 2D PCA space
- Compare cluster boundaries with confusion matrix errors
- Examine whether minority classes (e.g., 3, 4, 5) form distinct clusters
- Assess if classes 1 & 2 occupy overlapping feature regions
- Use silhouette scores to evaluate cluster quality

### Ecological Insights
- Break down precision, recall, and F1 per class
- Create partial dependence plots for top RF features
- Visualize how elevation and hydrology drive predictions
- Analyze soil-type contributions to specific classes
- Compare model errors to known ecological overlap zones
- Use feature importance to interpret ecological drivers
- Link confusion patterns to real-world forest distributions

---

**We would like to be considered for the Outstanding Project award.**

## Name / Personal Contributions

| Name | Personal Contributions |
|------|------------------------|
| Yizhen (Eric) Jia | Methods, Model Implementation |
| Mehul Rao | Methods, Results (Visualizations), Report |
| Zachary Walpole | Literature Review, Preprocessing, Results (Viz + Next Steps), Report |
| Jack Hughes | Preprocessing, Results (Analysis) |
| Riley Corzine | Preprocessing, Methods, Model Implementation |

## Citations

[1] C. Chen, A. Liaw, and L. Breiman, Using Random Forest to Learn Imbalanced Data. Statistics Department, University of California, Berkeley, 2004.

[2] A. Mellor, S. Boukir, A. Haywood, and S. Jones, "Exploring issues of training data imbalance and mislabelling on random forest performance for large area land cover classification using the ensemble margin," ISPRS Journal of Photogrammetry and Remote Sensing, vol. 105, pp. 155–168, Jul. 2015. doi:10.1016/j.isprsjprs.2015.03.014

[3] W. Chen, K. Yang, Z. Yu, Y. Shi, and C. L. Chen, "A survey on Imbalanced Learning: Latest Research, applications and Future Directions," Artificial Intelligence Review, vol. 57, no. 6, May 2024. doi:10.1007/s10462-024-10759-6
