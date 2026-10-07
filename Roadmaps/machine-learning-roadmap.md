# Machine Learning Roadmap: A Step-by-Step Guide from Python Basics to Deep Learning in 2026

A free, project-driven learning path for people who want to build, evaluate and understand machine learning (ML) models. It follows the order in which the ideas depend on each other: what the ML engineer role is, the mathematics behind models, programming and data tooling, collecting and cleaning data, classical machine learning (supervised, unsupervised and reinforcement), model evaluation, deep learning, and a first look at explainability and natural language processing. Every stage has a "why it matters" note, short explanations of each topic, a hands-on exercise and a self-check list. The guide ends with three capstone projects, a week-by-week study plan and a checklist you can tick off.

**Who it is for.** Beginners and career changers who want to understand how ML models work rather than only call a library: students, analysts, software developers and engineers moving into data and ML work. You do not need a degree, a GPU or paid tools. A laptop with Python and a free notebook environment (a local Jupyter install or a free hosted notebook service) is enough for almost every exercise; the deep learning stage runs on CPU at small scale, and a free or rented GPU only makes it faster.

**Prerequisites.** Basic Python: variables, functions, loops, lists and dictionaries, reading files, and installing packages in a virtual environment. If that is not yet true, start with the [Prerequisite Python roadmap](#prerequisite-python-roadmap) subsection first. No prior math beyond school algebra is assumed; stage 2 rebuilds what you need.

**What you will be able to do at the end.**

- Explain, in plain language and with small formulas, how common models learn: linear and logistic regression, trees and ensembles, clustering, neural networks, convolutional and recurrent networks, and transformers.
- Collect data from files, databases and APIs, clean it, engineer features and avoid data leakage.
- Run a complete scikit-learn workflow: split, preprocess, select a model, tune, evaluate and predict, using pipelines and cross-validation.
- Pick evaluation metrics that match the cost of mistakes, read a confusion matrix, and compare models fairly.
- Build and train small neural networks in PyTorch or Keras, including CNNs for images and sequence or attention models for text.
- Explain model decisions with basic explainability tools, and apply core natural language processing (NLP) steps.
- Present three portfolio projects with honest metrics and limitations.

**Time estimate.** About 24 to 30 weeks at 8 to 10 hours per week (roughly 200 to 300 hours). The study plan below uses 30 weeks including the three capstone projects. If you already code comfortably and remember school statistics, expect closer to 24 weeks. Time per stage assumes every topic is read and every exercise done.

## Originality and review status

This is an independently written, original learning guide. The text, examples, diagrams and structure were written for this repository from the author's own knowledge, official documentation and primary sources.

Topic coverage follows the community roadmap at [https://roadmap.sh/machine-learning](https://roadmap.sh/machine-learning). That site is linked for reference only: none of its content is copied or adapted here, and this guide is not affiliated with or endorsed by it.

**Last reviewed: October 2026.**

Libraries, APIs and best practices change. Package versions, default arguments, framework status and cloud offerings can shift within months. This guide prefers durable concepts, keeps code short, and marks perishable statements with "(as of Oct 2026)". Always confirm details in the official documentation of the library before you depend on them. Nothing here is legal, medical or financial advice. Any dataset you use must be used according to its licence and privacy terms.

## Table of contents

- [The path at a glance](#the-path-at-a-glance)
- [1. Orientation: Prerequisites and the ML Engineer Role](#1-orientation-prerequisites-and-the-ml-engineer-role)
- [2. Mathematical Foundations](#2-mathematical-foundations)
- [3. Programming Fundamentals](#3-programming-fundamentals)
- [4. Data Collection](#4-data-collection)
- [5. Data Cleaning and Preprocessing Techniques](#5-data-cleaning-and-preprocessing-techniques)
- [6. Machine Learning Basics and the Scikit-learn Workflow](#6-machine-learning-basics-and-the-scikit-learn-workflow)
- [7. Supervised Learning](#7-supervised-learning)
- [8. Unsupervised Learning](#8-unsupervised-learning)
- [9. Reinforcement Learning](#9-reinforcement-learning)
- [10. Model Evaluation](#10-model-evaluation)
- [11. Deep Learning](#11-deep-learning)
- [12. Advanced Concepts in ML](#12-advanced-concepts-in-ml)
- [13. Next Steps](#13-next-steps)
- [Capstone projects](#capstone-projects)
- [Suggested weekly study plan](#suggested-weekly-study-plan)
- [Related guides in this repository](#related-guides-in-this-repository)
- [Coverage checklist](#coverage-checklist)

## The path at a glance

Stages 1 to 12 are the learning path; stage 13 points to what comes next. Solid arrows are hard dependencies. The dotted arrow is a soft link: deep reinforcement learning in stage 9 is easier once you know neural networks from stage 11. Many learners also read stage 10 (evaluation) right after stage 7, because every later stage uses its metrics.

```mermaid
flowchart TD
    subgraph FND["Foundations"]
        S1["1 Orientation"] --> S2["2 Mathematical foundations"]
        S1 --> S3["3 Programming fundamentals"]
    end
    subgraph DAT["Data"]
        S4["4 Data collection"] --> S5["5 Cleaning and preprocessing"]
    end
    subgraph CML["Classical machine learning"]
        S6["6 ML basics and scikit-learn"] --> S7["7 Supervised learning"]
        S6 --> S8["8 Unsupervised learning"]
        S6 --> S9["9 Reinforcement learning"]
        S7 --> S10["10 Model evaluation"]
        S8 --> S10
    end
    subgraph DLP["Deep learning and beyond"]
        S11["11 Deep learning"] --> S12["12 Advanced: XAI and NLP"]
    end
    S13(["13 Next steps"])

    S3 --> S4
    S2 --> S6
    S5 --> S6
    S10 --> S11
    S9 -.->|"deep RL needs it"| S11
    S12 --> S13
```

| Stage | What you learn | Time | Key outcome |
|-------|----------------|------|-------------|
| 1. Orientation | The Python prerequisite, related roadmaps, what an ML engineer does, how the role differs from an AI engineer, skills and responsibilities | About 0.5 week | You can describe the job, pick your next roadmap, and set up a working environment |
| 2. Mathematical Foundations | Linear algebra, calculus, probability and statistics | 3 weeks | You can read the formulas in ML tutorials and run a gradient descent step by hand |
| 3. Programming Fundamentals | Python syntax, object oriented programming, NumPy, Pandas, Matplotlib, Seaborn | 2 weeks | You can load, reshape, summarize and plot a dataset with confidence |
| 4. Data Collection | Databases, the internet, APIs, mobile and IoT sources; CSV, Excel, JSON, Parquet and other formats | 1 week | You can pull data from several sources into one DataFrame and choose a storage format |
| 5. Data Cleaning and Preprocessing | Cleaning, feature engineering, scaling, dimensionality reduction, feature selection | 2 weeks | You can turn raw data into a leak-free numeric matrix with a reusable pipeline |
| 6. Machine Learning Basics | What ML is, the five types of ML, the scikit-learn workflow | 1 week | You can train, tune and use a first model end to end |
| 7. Supervised Learning | Classification and regression algorithms, regularization | 3 weeks | You can choose, tune and explain a classical model for a tabular problem |
| 8. Unsupervised Learning | PCA, autoencoders, four clustering families | 2 weeks | You can find structure in unlabeled data and judge whether it is meaningful |
| 9. Reinforcement Learning | Q-learning, deep Q networks, policy gradient, actor-critic | 2 weeks | You can describe the agent-environment loop and train a tabular agent |
| 10. Model Evaluation | Metrics, confusion matrix, cross-validation and leave-one-out | 1 week | You can pick the right metric and validate a model without fooling yourself |
| 11. Deep Learning | Neural network basics, libraries, CNNs, RNNs, attention, autoencoders, GANs | 7 weeks | You can train and diagnose neural networks in PyTorch or Keras |
| 12. Advanced Concepts in ML | Explainable AI, natural language processing | 3 weeks | You can explain a model's behavior and process text for modeling |
| 13. Next Steps | The AI and Data Scientist, MLOps, AI Engineer and AI Agents roadmaps | Ongoing | You know which path to take after this one |

## 1. Orientation: Prerequisites and the ML Engineer Role

**Why it matters.** Knowing what the job is, what you must already know, and where this roadmap fits among its neighbours helps you decide how deep to go in each later stage. This stage is short, but a clear picture of the role keeps you from over-investing in research-level theory or, on the other side, from stopping at "I can call `fit()`".

### Prerequisite Python roadmap

Almost all ML work is done in Python, so basic Python is the entry ticket. You should be able to write functions, loop over lists and dictionaries, read and write files, handle errors and install packages inside a virtual environment. If that is not yet comfortable, spend two to four weeks on a beginner Python course first: the official [Python tutorial](https://docs.python.org/3/tutorial/) is free, and [section 01 of the AI Engineer roadmap](../AI-Engineer-Roadmap/01-prerequisites-and-dev-foundations.md) in this repository lists what to cover. Stage 3 recaps only the parts that matter for ML; it does not teach programming from zero. The common mistake is skipping this step and then fighting syntax errors and ML concepts at the same time, which makes every bug look mysterious. A quick self-test: write a function that reads a CSV file with the standard `csv` module, counts rows per category in a dictionary and prints the three largest categories.

### Related roadmaps

This roadmap is one of several neighbouring paths, and the boundaries are blurry. Read job descriptions rather than titles.

| Roadmap | Focus | Take it when | In this repository |
|---------|-------|--------------|--------------------|
| AI Engineer | Building products on pre-trained models: LLM APIs, retrieval, agents, evals | You want to ship AI features without training models from scratch | [AI Engineer Roadmap](../AI-Engineer-Roadmap/README.md) |
| MLOps | Packaging, deploying, monitoring and retraining models reliably | Your models work in notebooks and now must run in production | [MLOps Roadmap](mlops-roadmap.md) |
| AI and Data Scientist | Statistics, experiments, analysis and communicating decisions from data | You enjoy questions, hypotheses and storytelling with data | [Machine Learning Beginner Roadmap](../Machine%20Learning%20Beginner%20Roadmap_.md) |

### What is an ML Engineer?

A machine learning engineer builds systems that learn from data and keep working after the demo: they frame the business problem as a prediction task, prepare data, train and evaluate models, and package them so that other software can use them. The work sits between data science, which focuses on analysis and experiments, and software engineering, which focuses on reliable code and services. Day to day, most of the time goes to data quality, pipelines, evaluation and debugging rather than to inventing new algorithms. A common misunderstanding is that the role is mostly about picking fancy models; in practice a clean dataset, a sound validation scheme and a simple, well-tuned model beat a complex model trained on messy data. Job titles and expectations vary a lot between companies (as of Oct 2026), so treat this description as a typical picture, not a rule.

### ML Engineer vs AI Engineer

Both roles ship intelligent features, but they start from different places. An ML engineer typically trains or adapts models on the organisation's own data and owns the training and serving pipeline. An AI engineer typically builds on a pre-trained model, often a large language model reached through an API, and spends more effort on prompts, retrieval, tool use and evaluation. The skills overlap heavily: software engineering, data handling and careful evaluation matter in both, and many people move between the two.

| Aspect | ML Engineer | AI Engineer |
|--------|-------------|-------------|
| Starting point | Your own data and a model you train or fine-tune | A pre-trained foundation model used through an API or run as open weights |
| Typical work | Feature pipelines, training code, validation, model serving, monitoring | Prompting, retrieval, agents, structured outputs, evals, safety controls |
| Math depth | Moderate to strong (linear algebra, optimization, statistics) | Light to moderate |
| Main risk | Data leakage, poor validation, drift | Hallucination, prompt injection, cost and latency |

### Skills and Responsibilities

An ML engineer needs a mix of skills: programming in Python and sometimes SQL, enough math to reason about models, data wrangling, model training and evaluation, software engineering habits (version control, tests, code review) and communication with non-technical stakeholders. The responsibilities usually follow the life of a model, from understanding the problem to running it in production. Beginners often overlook the last items on the list, yet they decide whether a model is ever used.

- Translate a business question into a measurable prediction task and a success metric.
- Find, collect, clean and version the data, and document its limits.
- Build baselines first, then train, tune and compare models with honest validation.
- Package models and pipelines as reproducible code with tests.
- Deploy or hand over to a serving platform and monitor quality, drift and cost.
- Explain results and risks to people who will rely on them.

**Try it.** Write a half-page "job card" for the role you want in 12 months (ML engineer, AI engineer, data scientist or MLOps engineer). Collect three real job descriptions, highlight the skills that appear in at least two of them, and mark which of those skills each stage of this roadmap covers.

**Self-check.**

- [ ] I can say what a machine learning engineer builds and how that differs from analysis-only work.
- [ ] I can explain the main differences between an ML engineer and an AI engineer.
- [ ] I can list five responsibilities across a model's life cycle, from problem framing to monitoring.
- [ ] I have passed the Python self-test above or planned a beginner course to pass it.
- [ ] I have chosen my next roadmap after this one and can say why.

## 2. Mathematical Foundations

**Why it matters.** An ML model is a function with many adjustable numbers, and training is a search for the numbers that make its mistakes small. Linear algebra describes the data and the function, calculus describes how to improve it, probability describes uncertainty, and statistics tells you whether a result can be trusted. You do not need proofs. You need working intuition for what each idea does, and the ability to read the formulas in documentation and papers without panic.

### Linear Algebra

Linear algebra is the language of data: a dataset is a matrix, a prediction is usually a matrix product, and many algorithms are decompositions of a matrix. The five topics below build on each other.

**Scalars, Vectors, Tensors.** A scalar is a single number, a vector is an ordered list of numbers (the features of one sample, or a word embedding), a matrix is a two-dimensional grid (samples by features) and a tensor generalizes this to any number of axes. A batch of colour images, for example, is a four-dimensional tensor: batch, height, width and channels. The skills to practise are reading shapes and using the dot product, which multiplies matching entries and sums them to measure how aligned two vectors are. Vector length (the norm) and distance appear in nearest neighbours, regularization and loss functions. The classic pitfall is shape confusion: in NumPy a vector of shape `(n,)` is not the same as a column of shape `(n, 1)`, and silent broadcasting can hide the bug.

**Matrix and Matrix Operations.** Matrix multiplication `A @ B` requires the inner sizes to match, so `(m, k) @ (k, n)` gives `(m, n)`. It is the workhorse of ML: the prediction of a linear model is `X @ w`, and one layer of a neural network is a matrix product plus a bias. You also need the transpose, the identity matrix, and the difference between element-wise multiplication and matrix multiplication. Matrix multiplication is not commutative, so `A @ B` and `B @ A` generally differ. The most common bug is writing `*` when you meant `@`, which still runs in NumPy when the shapes allow it.

**Determinants, inverse of a Matrix.** The determinant of a square matrix measures how it scales volume, and a determinant of zero means the matrix is singular: it squashes space and has no inverse. The inverse undoes the transformation, and it appears in the normal equation of linear regression, `w = (X^T X)^-1 X^T y`, and in the covariance matrices of Gaussian models. In practice you rarely compute an inverse explicitly. Use `np.linalg.solve` or a least-squares routine, which are faster and more numerically stable. Near-singular matrices arise when features are strongly correlated (multicollinearity), and regularization (stage 7) is the usual cure.

**Eigenvalues, Diagonalization.** An eigenvector of a matrix `A` is a direction that `A` only stretches, never turns: `A v = lambda v`, where `lambda` is the eigenvalue. A matrix with enough independent eigenvectors can be diagonalized, `A = P D P^-1`, which makes powers of the matrix cheap and shows its behaviour along its own axes. The eigenvectors of a covariance matrix are the principal components used by PCA (stage 8), and the eigenvalues tell how much variance each one carries. Symmetric matrices such as covariance matrices always have real eigenvalues and orthogonal eigenvectors, which is why PCA behaves well. A non-symmetric matrix can have complex eigenvalues, and that surprises many beginners.

**Singular Value Decomposition.** SVD factors any matrix, square or not, as `A = U S V^T`, where `U` and `V` have orthonormal columns and `S` holds non-negative singular values in decreasing order. Keeping only the top `k` singular values gives the best rank-`k` approximation of `A`, which is the idea behind image compression, latent semantic analysis, recommender-system factorization and PCA. The link to the previous topic is that the right singular vectors of `A` are the eigenvectors of `A^T A`. A frequent pitfall is forgetting to centre the columns of the data before using SVD for PCA, which makes the first component describe the mean instead of the variation.

```python
import numpy as np

A = np.array([[4.0, 2.0], [1.0, 3.0]])
b = np.array([2.0, 5.0])

print(A @ b)                           # matrix-vector product
print(np.linalg.det(A))                # about 10.0, non-zero, so A has an inverse
print(np.linalg.solve(A, b))           # solves A x = b without forming the inverse
print(np.linalg.eig(A)[0])             # eigenvalues 5 and 2 (order may vary)

U, s, Vt = np.linalg.svd(A)            # A = U @ diag(s) @ Vt
rank1 = s[0] * np.outer(U[:, 0], Vt[0])  # best rank-1 approximation of A
print(np.round(rank1, 2))
```

### Calculus

Calculus answers one question that matters to ML: if I change a parameter slightly, how does the error change? The four topics move from single-variable slopes to the multi-variable tools used by training algorithms.

**Derivatives, Partial Derivatives.** A derivative is the slope of a function at a point: the change in the output per tiny change in the input. When a function has many inputs, a partial derivative measures the slope along one input while the others are held fixed. A loss function depends on all model parameters, so its partial derivatives say which way to nudge each parameter to reduce the error. A handy test for your own derivative code is the finite-difference check, `(f(x + h) - f(x - h)) / (2h)` for a small `h`. Take `h` too small, such as `1e-12`, and floating-point rounding dominates the result.

**Chain rule of derivation.** If `y = f(g(x))`, then `dy/dx = f'(g(x)) * g'(x)`: the slope of a composition is the product of the slopes of its parts. A neural network is a long composition of simple functions, and backpropagation (stage 11) is the chain rule applied from the loss backwards while reusing intermediate results. Libraries such as PyTorch and TensorFlow do this automatically, but knowing the rule helps you understand vanishing and exploding gradients: a product of many numbers smaller than one shrinks towards zero, and a product of many numbers larger than one explodes.

**Gradient, Jacobian, Hessian.** The gradient is the vector of partial derivatives of a scalar function; it points in the direction of steepest increase, so training steps the other way: `theta <- theta - lr * gradient`. The Jacobian is the matrix of all first partial derivatives of a function with several outputs, and backpropagation multiplies Jacobians layer by layer. The Hessian is the matrix of second derivatives and describes curvature; second-order optimizers use it, but deep learning mostly avoids it because for `n` parameters it has `n^2` entries. A learning rate that is too large makes the loss bounce or diverge, one that is too small crawls, and non-convex losses add saddle points and local minima.

**Discrete Mathematics.** Discrete mathematics covers logic, sets, counting (combinatorics), graphs and trees, recursion and basic algorithm complexity. It shows up everywhere outside the smooth world of calculus: decision trees are trees, social networks and knowledge graphs are graphs, counting arguments give probabilities, and big-O thinking tells you whether brute-force nearest-neighbour search will survive ten times more data. Boolean logic underlies filters and rule-based features. The usual pitfall is ignoring it until a graph or complexity question appears; a short, focused course is enough for this roadmap.

```python
import numpy as np

# Fit y = w*x + b by minimizing mean squared error with gradient descent.
rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, 100)
y = 3.0 * x + 0.5 + rng.normal(0, 0.1, 100)

w, b, lr = 0.0, 0.0, 0.1
for _ in range(200):
    err = (w * x + b) - y
    w -= lr * 2 * np.mean(err * x)   # partial derivative of the loss w.r.t. w
    b -= lr * 2 * np.mean(err)       # partial derivative w.r.t. b
print(round(w, 2), round(b, 2))      # close to 3.0 and 0.5
```

### Probability

Probability is the grammar of uncertainty. Models output probabilities, data contains noise, and the loss functions of stage 11 come from probability assumptions.

**Basics of Probability.** A probability is a number between 0 and 1 attached to an event. The rules to know are the complement rule, the addition rule for "A or B", the multiplication rule for "A and B", conditional probability `P(A | B) = P(A and B) / P(B)`, and independence, where `P(A and B) = P(A) P(B)`. A classifier's output after softmax or a sigmoid is read as a probability, and sampling, noise and random initialization all rely on the same rules. The two classic errors are assuming independence where it does not hold (consecutive rows of a time series, several rows from the same customer) and mixing up `P(A | B)` with `P(B | A)`.

**Bayes Theorem.** Bayes' theorem turns a conditional probability around: `P(A | B) = P(B | A) P(A) / P(B)`, or "posterior is proportional to likelihood times prior". Naive Bayes classifiers use it directly, and Bayesian thinking explains why a positive result on a rare-condition test is often a false alarm. The common pitfall is base-rate neglect, which means ignoring the prior. The snippet below shows it with numbers.

```python
# A screening test: 1% prevalence, 90% sensitivity, 5% false-positive rate.
prior, sensitivity, false_positive = 0.01, 0.90, 0.05
evidence = sensitivity * prior + false_positive * (1 - prior)
posterior = sensitivity * prior / evidence
print(round(posterior, 3))   # about 0.154: most positive results are false alarms
```

**Random Variables, PDFs.** A random variable maps the outcomes of a random process to numbers. A discrete one has a probability mass function, and a continuous one has a probability density function (PDF) where the area under the curve over an interval is the probability of landing in it. Density values can exceed 1, because only areas count. The cumulative distribution function (CDF) accumulates probability from the left, and the expected value and variance summarize the centre and the spread. The likelihood is the density of the observed data viewed as a function of the parameters, and maximizing it explains why squared error goes with Gaussian noise and cross-entropy with categorical labels.

**Types of Distribution.** A handful of distributions cover most of what you meet. The skill is matching the data-generating process to a distribution and checking the match with a histogram or a Q-Q plot instead of assuming normality. Heavy-tailed data such as incomes or file sizes breaks the normal assumption and makes the mean misleading.

| Distribution | Describes | Typical ML use |
|--------------|-----------|----------------|
| Bernoulli and Binomial | One yes/no outcome, or the number of successes in `n` trials | Binary labels, click or conversion counts |
| Poisson | Number of events in a fixed interval | Arrival counts, rare-event modelling |
| Uniform | All values in a range equally likely | Random sampling, simple weight initialization |
| Normal (Gaussian) | Sums of many small effects, measurement noise | Noise models, weight initialization, Gaussian mixtures |
| Exponential | Waiting time between independent events | Survival and time-to-event features |

### Statistics

Statistics connects the numbers you computed to the world you care about: how typical is this value, how sure am I, and is this difference real?

**Basic concepts of statistics.** A population is everything you want to know about and a sample is the part you actually observed; a parameter describes the population and a statistic describes the sample. Variables are numerical (continuous or discrete) or categorical (nominal or ordinal), and the type decides which summaries and charts make sense. Sampling method matters: random or stratified samples are representative, whereas convenience samples carry bias such as selection or survivorship bias. Correlation does not imply causation, and a model trained on a biased sample will fail on the people it never saw.

**Descriptive Statistics.** Descriptive statistics summarise a dataset with a few numbers: centre (mean, median, mode), spread (range, variance, standard deviation, interquartile range) and shape (skewness, percentiles). The median and interquartile range resist outliers, so prefer them for skewed data. Correlation measures how two variables move together, but only linear association. In Pandas, `df.describe()` is a fast first look. The pitfall is summarizing with the mean alone: very different datasets can share the same mean and standard deviation, so always plot.

**Graphs and Charts.** Choose the chart for the question: a histogram for the shape of one variable, a box plot for spread and outliers, a scatter plot for the relation between two numeric variables, a bar chart for categories, a line chart for change over time and a heatmap for a correlation matrix. Label axes and units, start bar charts at zero, and avoid decoration that hides the data. Plots with thousands of points overplot, so use transparency or sampling. You will draw these with Matplotlib and Seaborn in stage 3.

**Inferential Statistics.** Inferential statistics draws conclusions about a population from a sample. A confidence interval gives a plausible range for a quantity, and a hypothesis test asks whether an observed difference is larger than chance would produce, using a p-value and a significance level. The central limit theorem explains why sample means are approximately normal for large samples, which makes many tests work. In ML you use these ideas to judge whether model A is really better than model B and whether an A/B test result is noise. A p-value is not the probability that your hypothesis is true, running many tests inflates false positives, and statistical significance is not the same as practical importance.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(1)
a = rng.normal(100, 15, 50)   # measurements for group A
b = rng.normal(108, 15, 50)   # measurements for group B

print(a.mean(), np.median(a), a.std(ddof=1))   # descriptive statistics
t, p = stats.ttest_ind(a, b, equal_var=False)  # Welch two-sample t-test
print(f"t={t:.2f}, p={p:.3f}")
```

**Try it.** Take any small numeric dataset (for example the `iris` or `diabetes` data from `sklearn.datasets`). Compute the mean, median and standard deviation of one column, draw its histogram, and fit a line to two columns with the gradient descent loop above. Check the gradient of your loss against a finite-difference estimate, and compute the inverse-free least-squares solution with `np.linalg.lstsq`; the two weights should agree closely.

**Self-check.**

- [ ] I can state the shape of the result of a matrix product and spot a shape error before running code.
- [ ] I can explain what an eigenvector and a singular value are and why PCA needs them.
- [ ] I can compute a derivative of a simple function by hand and verify it numerically.
- [ ] I can write one gradient descent step and explain what the learning rate does.
- [ ] I can apply Bayes' theorem to a diagnostic-test problem and explain base-rate neglect.
- [ ] I can pick a sensible distribution for count data, measurement noise and waiting times.
- [ ] I can compute and interpret a confidence interval or a t-test, and say what a p-value is not.

## 3. Programming Fundamentals

**Why it matters.** Most ML code is data manipulation: loading, reshaping, filtering, summarizing and plotting arrays and tables. A small amount of Python and four libraries cover the vast majority of early work, and fluency here makes every later stage faster. The goal is not clever code but readable, vectorized, reproducible code that someone else (including you in three months) can rerun.

### Python

Python dominates ML because it is readable, has a huge ecosystem, runs well in notebooks, and hands the heavy numeric work to fast compiled code underneath (NumPy, PyTorch, TensorFlow). Install a current Python 3 release from the official [Python site](https://www.python.org/), create one virtual environment per project, install packages into it, and record them in a `requirements.txt` or lock file so the project can be rebuilt (supported versions change, so check the official site, as of Oct 2026). Jupyter notebooks are excellent for exploration, but move stable logic into `.py` modules with functions you can test and import. The classic pitfall is installing everything globally: version conflicts then appear a few weeks later and are painful to untangle.

### Basic Syntax

The syntax needed for ML is small. Learn these six pieces well and the libraries will feel natural.

**Variables and Data Types.** A variable is a name bound to an object, and the main built-in types are `int`, `float`, `str`, `bool` and `None`. Python is dynamically typed, so the same name can hold different types, while NumPy and Pandas add fixed-size types such as `float32` and `int64` that save memory and speed up math. Floating-point numbers are approximate (`0.1 + 0.2` is not exactly `0.3`), so compare them with a tolerance. Type hints such as `x: float` are optional but catch many bugs when used with a checker.

**Data Structures.** Lists are ordered and mutable, tuples are ordered and immutable, dictionaries map keys to values and sets hold unique items. Choose by access pattern: a membership test in a set or dictionary is fast, while in a long list it scans every item. Assigning one list to another name creates an alias, not a copy, so edits show up in both; use `.copy()` when you need independence. Nested structures such as a list of dictionaries are exactly what JSON APIs return, which links this topic to stage 4.

**Loops.** `for` loops walk over any iterable, and `enumerate` and `zip` remove most index bookkeeping. List and dictionary comprehensions express simple loops in one line. For numeric work, prefer vectorized NumPy and Pandas operations over Python loops, which are often much faster because the loop runs in compiled code. Do not add or remove items from a list while you iterate over it; build a new list instead.

**Conditionals.** `if`, `elif` and `else` choose between branches, and any value has a truth value (empty containers, zero and `None` are false). On arrays and DataFrames, boolean masks replace most `if` statements: `df[df["age"] > 30]` keeps matching rows. Inside Pandas conditions use `&` and `|` with parentheses around each comparison, because `and` and `or` raise an error on whole arrays.

**Exceptions.** `try` and `except` handle errors you can recover from, `finally` runs cleanup, and `raise` signals a problem. Catch specific exception types and let the rest crash loudly, because a data pipeline that swallows errors produces silently wrong models. A bare `except:` hides real bugs and even catches `KeyboardInterrupt`. When you skip a bad record on purpose, count it and report the count.

**Functions and Built-in Functions.** A function groups logic behind a name with parameters (positional, keyword, default values, `*args`, `**kwargs`) and a return value. Small functions that take data in and return data out are easy to test and reuse. Built-ins such as `len`, `sum`, `min`, `max`, `sorted`, `zip`, `enumerate`, `range`, `round` and `isinstance` solve many tasks without imports. Never use a mutable default argument such as `def f(x, acc=[])`: the list is created once and shared between calls.

```python
def mean_by_group(rows, key, value):
    """Average `value` per distinct `key`; count and report rows that cannot be parsed."""
    groups, skipped = {}, 0
    for row in rows:
        try:
            groups.setdefault(row[key], []).append(float(row[value]))
        except (KeyError, ValueError, TypeError):
            skipped += 1
    if skipped:
        print(f"skipped {skipped} malformed rows")
    return {k: sum(v) / len(v) for k, v in groups.items()}

rows = [{"city": "A", "price": "10"}, {"city": "A", "price": "x"}, {"city": "B", "price": 30}]
print(mean_by_group(rows, "city", "price"))   # {'A': 10.0, 'B': 30.0}
```

### Object Oriented Programming

A class bundles data (attributes) with behaviour (methods); an object is one instance of a class, `__init__` sets it up, and inheritance or composition lets classes reuse each other. OOP matters for ML because the tools you use are built from it: every scikit-learn estimator has `fit`, `predict` or `transform` methods, and every PyTorch model is a subclass of `nn.Module`. Writing your own small estimator or transformer teaches you the conventions, and it lets custom preprocessing live inside a pipeline. Prefer composition (an object that holds other objects) over deep inheritance trees, and keep learned state in attributes created during `fit`; scikit-learn ends their names with an underscore. A subtle pitfall is a `predict` method that changes the object's state, because it makes results depend on call order.

```python
import numpy as np

class MeanBaseline:
    """Predicts the training mean: the simplest regressor and a useful baseline."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))   # learned state, scikit-learn naming style
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)

model = MeanBaseline().fit([[1], [2], [3]], [10, 20, 30])
print(model.predict([[4], [5]]))   # [20. 20.]
```

### Essential Libraries

Four libraries form the core of the Python data stack, and each builds on the previous one. Their official documentation is the best reference: [NumPy](https://numpy.org/doc/stable/), [Pandas](https://pandas.pydata.org/docs/), [Matplotlib](https://matplotlib.org/stable/) and [Seaborn](https://seaborn.pydata.org/).

| Library | Role | Core object | Reach for it when |
|---------|------|-------------|-------------------|
| NumPy | Fast numeric arrays and linear algebra | `ndarray` | You have homogeneous numbers: images, matrices, model inputs |
| Pandas | Labeled tabular data | `DataFrame`, `Series` | You have columns with names and types: CSV, SQL results, logs |
| Matplotlib | Low-level, fully controllable plotting | `Figure`, `Axes` | You need precise control or custom figures |
| Seaborn | Statistical plots with short code | Plot functions over a `DataFrame` | You want quick, good-looking distributions and relationships |

#### NumPy

NumPy provides the `ndarray`, a typed, fixed-shape array stored contiguously in memory, and operations that apply to whole arrays without Python loops. Key ideas are vectorization, broadcasting (how arrays of different shapes combine), the `axis` argument of reductions such as `mean` and `sum`, and the modern random generator `np.random.default_rng(seed)`. Slicing returns views that share memory with the original, so writing into a slice changes the source array. A frequent pitfall is mixing up axes (`axis=0` collapses rows, giving one value per column) and forgetting that the default float type is 64-bit while many deep learning frameworks prefer 32-bit.

#### Pandas

Pandas gives you the `DataFrame`, a table with labeled columns and an index, plus tools to read files, select with `.loc` and `.iloc`, filter, group (`groupby`), join (`merge`), reshape and handle dates and missing values. Think in whole-column operations rather than row loops, and use `df.info()`, `df.describe()` and `df.isna().sum()` as your first checks on any new table. Pandas aligns data by index, which is powerful but can produce surprising `NaN` results when indexes differ. Avoid chained assignment such as `df[df["a"] > 0]["b"] = 1`, which may modify a temporary copy; write `df.loc[df["a"] > 0, "b"] = 1` instead.

#### Matplotlib

Matplotlib is the foundation under most Python plotting. Use its object-oriented interface, `fig, ax = plt.subplots()`, so that you always know which axes you draw on, then set titles, axis labels and units, and save the figure to a file. It can draw anything, which also means that even simple polish takes some lines. Common pitfalls are mixing the implicit `plt.plot` state machine with explicit axes, and leaving hundreds of figures open in a loop, which wastes memory; call `plt.close(fig)` when you are done with a figure.

#### Seaborn

Seaborn sits on top of Matplotlib and works directly with DataFrames: `histplot`, `boxplot`, `scatterplot`, `heatmap` and `pairplot` produce informative statistical charts in one line, with grouping by colour through `hue`. It is ideal for exploratory data analysis (EDA), when you quickly want to see distributions, outliers and relations. Remember that many Seaborn functions aggregate for you (a bar plot shows a mean with an uncertainty interval), so read what is being computed before drawing conclusions. For publication-style control you can still reach the underlying Matplotlib objects.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

rng = np.random.default_rng(0)
df = pd.DataFrame({"area": rng.uniform(30, 150, 200)})
df["price"] = 2000 * df["area"] + rng.normal(0, 20000, 200)          # synthetic data
df["size"] = pd.cut(df["area"], [0, 60, 100, 200], labels=["small", "medium", "large"])

print(df.groupby("size", observed=True)["price"].agg(["count", "mean"]))
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
sns.scatterplot(data=df, x="area", y="price", hue="size", ax=axes[0])
sns.boxplot(data=df, x="size", y="price", ax=axes[1])
fig.tight_layout()
fig.savefig("eda.png")
```

**Try it.** Load `sklearn.datasets.load_diabetes(as_frame=True).frame` into a DataFrame. Use Pandas to report missing values, per-column summary statistics and the three columns most correlated with the target `target`. Then draw a histogram of the target, a scatter plot of the most correlated feature against it, and a correlation heatmap with Seaborn. Write three sentences about what the data shows, and save the notebook as your first EDA.

**Self-check.**

- [ ] I can write a function with default arguments, a docstring and error handling that fails loudly on unexpected input.
- [ ] I can choose between a list, tuple, dictionary and set and explain why.
- [ ] I can write a small class with `fit` and `predict` methods and learned attributes.
- [ ] I can vectorize a loop with NumPy and explain broadcasting and the `axis` argument.
- [ ] I can load, filter, group and merge tables in Pandas without chained assignment.
- [ ] I can produce labeled Matplotlib and Seaborn charts that answer a specific question.

## 4. Data Collection

**Why it matters.** A model can only learn what its data contains. Where data comes from decides its biases, its freshness, its legal status and how expensive it is to keep up to date. Beginners usually start with a tidy CSV, but real projects begin with a question like "where do we even have this information?", and the quality of the answer limits everything downstream.

### Data Sources

**Databases (SQL and NoSQL).** Relational databases (PostgreSQL, MySQL, SQLite) store tables linked by keys and are queried with SQL; most company data about customers, orders and events lives there, so `SELECT`, `JOIN`, `GROUP BY` and window functions are core ML skills. NoSQL databases trade fixed schemas for flexibility or scale: document stores such as MongoDB hold JSON-like records, key-value stores serve fast lookups, and graph stores model relationships. Official documentation: [SQLite](https://www.sqlite.org/docs.html) and [MongoDB](https://www.mongodb.com/docs/). The big pitfall is building a training table with information that was not yet known at prediction time (a join that pulls in later events); always build features "as of" the prediction moment.

**Internet.** Public datasets from places like [Kaggle](https://www.kaggle.com/datasets), the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) and [OpenML](https://www.openml.org/) are ideal for learning, and open web pages can be scraped when no better source exists. Before scraping, read the site's terms of service and `robots.txt`, respect rate limits, and think about copyright and personal data. Scrapers break whenever a page layout changes, so prefer official downloads or APIs. Scraped text is also full of duplicates, boilerplate and spam that you must clean.

**APIs.** Many services expose data through HTTP APIs that return JSON. Call them with a library such as [Requests](https://requests.readthedocs.io/), always set a timeout, handle pagination and rate limits with retries and backoff, and cache raw responses so a failed run does not force a complete re-download. Keep keys and tokens in environment variables, never in code or notebooks that you commit. The two most common mistakes are hard-coded secrets and calls without timeouts that hang a whole job.

**Mobile Apps.** Mobile apps generate event logs (taps, screens, purchases), sensor readings (location, motion) and media. These arrive through analytics SDKs or your own backend, so the data engineering team must agree on event names and fields. Consent and privacy rules apply strongly here: collect only what you need, tell users, and anonymize where possible. Beware of schema drift between app versions and operating systems, and of sampling bias, because logged data describes only the people who installed and used the app.

**IoT.** Internet of Things devices stream time-stamped sensor readings, often through gateways and messaging protocols such as MQTT. Typical problems are missing or delayed packets, clock drift, sensor calibration, different sampling rates and sheer volume, so teams often aggregate at the edge and store the rest in time-series databases or files. Record the device ID, firmware version and units alongside every reading, or the data becomes impossible to interpret later. The usual modelling work includes resampling to a common time grid and handling gaps.

```python
import os
import requests

API_KEY = os.environ["DATA_API_KEY"]                       # never hard-code secrets
URL = os.environ.get("DATA_API_URL", "https://api.example.com/v1/items")

resp = requests.get(URL, params={"page": 1}, timeout=10,
                    headers={"Authorization": f"Bearer {API_KEY}"})
resp.raise_for_status()
records = resp.json()                                      # often a list of dicts for pd.DataFrame(records)
```

```python
import sqlite3
import pandas as pd

con = sqlite3.connect(":memory:")
pd.DataFrame({"user_id": [1, 2, 3], "plan": ["free", "pro", "free"]}).to_sql("users", con, index=False)
pd.DataFrame({"user_id": [1, 1, 3], "amount": [5.0, 7.5, 2.0]}).to_sql("orders", con, index=False)

query = """
SELECT u.user_id, u.plan, COALESCE(SUM(o.amount), 0) AS total_spent
FROM users u LEFT JOIN orders o ON o.user_id = u.user_id
GROUP BY u.user_id, u.plan
"""
print(pd.read_sql_query(query, con))
```

### Data Formats

The format you store data in affects size, speed, type safety and who can open it. Choose deliberately; do not leave everything as CSV by habit.

**CSV.** A plain-text, row-based format that every tool can read. It stores no types, and delimiters, quoting, encodings and decimal commas vary, so pass `dtype`, `parse_dates` and `encoding` explicitly to `pd.read_csv`. Identifiers such as `00123` lose their leading zeros if they are guessed to be numbers.

**Excel.** `.xlsx` workbooks hold several sheets, formatting and formulas, and are the working format of many business teams; `pd.read_excel` reads them (it needs an engine such as `openpyxl`). Merged cells, headers in the middle of a sheet, dates stored as numbers and formulas versus cached values cause trouble, so export to CSV or Parquet early and treat the workbook as an input, not as pipeline storage.

**JSON.** A nested, human-readable format used by APIs and configuration. Use `pd.json_normalize` to flatten nested records, and JSON Lines (one object per line) for large files because it can be streamed. Watch for inconsistent keys and arrays of different lengths between records.

**Parquet.** A columnar, compressed, typed binary format that is the usual choice for analytics and ML data; reading only the columns you need is fast, and types survive round trips. It needs `pyarrow` or `fastparquet` and is not human-readable. Many tiny files hurt performance, so write fewer, larger files. See the [Apache Parquet](https://parquet.apache.org/) site for the specification.

**Other formats.** XML for legacy systems, Feather and Avro or ORC for interchange, HDF5, Zarr and NumPy `.npz` files for large arrays, SQLite as a single-file database, and TFRecord for TensorFlow pipelines. Images, audio and video are usually stored as files with a manifest table that lists paths and labels. Python's `pickle` can run code when loading a file, so never unpickle files from sources you do not trust.

| Format | Strengths | Weaknesses | Use when |
|--------|-----------|------------|----------|
| CSV | Universal, readable, tiny tooling needs | No types, slow for big data, encoding traps | Sharing small tables, simple exports |
| Excel | Familiar to business users, multiple sheets | Messy structure, not reproducible | Receiving data from non-technical teams |
| JSON / JSON Lines | Nested records, API-native | Verbose, schema not enforced | API responses, logs, configuration |
| Parquet | Compact, fast, typed, column selection | Binary, needs a library | Training datasets and feature stores |
| Arrays (`.npz`, HDF5, Zarr) | Efficient for big numeric arrays | Less suited to mixed tabular data | Images, embeddings, scientific data |

```python
import pandas as pd

df = pd.DataFrame({"id": [1, 2], "city": ["Pune", "Oslo"], "score": [0.5, 0.9]})
df.to_csv("data.csv", index=False)
df.to_json("data.jsonl", orient="records", lines=True)
df.to_parquet("data.parquet")                        # needs pyarrow or fastparquet
print(pd.read_parquet("data.parquet", columns=["id", "score"]))
```

**Try it.** Build one small dataset from three sources: a CSV you download, a JSON response from a public API of your choice (key in an environment variable), and a SQLite table you create. Join them into one DataFrame, store it as Parquet, and write a short `DATA_NOTES.md` that records each source, its licence, the collection date and the known biases.

**Self-check.**

- [ ] I can write a SQL query with a join and a group-by and load the result into Pandas.
- [ ] I can call a JSON API safely, with a timeout, error handling and the key in an environment variable.
- [ ] I can explain the legal and ethical checks I run before scraping or collecting user data.
- [ ] I can name two problems specific to mobile-app data and two specific to IoT data.
- [ ] I can pick between CSV, JSON, Parquet and array formats for a given dataset and justify it.
- [ ] I can document a dataset's source, collection date, licence and known biases.

## 5. Data Cleaning and Preprocessing Techniques

**Why it matters.** Raw data is rarely model-ready: it has gaps, typos, duplicates, mixed units, text where numbers should be, and columns on wildly different scales. Experienced practitioners often spend more time here than on modelling, and good preprocessing frequently beats a fancier algorithm. The central rule of this stage is to avoid **data leakage**: every transformation that learns something from data (a mean, a vocabulary, a set of selected features) must be fitted on the training data only and then applied unchanged to validation and test data. Putting all steps inside a scikit-learn `Pipeline` enforces that rule.

### Data Cleaning

Cleaning means making the data correct and consistent. Handle **missing values** by understanding why they are missing (completely at random, depending on other columns, or depending on the missing value itself), then drop, fill with a median, mode or constant, impute with a model, and often add an "is missing" indicator because missingness itself can be informative. Remove exact **duplicates**, fix wrong types and units, standardize category labels ("NY", "New York", "ny"), and check ranges (negative ages, dates in the future). Treat **outliers** carefully: investigate whether they are errors or rare real events before removing, capping (winsorizing) or leaving them. Keep the raw data untouched and perform every cleaning step in code, so the process is documented and repeatable; dropping rows silently can bias the dataset.

### Feature Engineering

Feature engineering creates inputs that make the pattern easier to learn. Common moves are ratios and differences (price per square metre), date parts (hour, weekday, month), group aggregates (a customer's average order value), counts and lengths from text, interaction terms, binning, and log transforms for skewed values. Categorical variables must become numbers: one-hot encoding for a few unordered categories, ordinal encoding when order is real, and carefully cross-validated target encoding for high-cardinality columns. Cyclical features such as hour of day are better encoded with sine and cosine so that 23:00 and 00:00 are close. The dangerous pitfall is **target leakage**, where a feature secretly contains the answer (for example a "refund issued" flag when predicting returns), which gives great validation scores and a useless production model.

### Feature Scaling and Normalization

Scaling puts numeric features on comparable ranges. Distance-based and gradient-based methods (k-nearest neighbours, SVMs, regularized linear models, neural networks, PCA) are sensitive to scale, whereas tree-based models are not. The word "normalization" is used loosely, so always check which transformation someone means. Fit the scaler on the training data only, then reuse its stored statistics on new data; fitting on everything is a quiet form of leakage.

| Method | Idea | Use when |
|--------|------|----------|
| Standardization (z-score), `StandardScaler` | Subtract the mean, divide by the standard deviation | Default for most linear, distance and neural models |
| Min-max scaling, `MinMaxScaler` | Rescale to a fixed range such as 0 to 1 | Bounded inputs, image pixels, some neural networks |
| Robust scaling, `RobustScaler` | Use median and interquartile range | Data with many outliers |
| Log or power transform | Compress long right tails | Prices, counts and other skewed positive values |

### Dimensionality Reduction

Datasets with hundreds or thousands of features suffer from the curse of dimensionality: distances lose meaning, models overfit more easily, and training gets slower. Dimensionality reduction either selects a subset of the original columns or builds a smaller set of new features that keeps most of the information. Principal component analysis (PCA, covered in stage 8) is the standard linear method, and t-SNE and UMAP are mainly used to visualize data in two dimensions, not as inputs to a model. Fit reduction on the training data only and scale features first. Remember that new features are mixtures of the old ones, so they are harder to explain.

### Feature Selection

Feature selection keeps the columns that help and drops the rest, which reduces overfitting, speeds up training and makes models easier to explain. Three families exist, and the right one depends on cost and the model you use.

| Family | Examples | Cost | Watch out for |
|--------|----------|------|---------------|
| Filter | Variance threshold, correlation, mutual information, chi-squared | Cheap, model-independent | Ignores interactions between features |
| Wrapper | Recursive feature elimination (`RFE`) | Expensive, trains many models | Overfits the selection process on small data |
| Embedded | L1 (Lasso) weights, tree-based importances | Free while training | Correlated features share or split importance |

Do the selection inside the cross-validation loop. Choosing features with the full dataset first and then cross-validating makes the scores look better than reality.

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.DataFrame({
    "age": [25, 32, None, 51, 46, 29, 38, 60, 41, 35, 27, 55],
    "income": [30, 52, 41, None, 75, 38, 60, 80, 58, 49, 33, 72],
    "city": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C", "B", "A"],
    "bought": [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
})
X, y = df.drop(columns="bought"), df["bought"]

prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["age", "income"]),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"]),
])
pipe = Pipeline([("prep", prep), ("select", SelectKBest(f_classif, k=3)), ("clf", LogisticRegression())])
print(cross_val_score(pipe, X, y, cv=3).mean())   # every step is re-fitted inside each fold: no leakage
```

**Try it.** Take a messy public dataset (for example a Kaggle housing or Titanic file). Write a cleaning report that lists missing values per column, duplicates, odd categories and outliers, then build a `ColumnTransformer` with imputation, scaling and one-hot encoding. Compare cross-validated accuracy or error of a simple model with and without feature engineering, and check that fitting the scaler on the full dataset instead of the training folds changes the score slightly (that difference is the leakage).

**Self-check.**

- [ ] I can inspect a new dataset for missing values, duplicates, type problems and outliers.
- [ ] I can choose an imputation strategy and explain what it assumes about the missingness.
- [ ] I can create at least five kinds of engineered features and name a target-leakage risk for each.
- [ ] I can pick a scaler for a given model and say why trees do not need one.
- [ ] I can explain the curse of dimensionality and when to use selection versus extraction.
- [ ] I can build a Pipeline with a ColumnTransformer so that nothing is fitted on test data.

## 6. Machine Learning Basics and the Scikit-learn Workflow

**Why it matters.** This stage gives you the mental model and the working routine that every later algorithm plugs into. Once you know what a model is, what "learning" optimizes, which of the five learning settings your problem belongs to and how a scikit-learn project flows from data to predictions, learning each new algorithm becomes a matter of filling a slot rather than starting over. The [scikit-learn documentation](https://scikit-learn.org/stable/) is one of the best teaching resources in the field; keep it open.

### What is Machine Learning?

Machine learning is building programs that improve at a task by learning from examples instead of following hand-written rules. A **model** is a function with adjustable parameters, **training** searches for the parameter values that minimize a **loss** (a number measuring the mistakes on the training data), and the real goal is **generalization**: good performance on data the model has never seen. Too simple a model **underfits** (it misses the pattern), while one that is too flexible **overfits** (it memorizes noise), and this bias-variance trade-off appears in every algorithm. ML is a poor choice when simple rules suffice, when there is little relevant data, or when every error is unacceptable; start by asking whether a spreadsheet formula or a lookup table already solves the problem. The biggest beginner trap is trusting training accuracy, which says almost nothing about future performance.

### Types of ML

Learning problems are classified by the kind of feedback the learner receives. The same model family can appear in several settings, so think of these as problem types.

**Supervised.** Every training example comes with a label, and the model learns to predict it: a category (classification) or a number (regression). This is the most common setting in industry and the subject of stage 7. The main cost is getting enough accurate labels.

**Unsupervised.** There are no labels; the model looks for structure such as groups (clustering), compact representations (dimensionality reduction) or unusual points (anomaly detection). Stage 8 covers it. Because there is no ground truth, judging results takes more care.

**Semi-supervised.** A few labeled examples are combined with many unlabeled ones, for instance by training on the labeled set, predicting labels for unlabeled data and retraining on confident predictions (self-training, as in scikit-learn's `SelfTrainingClassifier`) or by propagating labels across a similarity graph. It helps when labeling is expensive, but wrong pseudo-labels can reinforce themselves, so validate on a clean labeled set.

**Self-supervised.** The labels are created from the data itself: hide a word and predict it, predict the next token, or recognize two augmented views of the same image. It lets models learn from huge unlabeled collections, and it is how the base models behind modern language and vision systems are pre-trained (stages 11 and 12). Do not confuse it with semi-supervised learning, which still needs some human labels.

**Reinforcement.** An agent acts in an environment and receives rewards, learning a policy that maximizes long-term reward. There are no labeled examples, only trial, error and delayed feedback; stage 9 covers it.

| Type | Data you need | Typical goal | Example |
|------|---------------|--------------|---------|
| Supervised | Inputs with labels | Predict a label or number | Spam filter, house price model |
| Unsupervised | Inputs only | Find groups or structure | Customer segments |
| Semi-supervised | Few labels, many unlabeled inputs | Predict labels cheaply | Medical images with few expert labels |
| Self-supervised | Raw inputs; labels built from the data | Learn general representations | Language model pre-training |
| Reinforcement | An environment with rewards | Learn a good sequence of actions | Game-playing agent, robot control |

### The Scikit-learn Workflow

Scikit-learn gives nearly every algorithm the same interface: an **estimator** has `fit`, a model also has `predict`, and a preprocessing step has `transform`. A `Pipeline` chains steps into one estimator, so the whole workflow can be cross-validated, tuned and saved as one object. The six steps below describe the usual order; the code after them runs end to end.

**Data Loading.** Get the features `X` and target `y` from a built-in dataset (`load_*` functions are bundled; `fetch_*` ones download), a file or a database. Check shapes, types, missing values and class balance before anything else, and make sure identifiers or columns derived from the target are not left in `X`.

**Train-Test Data.** `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)` keeps a final test set that you touch only once at the end; stratify classification splits so each class appears in the same proportion. For time-ordered data split by time instead of randomly. Making repeated changes until the test score looks good turns the test set into a second training set.

**Data Preparation.** Imputation, encoding and scaling go inside a `Pipeline` or `ColumnTransformer` (stage 5) so they are fitted on training folds only.

**Model Selection.** Begin with a baseline (`DummyClassifier`, or a plain linear model), then compare two to four candidate algorithms with cross-validation on the training data under the same metric. Prefer the simplest model that does the job. Comparing candidates on the test set means the test result is no longer an honest estimate.

**Tuning.** Hyperparameters are settings chosen before training, such as regularization strength, tree depth or neighbours `k`. `GridSearchCV` tries every combination and `RandomizedSearchCV` samples them, which scales better to many settings; both use cross-validation internally. A search over too many options on a small dataset can overfit the validation folds, so keep the final test set for one last check.

**Prediction.** `predict` returns labels and `predict_proba` returns class probabilities, from which you can choose a decision threshold that fits the cost of errors. Save the fitted pipeline (not just the model) with `joblib` and load only files you trust. In production the input columns must match the training columns exactly, and training/serving skew is a classic source of silent failure.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)                       # 1. data loading
X_train, X_test, y_train, y_test = train_test_split(             # 2. train-test split
    X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])  # 3. preparation
search = GridSearchCV(                                           # 4-5. selection and tuning, 5-fold CV
    pipe,
    param_grid=[{"clf": [LogisticRegression(max_iter=1000)], "clf__C": [0.1, 1, 10]},
                {"clf": [RandomForestClassifier(random_state=0)], "clf__n_estimators": [100, 300]}],
    cv=5, scoring="f1")
search.fit(X_train, y_train)
print(search.best_params_, round(search.best_score_, 3))

print(classification_report(y_test, search.predict(X_test)))     # 6. prediction on untouched test data
```

**Try it.** Repeat the workflow on a regression dataset such as `fetch_california_housing` (it downloads on first use) or `load_diabetes`. Start with `DummyRegressor`, then add a linear model and a random forest, and report the cross-validated error of each plus the final test error of the winner. Then deliberately break the rule: scale the full dataset before splitting and note how the score changes, so you can recognize leakage when you see it.

**Self-check.**

- [ ] I can define model, parameters, loss, generalization, overfitting and underfitting.
- [ ] I can say when ML is the wrong tool for a problem.
- [ ] I can place a problem into supervised, unsupervised, semi-supervised, self-supervised or reinforcement learning, with an example of each.
- [ ] I can write the six workflow steps from memory and explain what each protects against.
- [ ] I can build a Pipeline, tune it with `GridSearchCV` and report a final test score once.
- [ ] I can save a fitted pipeline and predict with it on new data.

@@CONTINUE@@
