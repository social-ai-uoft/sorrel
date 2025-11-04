# Unified Icon Generation System Design Specification

## **Core Concept: Shared Autoencoder with Type Latent**

The Unified Icon Generation System uses a single shared autoencoder with a type latent factor that distinguishes between agents (0) and resources (1). This creates entangled representations for both entity types in a unified space, enabling consistent icon generation across all entities in the environment.

## **1. Latent Factor Structure**

The system uses continuous latent factors for both agents and resources. The type factor distinguishes between agents (positive values) and resources (negative values). The other factors are continuous values sampled from a standard normal distribution: color (representing hue), size (representing scale), shape (representing form), and pattern (representing texture). The factor names like "color", "size", "shape", and "pattern" are only for human understanding and do not represent the true underlying factors that the autoencoder learns to disentangle.

## **2. Input Encoding Strategy**

Each entity configuration is encoded as an N-dimensional vector sampled from a standard normal distribution, where N is the number of latent factors used in the study. The first dimension represents the type factor (positive values for agents, negative values for resources). The remaining N-1 dimensions represent continuous latent factors: color (hue), size (scale), shape (form), and pattern (texture), each sampled independently from N(0,1). This creates a representation where the true underlying factors are learned by the autoencoder rather than being explicitly encoded.

## **3. Autoencoder Architecture**

The autoencoder takes N-dimensional input vectors and produces 32-dimensional entangled representations, where N is the number of latent factors used in the study. The encoder consists of a single hidden layer: a linear transformation from N to 32 units followed by a ReLU nonlinearity. The decoder is a linear reconstruction layer from 32 back to N units. This shallow architecture ensures that the resulting 32-dimensional codes are smooth, nonlinear combinations of the latent factors rather than compressed or axis-aligned representations.

## **4. Training Strategy**

To generate high-dimensional, smoothly entangled representations of low-dimensional latent variables, we trained a single-hidden-layer feedforward autoencoder, following the design of Johnston and Fusi (2023). The goal of this model was not to discover abstract structure but to produce non-abstract, information-preserving representations that could later serve as the input to a multitasking network. The autoencoder consisted of one encoder layer and one decoder layer: a linear transformation from N to 32 units followed by a ReLU nonlinearity, and a linear reconstruction layer from 32 back to N units, where N is the number of latent factors used in the study. This shallow architecture ensures that the resulting 32-dimensional codes are smooth, nonlinear combinations of the latent factors rather than compressed or axis-aligned representations.

Each input vector z ∈ R^N was sampled independently from a standard normal distribution, z_i ∼ N(0,1). A total of 100,000 such latent vectors were used for training and an additional 10,000 independently sampled vectors for testing. Because the latent space is continuous, the train/test split simply involved sampling disjoint sets from the same distribution rather than partitioning a fixed dataset. This setup ensures that the model generalizes smoothly to unseen latent combinations without overfitting specific input configurations.

The autoencoder was optimized to minimize mean-squared reconstruction error between input and output, defined as L = ∥z - ẑ∥², where ẑ is the reconstructed latent vector. Training was performed using the Adam optimizer with a learning rate of 10^-3, weight decay of 10^-4, and a batch size of 256. The model was trained for approximately 20,000 update steps, which was sufficient for convergence of the reconstruction loss. No additional regularization, dropout, or normalization layers were applied, as the objective was to maintain full information content while embedding the latent variables in a higher-dimensional manifold.

After training, the model's reconstruction error on the held-out test set was verified to be comparable to the training error, confirming smooth generalization. The decoder was then discarded, and the encoder weights were frozen. The resulting encoder function f_enc: R^N → R^32 was used to transform latent vectors into 32-dimensional entangled representations for all subsequent multitasking experiments. These entangled codes are high-dimensional expansions of the original latent variables: they preserve all underlying information but in a nonlinear, non-factorized form.

To confirm that the encoder produced genuinely entangled rather than abstract representations, we examined the participation ratio (effective dimensionality) and inspected pairwise correlations between latent dimensions and encoder activations. The representations were found to be high-dimensional (approximately 32 effective components) and non-axis-aligned, consistent with the intended entangled geometry. This procedure ensures that any emergence of abstraction or disentanglement in later models can be attributed to the effects of coordination learning, rather than preexisting structure in the input encoding.

## **Participation Ratio (PR): Measuring Representation Dimensionality**

The **Participation Ratio (PR)** quantifies the *effective dimensionality* of a neural representation — that is, how many independent directions of variance are meaningfully used in a population of activations.  
It was used in **Johnston & Fusi (2023)** to confirm that the input autoencoder's representation layer was *high-dimensional and entangled* (≈200 effective dimensions).

### **1. Concept**

Neural activations often lie in a lower-dimensional subspace than their raw number of units.  
The PR measures how "spread out" these activations are across orthogonal directions.

- If all neurons vary **independently**, PR ≈ number of neurons (full dimensionality).  
- If many neurons are **correlated or redundant**, PR is smaller (low effective dimensionality).

In the context of *Johnston & Fusi (2023)*:
- A **high PR** means the representation is **entangled**, nonlinear, and information-rich.
- A **low PR** means it has become **factorized or abstract**, with redundant or compressed structure.

### **2. Mathematical Definition**

Given an activation matrix X ∈ R^(N×M)  
where N = number of neurons and M = number of samples, compute the covariance matrix:

C = (1/M) X X^T

Let λ_i denote the eigenvalues of C, representing variance captured by each principal component.

Then:

PR = (∑_i λ_i)² / ∑_i λ_i²

Equivalently, if p_i = λ_i / ∑_j λ_j are normalized variances:

PR = 1 / ∑_i p_i²

#### **Interpretation:**
- **PR = N** → all eigenvalues equal → full-rank, high-dimensional code.  
- **PR = 1** → only one dominant eigenvalue → collapsed or one-dimensional representation.

### **3. PyTorch Implementation**

```python
import torch

def participation_ratio(activations):
    """
    activations: torch.Tensor of shape [samples, neurons]
    returns: scalar PR value
    """
    # Center data
    X = activations - activations.mean(dim=0, keepdim=True)
    # Covariance
    C = X.T @ X / X.shape[0]
    # Eigenvalues
    eigvals = torch.linalg.eigvalsh(C)
    # Participation ratio
    pr = (eigvals.sum() ** 2 / (eigvals ** 2).sum()).item()
    return pr
```



## **5. Icon Generation Process**

To generate an agent icon, we create an N-dimensional vector with a positive type value and continuous values for the remaining latent factors, then pass it through the frozen encoder to get a 32-dimensional entangled representation. For resource icons, we use a negative type value and include all continuous factors. Both use the same frozen encoder, ensuring consistent representation space. The continuous nature of the factors allows for smooth interpolation and generation of novel entity configurations.

## **7. Implementation Structure**

The Unified Icon Generation System consists of the following components:

- **UnifiedIconSystem**: Core autoencoder class that handles training and icon generation
- **Icon generation utilities**: Functions for creating visual representations from latent factors
- **Validation utilities**: Functions for testing entanglement quality and reconstruction accuracy
- **Integration interface**: Methods for integrating with existing agent and resource classes

## **8. Validation Strategy**

The system validates entanglement across both agent and resource types using multiple approaches:

### **Linear Separability Testing**
Type separability should be high since agents and resources should be distinguishable. The continuous factors (color, size, shape, pattern) should show poor separability since they should be entangled in the learned representation space. The entanglement score is calculated as one minus the mean accuracy of the non-type factors, measuring how well the autoencoder has learned to create entangled representations of the continuous latent factors.

### **Participation Ratio Validation**
The participation ratio should be high (close to 32) to confirm that the 32-dimensional representations are genuinely high-dimensional and entangled rather than compressed or factorized. A high PR indicates that the autoencoder has successfully created non-abstract, information-preserving representations that use the full dimensionality of the representation space.

## **9. Training Data Examples**

Training data consists of random samples from the continuous latent space. For agents, we generate random values for color, size, and shape factors (with pattern fixed at 0), combined with type [1,0]. For resources, we generate random values for all four factors (color, size, shape, pattern) combined with type [0,1]. We generate thousands of random samples to ensure the autoencoder learns to handle the full continuous range of possible factor combinations, rather than being limited to discrete categorical values.

## **10. Success Metrics**

- **Type accuracy**: Should be greater than 0.8 since agents versus resources should be distinguishable
- **Factor entanglement**: Continuous factors (color, size, shape, pattern) should show entanglement scores less than 0.6, indicating they are poorly separable in the learned representation space
- **Overall entanglement score**: Should be greater than 0.4 for good entanglement
- **Participation ratio**: Should be close to 32 (the full dimensionality) to confirm high-dimensional, entangled representations
- **Reconstruction error**: Should achieve low MSE (< 0.01) on the continuous input space, demonstrating meaningful representation learning

## **11. API Specification**

### **UnifiedIconSystem Class**
- `__init__(input_dim=N, hidden_dim=32, output_dim=32)`: Initialize the autoencoder where N is the number of latent factors
- `train(training_data, steps=20000, learning_rate=0.001)`: Train the autoencoder on continuous latent factors
- `generate_icon(latent_vector)`: Generate entangled representation from latent factors
- `validate_entanglement(test_data)`: Test entanglement quality using linear separability
- `compute_participation_ratio(activations)`: Compute participation ratio for validation

### **Integration Interface**
- `create_agent_icon(latent_factors)`: Generate visual representation for an agent given its latent factors
- `create_resource_icon(latent_factors)`: Generate visual representation for a resource given its latent factors
- `get_icon_dimensions()`: Return the dimensions of the generated icon vectors

### **Usage Pattern**
The system integrates with existing agent and resource classes by replacing their one-hot encoding with generated icons. Agents and resources call the icon generation functions with their latent factors to get visual representations that can be used in the observation system.

This design specification provides a complete blueprint for implementing the Unified Icon Generation System as a standalone appearance generation component.