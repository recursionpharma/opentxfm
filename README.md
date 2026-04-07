# [Effective Biological Representation Learning by Masking Gene Expression](https://openreview.net/forum?id=NqZqClqtTK)

## Accepted at the ICLR 2026 Workshop on Foundation Models for Science (FM4Sci).

This repo is under construction - stay tuned for updates as we share relevant code and artifacts.

At a high level, see the following definitions for our loss and activation functions. Note that RectifiedSigmoid is mathematically equivalent to the more elegantly defined RectifiedTanh as presented in the paper.

```python

class PoissonNLLLogSpace(nn.Module):
    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction

    def forward(self, log1p_input, log1p_target):
        # we must transform target out of log space to have valid poisson loss
        loss = log1p_input.exp() - log1p_target.exp() * log1p_input
        return loss

class RectifiedSigmoid(nn.Module):
    def __init__(self, upper_bound: int = 100000):
        super().__init__()
        self.upper_bound = np.log1p(upper_bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upper_bound * F.relu(2 * torch.sigmoid(x / (2 * np.e)) - 1) 
```


## Benchmarking

See [this paper](https://arxiv.org/abs/2410.13956), "Benchmarking Transcriptomics Foundation Models for Perturbation Analysis : one PCA still rules them all", which defines the major perturbational benchmarks we use in this work ([repo](https://github.com/valence-labs/Tx-Evaluation)). Note that we have made some improvements to some of the evaluations originally presented in that paper 2 years ago, as we describe in the appendixc. More updates to come.

