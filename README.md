---
author:
- Your Name
authors:
- Your Name
title: Argmin Inference with argminCS
toc-title: Table of contents
---

Welcome to the tutorial of R package `argminCS`! Given noisy measures of
several competing agents---they can be machine learning algorithms,
voting candidates or maximum likelihood candidate parameters---we are
interested in which one(s) of them achieving the best performance.

Suppose we are comparing the performance of 20 agents, we have 200 loss
noise-contaminated measurements for each of them.

::: cell
``` {.r .cell-code}
sample_size = 200
agent_num = 20

set.seed(108)
true_loss = ((1:agent_num)/agent_num)^2
cov = diag(length(true_loss))
data = MASS::mvrnorm(sample_size, true_loss, cov)
rownames(data) = paste0("Obs_", 1:sample_size)
colnames(data) = paste0("Agent_", 1:agent_num)
```
:::

Here is the observed data.

:::: cell
``` {.r .cell-code}
print(data[1:5,1:5])
```

::: {.cell-output .cell-output-stdout}
             Agent_1    Agent_2    Agent_3    Agent_4     Agent_5
    Obs_1 -0.9433241 -0.3155622  0.8122737  0.6012614  1.04739675
    Obs_2 -0.1276607  0.4080168  0.6417950  0.1001458 -0.33183949
    Obs_3  1.1252646  0.7435761 -0.2231430  0.8091007 -1.24457611
    Obs_4  0.4936495  0.5179427  0.9584380 -0.9977787 -0.02076811
    Obs_5  1.4085057  0.1973783 -0.3347059  0.1026901  0.95850312
:::
::::

We know agent #1 should be the winner with average/true loss.

:::: cell
``` {.r .cell-code}
library(ggplot2)
df <- data.frame(
    Agent = factor(1:agent_num),
    True_Loss = true_loss
)
ggplot(df, aes(x = Agent, y = True_Loss, group = 1)) +
    geom_line(color = "steelblue", linewidth = 1) +
    geom_point(color = "darkred", size = 2) +
    theme_minimal() +
    labs(title = "True Loss for Each Agent",
             x = "Agent",
             y = "True Loss")
```

::: cell-output-display
![](demo_CSargmin_files/figure-markdown/unnamed-chunk-3-1.png)
:::
::::

And we can easily use argmin of the simple sample mean as our estimate
of the best performer.

::::: cell
``` {.r .cell-code}
col_avg <- colMeans(data)
winner_est <- which.min(col_avg)
cat("Estimated winner: Agent", winner_est, "with average loss", col_avg[winner_est], "\n")
```

::: {.cell-output .cell-output-stdout}
    Estimated winner: Agent 2 with average loss -0.1070976 
:::

``` {.r .cell-code}
if (winner_est != 1) cat("Opps: True winner should be Agent 1\n")
```

::: {.cell-output .cell-output-stdout}
    Opps: True winner should be Agent 1
:::
:::::

Since the data we observed in real application is always noise
contaminated, it is important to quantify the uncertainty in our winner
selection. The `argminCS` package provides a way to construct a
confidence set for the best agent.

::: cell
``` {.r .cell-code}
if (!requireNamespace("argminCS", quietly = TRUE)) {
    install.packages("argminCS")
}
library(argminCS)
# CS.argmin(data, method='SML') #CS.argmin is not working
```
:::

:::: cell
``` {.r .cell-code}
dimension = 4
difference.matrix <- matrix(rep(data[, dimension], agent_num-1), 
                            ncol = agent_num-1, 
                            byrow = FALSE) - data[, -dimension]
argmin.HT(difference.matrix, method='SML')
```

::: {.cell-output .cell-output-stdout}
    $test.stat.scale
    [1] -1.162869

    $critical.value
    [1] 1.644854

    $std
    [1] 0.7084998

    $ans
    [1] "Accept"

    $lambda
    [1] 5.656854

    $lambda.capped
    [1] FALSE

    $residual.slepian
    [1] 0.05538992

    $variance.bound
    [1] 0.02660854

    $test.stat.centered
    NULL
:::
::::
