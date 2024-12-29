import torch
import numpy as np


def max_sharpe(y_return, weights):
    weights = torch.unsqueeze(weights, 1)
    meanReturn = torch.unsqueeze(torch.mean(y_return, axis=1), 2)
    covmat = torch.Tensor([np.cov(batch.cpu().T, ddof=0) for batch in y_return]).to("cpu")
    portReturn = torch.matmul(weights, meanReturn)
    portVol = torch.matmul(
        weights, torch.matmul(covmat, torch.transpose(weights, 2, 1))
    )
    objective = (portReturn * 12 - 0.02) / (torch.sqrt(portVol * 12))
    return -objective.mean()


def equal_risk_parity(y_return, weights):
    B = y_return.shape[0]
    F = y_return.shape[2]
    weights = torch.unsqueeze(weights, 1).to("cpu")
    covmat = torch.Tensor(
        [np.cov(batch.cpu().T, ddof=0) for batch in y_return]
    )  # (batch, 50, 50)
    covmat = covmat.to("cpu")
    sigma = torch.sqrt(
        torch.matmul(weights, torch.matmul(covmat, torch.transpose(weights, 2, 1)))
    )
    mrc = (1 / sigma) * (covmat @ torch.transpose(weights, 2, 1))
    rc = weights.view(B, F) * mrc.view(B, F)
    target = (torch.ones((B, F)) * (1 / F)).to("cpu")
    risk_diffs = rc - target
    sum_risk_diffs_squared = torch.mean(torch.square(risk_diffs))
    return sum_risk_diffs_squared

def max_r2(y_return, x_pred):
    #print(y_return[0, 0, 0])
    #print(x_pred[0, 0])
    ssr = torch.sum((y_return[:, 0, :] - x_pred) ** 2, dim=1)
    mean_obs = torch.mean(y_return[:, 0, :], dim=1, keepdim=True)
    sst = torch.sum((y_return[:, 0, :] - mean_obs) ** 2, dim=1)

    r2 = 1 - (ssr / sst)
    #print(r2.mean())
    return -r2.mean()


if __name__ == "__main__":
    pass
