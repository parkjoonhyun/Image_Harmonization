import torch

def reconLoss(contentA , recontentA):
    loss = torch.nn.MSELoss()
    output = loss(contentA, recontentA)
    output *= 0.4
    return output

def harmonizedLoss(contentB , recontentB):
    loss = torch.nn.MSELoss()
    output = loss(contentB, recontentB)
    return output

def distillLoss(contentA , contentB , contentReferenceAFV, referenceA):
    loss = torch.nn.MSELoss()
    output_1 = loss(contentA , contentB)
    output_2 = loss(contentReferenceAFV , referenceA)
    distLoss = output_1 + output_2
    distLoss *= 0.05
    return distLoss