from sklearn.metrics import f1_score


def score_function(pred, target):
    score = f1_score(pred, target, average="macro")
    return score
