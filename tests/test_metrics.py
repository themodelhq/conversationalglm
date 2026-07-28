from evaluation.metrics import word_error_rate
def test_wer():
 assert word_error_rate(["the quick brown fox"],["the quick brown fox"])==0
 assert 0<word_error_rate(["the quick brown fox"],["the slow fox"])<=1
