import pytest
import numpy as np
from src.score import compute_pbf_score, compute_pbi_score

def test_pbf_score_perfect_fidelity():
    true_bias = np.array([-1, 0., 1.])
    predicted_bias = np.array([[-1, -1],
                                [0, 0],
                                [1, 1]])
    score = compute_pbf_score(true_bias, predicted_bias)
    assert 1.  == pytest.approx(score)

def test_pbf_score_no_fidelity():
    true_bias = np.array([-1, 0., 1.])
    predicted_bias = np.array([[1, 1],
                                [-1, 1],
                                [-1, -1]])
    score = compute_pbf_score(true_bias, predicted_bias)
    assert 0.1339745962155614 == pytest.approx(score)

def test_pbf_score_lowest_fidelity():
    true_bias = np.array([-1, 1, 1.])
    predicted_bias = np.array([[1, 1],
                                [-1, -1],
                                [-1, -1]])
    score = compute_pbf_score(true_bias, predicted_bias)
    assert 0 == pytest.approx(score)

def test_pbf_score_no_bias():
    true_bias = np.array([-1, 1., 1.])
    predicted_bias = np.array([[0, 0],
                                [0, 0],
                                [0, 0]])
    score = compute_pbf_score(true_bias, predicted_bias)
    assert 0.5 == pytest.approx(score)

def test_pbf_score_no_bias2():
    true_bias = np.array([-1, 0, 1.])
    predicted_bias = np.array([[0, 0],
                                [0, 0],
                                [0, 0]])
    score = compute_pbf_score(true_bias, predicted_bias)
    assert 0.591751709536137 == pytest.approx(score)

def test_pbi_score_perfect_fidelity():
    true_bias = np.array([-1, 0., 1.])
    predicted_bias = np.array([[-1, -1],
                                [0, 0],
                                [1, 1]])
    score = compute_pbi_score(true_bias, predicted_bias)
    assert 0. == pytest.approx(score)

def test_pbi_score_no_fidelity():
    true_bias = np.array([-1, 0., 1.])
    predicted_bias = np.array([[1, 1],
                                [-1, 1],
                                [-1, -1]])
    score = compute_pbi_score(true_bias, predicted_bias)
    assert 1. == pytest.approx(score)

def test_pbi_score_no_fidelity2():
    true_bias = np.array([-1, 0., 1.])
    predicted_bias = np.array([[1, 1, 1],
                                [-1, 1, -1],
                                [-1, -1, -1]])
    score = compute_pbi_score(true_bias, predicted_bias)
    assert 1. == pytest.approx(score)

def test_pbi_score_mixed_fidelity():
    true_bias = np.array([-1, 0., 1.])
    predicted_bias = np.array([[1, 1],
                                [-1, 0],
                                [-1, -1]])
    score = compute_pbi_score(true_bias, predicted_bias)
    assert 0.8333333333333333 == pytest.approx(score)