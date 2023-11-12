import pytest
import numpy as np
from src.score import compute_pbf_score, compute_pbi_score

def test_pbf_score_perfect_fidelity():
    true_bias = np.array([-1, 0., 1.])
    predicted_bias = np.array([[-1, -1],
                                [0, 0],
                                [1, 1]])
    score = compute_pbf_score(true_bias, predicted_bias)
    assert 0.0 == pytest.approx(score)

def test_pbf_score_no_fidelity():
    true_bias = np.array([-1, 0., 1.])
    predicted_bias = np.array([[1, 1],
                                [-1, 1],
                                [-1, -1]])
    score = compute_pbf_score(true_bias, predicted_bias)
    assert 3.0 == pytest.approx(score)

def test_pbf_score_no_fidelity2():
    true_bias = np.array([-1, 0., 1.])
    predicted_bias = np.array([[1, 1, 1],
                                [-1, 1, -1],
                                [-1, -1, -1]])
    score = compute_pbf_score(true_bias, predicted_bias)
    assert 3.0 == pytest.approx(score)

def test_pbf_score_no_bias():
    true_bias = np.array([-1, 0., 1.])
    predicted_bias = np.array([[0, 0],
                                [0, 0],
                                [0, 0]])
    score = compute_pbf_score(true_bias, predicted_bias)
    assert 1.4142135623730951 == pytest.approx(score)

def test_pbi_score_perfect_fidelity():
    true_bias = np.array([-1, 0., 1.])
    predicted_bias = np.array([[-1, -1],
                                [0, 0],
                                [1, 1]])
    score = compute_pbi_score(true_bias, predicted_bias)
    assert 2.449489742783178 == pytest.approx(score)

def test_pbi_score_no_fidelity():
    true_bias = np.array([-1, 0., 1.])
    predicted_bias = np.array([[1, 1],
                                [-1, 1],
                                [-1, -1]])
    score = compute_pbi_score(true_bias, predicted_bias)
    assert 1.7320508075688772 == pytest.approx(score)

def test_pbi_score_no_fidelity2():
    true_bias = np.array([-1, 0., 1.])
    predicted_bias = np.array([[1, 1, 1],
                                [-1, 1, -1],
                                [-1, -1, -1]])
    score = compute_pbi_score(true_bias, predicted_bias)
    assert 1.7320508075688772 == pytest.approx(score)