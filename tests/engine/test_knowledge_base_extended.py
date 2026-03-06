"""Tests for KnowledgeBase negative transfer detection."""

from __future__ import annotations

from vamos.engine.tuning.knowledge_base import KnowledgeBase


def test_negative_transfer_below_percentile():
    random_perfs = [0.5, 0.6, 0.7, 0.8, 0.9]
    assert KnowledgeBase.is_negative_transfer(0.4, random_perfs)


def test_no_negative_transfer_above_percentile():
    random_perfs = [0.5, 0.6, 0.7, 0.8, 0.9]
    assert not KnowledgeBase.is_negative_transfer(0.8, random_perfs)


def test_negative_transfer_empty_random():
    assert not KnowledgeBase.is_negative_transfer(0.1, [])


def test_negative_transfer_custom_percentile():
    random_perfs = [0.1, 0.2, 0.3, 0.4, 0.5]
    # 50th percentile = 0.3
    assert KnowledgeBase.is_negative_transfer(0.2, random_perfs, percentile=50.0)
    assert not KnowledgeBase.is_negative_transfer(0.4, random_perfs, percentile=50.0)


def test_negative_transfer_at_boundary():
    random_perfs = [1.0, 2.0, 3.0, 4.0]
    # 25th percentile = 1.75
    assert KnowledgeBase.is_negative_transfer(1.5, random_perfs)
    assert not KnowledgeBase.is_negative_transfer(2.0, random_perfs)
