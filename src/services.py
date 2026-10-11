"""Streamlit-cached singletons: the dataset, the classifier and the storage object."""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from .config import DATASET_DIR, STATE_FILE
from .dataset import CifarDataset, DatasetUnavailableError
from .model import Classifier, ModelUnavailableError
from .storage import Storage


@st.cache_resource(show_spinner="Preparing the CIFAR-10 dataset (first run downloads about 170 MB)...")
def load_dataset() -> CifarDataset:
    return CifarDataset.load(DATASET_DIR, download=True)


@st.cache_resource(show_spinner="Loading MobileNetV3-Small (first run downloads about 10 MB)...")
def load_classifier() -> Classifier:
    return Classifier.load()


@dataclass(frozen=True)
class DatasetStatus:
    dataset: CifarDataset | None
    needs_download: bool = False
    error: str | None = None


def dataset_status() -> DatasetStatus:
    if not (CifarDataset.is_downloaded(DATASET_DIR) or st.session_state.get("download_requested")):
        return DatasetStatus(None, needs_download=True)
    try:
        return DatasetStatus(load_dataset())
    except DatasetUnavailableError as exc:
        return DatasetStatus(None, needs_download=True, error=str(exc))


def classifier_or_error() -> tuple[Classifier | None, str | None]:
    try:
        return load_classifier(), None
    except ModelUnavailableError as exc:
        return None, str(exc)


def get_storage() -> Storage:
    return Storage(STATE_FILE)
