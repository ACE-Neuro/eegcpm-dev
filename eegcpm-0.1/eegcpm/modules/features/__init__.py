"""Feature extraction module for predictive modeling."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from eegcpm.pipeline.base import BaseModule, ModuleResult
from eegcpm.data.conn_rois import get_network_indices, CONN_NETWORKS

# Import ERP feature extraction modules
from .erp_features import ERPFeatureModule
from .source_erp_features import SourceERPFeatureModule

logger = logging.getLogger(__name__)

__all__ = [
    "FeatureExtractionModule",
    "ERPFeatureModule",
    "SourceERPFeatureModule",
]


class FeatureExtractionModule(BaseModule):
    """
    Extract features from connectivity and other analysis outputs.

    Features:
    - Connectivity matrix edges (upper triangle)
    - Network-level summaries
    - Graph metrics
    - ERP features
    """

    name = "feature_extraction"
    version = "0.1.0"
    description = "Extract features for prediction"

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config, output_dir)
        self.feature_types = config.get("feature_types", ["edges", "network_means"])
        self.include_graph_metrics = config.get("include_graph_metrics", False)

    def validate_input(self, data: Any) -> bool:
        """Validate input is connectivity dict."""
        return isinstance(data, dict)

    def process(
        self,
        data: Dict[str, np.ndarray],
        subject: Optional[Any] = None,
        **kwargs,
    ) -> ModuleResult:
        """
        Extract features from connectivity matrices.

        Args:
            data: Connectivity matrices dict
            subject: Subject info

        Returns:
            ModuleResult with feature vector
        """
        start_time = time.time()
        output_files = []
        warnings = []

        try:
            features = {}
            feature_names = []

            # Process each connectivity matrix
            for key, matrix in data.items():
                if not isinstance(matrix, np.ndarray):
                    continue
                if matrix.ndim != 2:
                    continue

                # Edge features (upper triangle)
                if "edges" in self.feature_types:
                    edges = self._extract_edges(matrix)
                    edge_names = [f"{key}_edge_{i}" for i in range(len(edges))]
                    features[f"{key}_edges"] = edges
                    feature_names.extend(edge_names)

                # Network-level features
                if "network_means" in self.feature_types:
                    network_features, network_names = self._extract_network_features(
                        matrix, key
                    )
                    features.update(network_features)
                    feature_names.extend(network_names)

                # Graph metrics
                if self.include_graph_metrics:
                    graph_features, graph_names = self._extract_graph_metrics(
                        matrix, key
                    )
                    features.update(graph_features)
                    feature_names.extend(graph_names)

            # Combine all features into single vector
            feature_vector = np.concatenate([
                np.atleast_1d(v) for v in features.values()
                if isinstance(v, (np.ndarray, float, int))
            ])

            # Save features
            subject_id = subject.id if subject else "unknown"
            features_path = self.output_dir / f"{subject_id}_features.npz"
            np.savez(
                features_path,
                features=feature_vector,
                feature_names=feature_names,
                **features,
            )
            output_files.append(features_path)

            return ModuleResult(
                success=True,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                outputs={
                    "data": feature_vector,
                    "features": feature_vector,
                    "feature_names": feature_names,
                    "feature_dict": features,
                },
                output_files=output_files,
                warnings=warnings,
                metadata={
                    "n_features": len(feature_vector),
                    "feature_types": self.feature_types,
                    "n_matrices_processed": len([k for k in data if isinstance(data[k], np.ndarray)]),
                },
            )

        except Exception as e:
            return ModuleResult(
                success=False,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                errors=[str(e)],
            )

    def _extract_edges(self, matrix: np.ndarray) -> np.ndarray:
        """Extract upper triangle of connectivity matrix."""
        return matrix[np.triu_indices(matrix.shape[0], k=1)]

    def _extract_network_features(
        self,
        matrix: np.ndarray,
        prefix: str,
    ) -> tuple:
        """Extract network-level summary features."""
        features = {}
        names = []

        network_indices = get_network_indices()

        # Within-network connectivity
        for network, indices in network_indices.items():
            if len(indices) > 1:
                submatrix = matrix[np.ix_(indices, indices)]
                within = np.mean(submatrix[np.triu_indices(len(indices), k=1)])
                key = f"{prefix}_{network}_within"
                features[key] = within
                names.append(key)

        # Between-network connectivity
        networks = list(network_indices.keys())
        for i, net1 in enumerate(networks):
            for net2 in networks[i + 1:]:
                idx1 = network_indices[net1]
                idx2 = network_indices[net2]
                between = np.mean(matrix[np.ix_(idx1, idx2)])
                key = f"{prefix}_{net1}_{net2}_between"
                features[key] = between
                names.append(key)

        # Global mean
        global_mean = np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)])
        features[f"{prefix}_global_mean"] = global_mean
        names.append(f"{prefix}_global_mean")

        return features, names

    def _extract_graph_metrics(
        self,
        matrix: np.ndarray,
        prefix: str,
    ) -> tuple:
        """Extract graph-theoretic metrics."""
        features = {}
        names = []

        # Node degree (sum of connections)
        threshold = np.percentile(matrix[matrix > 0], 50) if np.any(matrix > 0) else 0
        binary = (matrix > threshold).astype(float)
        degree = np.sum(binary, axis=1)

        features[f"{prefix}_mean_degree"] = np.mean(degree)
        names.append(f"{prefix}_mean_degree")

        features[f"{prefix}_std_degree"] = np.std(degree)
        names.append(f"{prefix}_std_degree")

        # Strength (weighted degree)
        strength = np.sum(matrix, axis=1)
        features[f"{prefix}_mean_strength"] = np.mean(strength)
        names.append(f"{prefix}_mean_strength")

        try:
            import networkx as nx
        except ImportError as exc:
            # networkx is an optional extra (``all``). Record why the graph
            # metrics are absent instead of silently omitting the columns
            # (P1-F11).
            logger.warning(
                "graph metrics (%s_*) skipped: networkx is not installed "
                "(%s); install with `pip install networkx`", prefix, exc)
            return features, names

        # Create graph
        G = nx.from_numpy_array(binary)

        # Clustering coefficient
        clustering = nx.average_clustering(G)
        features[f"{prefix}_clustering"] = clustering
        names.append(f"{prefix}_clustering")

        # Path length is undefined for disconnected graphs (and for graphs with
        # fewer than two nodes); record the reason instead of dropping it
        # silently (P1-F11).
        if nx.is_connected(G):
            try:
                path_length = nx.average_shortest_path_length(G)
            except (nx.NetworkXError, nx.NetworkXPointlessConcept) as exc:
                logger.warning(
                    "graph metric %s_path_length unavailable: %s", prefix, exc)
            else:
                features[f"{prefix}_path_length"] = path_length
                names.append(f"{prefix}_path_length")

        return features, names

    def get_output_spec(self) -> Dict[str, str]:
        return {
            "features": "Feature vector array",
            "feature_names": "List of feature names",
            "feature_dict": "Dict of named features",
        }
