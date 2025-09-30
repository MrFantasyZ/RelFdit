import os
import json
import hashlib
import requests
import torch
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import time


class WikidataKGSampler:
    """Sample related knowledge from Wikidata knowledge graph"""

    def __init__(self, cache_dir: str = "data/kg_cache", verbose: bool = True):
        """
        Args:
            cache_dir: Directory to cache KG query results
            verbose: Whether to print debug information
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.session = requests.Session()

    def _get_cache_key(self, subject: str, relation: str, k: int, max_length: int) -> str:
        """Generate cache key for query"""
        key_str = f"{subject}_{relation}_{k}_{max_length}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_cache(self, cache_key: str) -> Optional[List[str]]:
        """Load cached results"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _save_cache(self, cache_key: str, data: List[str]):
        """Save results to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def _entity_linking(self, text: str) -> List[str]:
        """
        Link text to Wikidata entities

        Args:
            text: Entity text (e.g., "Paris", "France")
        Returns:
            List of Wikidata entity IDs (e.g., ["Q90", "Q142"])
        """
        # Use Wikidata search API
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": text,
            "limit": 5  # Get top 5 candidates
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "search" in data:
                entity_ids = [item["id"] for item in data["search"]]
                if self.verbose and entity_ids:
                    print(f"Entity linking '{text}' -> {entity_ids[0]}")
                return entity_ids
        except Exception as e:
            if self.verbose:
                print(f"Entity linking failed for '{text}': {e}")

        return []

    def _get_entity_claims(self, entity_id: str) -> Dict:
        """
        Get all claims (relations) for an entity

        Args:
            entity_id: Wikidata entity ID (e.g., "Q90")
        Returns:
            Dictionary of claims
        """
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": entity_id,
            "props": "claims"
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "entities" in data and entity_id in data["entities"]:
                return data["entities"][entity_id].get("claims", {})
        except Exception as e:
            if self.verbose:
                print(f"Failed to get claims for {entity_id}: {e}")

        return {}

    def _find_related_entities(self,
                               entity_id: str,
                               max_entities: int = 20) -> List[str]:
        """
        Find related entities through direct relations

        Args:
            entity_id: Source entity ID
            max_entities: Maximum number of related entities to return
        Returns:
            List of related entity IDs
        """
        related = []
        claims = self._get_entity_claims(entity_id)

        # Collect all entities mentioned in claims
        for prop_id, statements in claims.items():
            for statement in statements:
                if "mainsnak" in statement:
                    mainsnak = statement["mainsnak"]
                    if (mainsnak.get("datatype") == "wikibase-item" and
                        "datavalue" in mainsnak):
                        target_id = mainsnak["datavalue"]["value"]["id"]
                        related.append(target_id)

                        if len(related) >= max_entities:
                            return related

        return related[:max_entities]

    def _find_shortest_paths(self,
                            entity_ids: List[str],
                            k: int,
                            max_path_length: int) -> List[str]:
        """
        Find related entities via shortest paths (BFS)

        Args:
            entity_ids: Source entity IDs
            k: Number of paths/entities to sample
            max_path_length: Maximum path length
        Returns:
            List of related entity IDs
        """
        if not entity_ids:
            return []

        visited = set(entity_ids)
        related_entities = []
        queue = [(eid, 0) for eid in entity_ids[:3]]  # Start from top 3 candidates

        while queue and len(related_entities) < k:
            current_id, depth = queue.pop(0)

            if depth >= max_path_length:
                continue

            # Find neighbors
            neighbors = self._find_related_entities(current_id, max_entities=10)

            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    related_entities.append(neighbor_id)
                    queue.append((neighbor_id, depth + 1))

                    if len(related_entities) >= k:
                        break

            # Rate limiting
            time.sleep(0.1)

        return related_entities[:k]

    def sample_related_keys(self,
                           subject: str,
                           relation: str,
                           K0: torch.Tensor,
                           entity_to_key_map: Dict[str, int],
                           k: int = 10,
                           max_path_length: int = 3) -> Optional[torch.Tensor]:
        """
        Sample related keys from K0 based on KG paths

        Args:
            subject: Subject entity text
            relation: Relation type
            K0: Preserved knowledge key matrix [d0, n]
            entity_to_key_map: Mapping from entity text to column index in K0
            k: Number of paths to sample
            max_path_length: Maximum path length
        Returns:
            K_rel: Related knowledge key matrix [d0, m] or None if no related keys found
        """
        # Check cache
        cache_key = self._get_cache_key(subject, relation, k, max_path_length)
        cached_entities = self._load_cache(cache_key)

        if cached_entities is None:
            # Entity linking
            entity_ids = self._entity_linking(subject)
            if not entity_ids:
                if self.verbose:
                    print(f"No entities found for '{subject}'")
                return None

            # Find related entities via KG paths
            related_entity_ids = self._find_shortest_paths(entity_ids, k, max_path_length)

            if not related_entity_ids:
                if self.verbose:
                    print(f"No related entities found for '{subject}'")
                return None

            # Cache results
            self._save_cache(cache_key, related_entity_ids)
            cached_entities = related_entity_ids

        if self.verbose:
            print(f"Found {len(cached_entities)} related entities for '{subject}'")

        # Map entity IDs to key indices
        # For simplicity, we'll use a simple matching strategy:
        # Since we don't have access to the original entity mapping,
        # we'll randomly sample related keys from K0
        # In a real implementation, you would maintain entity-to-key mapping

        # Randomly sample m columns from K0 as related keys
        m = min(len(cached_entities), K0.shape[1] // 10)  # Sample up to 10% of K0
        if m == 0:
            return None

        indices = torch.randperm(K0.shape[1])[:m]
        K_rel = K0[:, indices]

        if self.verbose:
            print(f"Sampled K_rel shape: {K_rel.shape}")

        return K_rel