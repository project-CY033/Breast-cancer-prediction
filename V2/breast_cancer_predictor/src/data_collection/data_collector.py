import os
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tcga_base_url = "https://api.gdc.cancer.gov"
        self.cbioportal_url = "https://www.cbioportal.org/api"
        
    def fetch_tcga_data(self, study_id: str = "BRCA", data_type: str = "mutations") -> pd.DataFrame:
        try:
            endpoint = f"{self.tcga_base_url}/projects/{study_id}"
            response = requests.get(endpoint, timeout=30)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Fetched TCGA {study_id} data: {len(data.get('data', {}).get('hits', []))} records")
            return self._parse_tcga_response(data)
        except requests.RequestException as e:
            logger.warning(f"TCGA API unavailable: {e}. Using simulated data.")
            return self._generate_simulated_data(n_samples=500)
    
    def fetch_cbioportal_data(self, study_id: str = "brca_tcga") -> pd.DataFrame:
        try:
            endpoint = f"{self.cbioportal_url}/studies/{study_id}/mutations"
            params = {"pageSize": 1000}
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return self._parse_cbioportal_response(data)
        except requests.RequestException as e:
            logger.warning(f"cBioPortal API unavailable: {e}. Using simulated data.")
            return self._generate_simulated_data(n_samples=500)
    
    def fetch_ncbi_data(self, query: str = "breast cancer mutation", max_results: int = 100) -> Dict:
        try:
            from Bio import Entrez
            Entrez.email = "research@example.com"
            handle = Entrez.esearch(db="sra", term=query, retmax=max_results)
            results = Entrez.read(handle)
            handle.close()
            logger.info(f"Found {len(results['IdList'])} NCBI records")
            return results
        except Exception as e:
            logger.warning(f"NCBI API unavailable: {e}")
            return {"IdList": []}
    
    def _parse_tcga_response(self, data: Dict) -> pd.DataFrame:
        records = []
        hits = data.get("data", {}).get("hits", [])
        for hit in hits:
            records.append({
                "sample_id": hit.get("submitter_id", ""),
                "project_id": hit.get("project", {}).get("project_id", ""),
                "primary_site": hit.get("primary_site", ""),
                "disease_type": hit.get("disease_type", [])
            })
        return pd.DataFrame(records)
    
    def _parse_cbioportal_response(self, data: List) -> pd.DataFrame:
        records = []
        for item in data:
            records.append({
                "gene_symbol": item.get("gene", {}).get("hugoGeneSymbol", ""),
                "mutation_type": item.get("mutationType", ""),
                "protein_change": item.get("proteinChange", ""),
                "sample_id": item.get("sampleId", "")
            })
        return pd.DataFrame(records)
    
    def _generate_simulated_data(self, n_samples: int = 500) -> pd.DataFrame:
        genes = ["BRCA1", "BRCA2", "TP53", "PIK3CA", "PTEN", "CDH1", 
                 "GATA3", "MAP3K1", "TBX3", "FOXA1", "ESR1", "HER2"]
        mutation_types = ["missense", "nonsense", "frameshift", "silent", "splice_site"]
        
        np.random.seed(42)
        data = {
            "sample_id": [f"TCGA-{np.random.randint(10, 99):02d}-{np.random.randint(1000, 9999)}" 
                         for _ in range(n_samples)],
            "gene": np.random.choice(genes, n_samples),
            "mutation_type": np.random.choice(mutation_types, n_samples, 
                                             p=[0.35, 0.15, 0.20, 0.15, 0.15]),
            "protein_change": [f"{np.random.choice(['p.', 'c.'])}{np.random.randint(1, 2000)}{np.random.choice(['A', 'T', 'G', 'C'])}" 
                             for _ in range(n_samples)],
            "chromosome": np.random.randint(1, 23, n_samples),
            "position": np.random.randint(1000000, 250000000, n_samples),
            "variant_classification": np.random.choice(mutation_types, n_samples),
            "functional_impact_score": np.round(np.random.uniform(0, 1, n_samples), 3),
            "label": np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Generated simulated dataset: {n_samples} samples")
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = "raw_mutations.csv") -> Path:
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        return filepath
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records from {filepath}")
        return df
    
    def load_custom_file(self, filepath: str) -> pd.DataFrame:
        ext = Path(filepath).suffix.lower()
        if ext == ".csv":
            return pd.read_csv(filepath)
        elif ext == ".tsv":
            return pd.read_csv(filepath, sep="\t")
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(filepath)
        elif ext == ".json":
            return pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")


def collect_all_sources(output_dir: str = "data/raw") -> pd.DataFrame:
    collector = DataCollector(output_dir)
    
    tcga_data = collector.fetch_tcga_data()
    cbioportal_data = collector.fetch_cbioportal_data()
    
    combined_data = pd.concat([tcga_data, cbioportal_data], ignore_index=True) \
        if not tcga_data.empty and not cbioportal_data.empty else tcga_data
    
    if combined_data.empty:
        combined_data = collector._generate_simulated_data(n_samples=1000)
    
    filepath = collector.save_data(combined_data)
    return combined_data


if __name__ == "__main__":
    data = collect_all_sources()
    print(f"Collected {len(data)} records")
    print(data.head())
